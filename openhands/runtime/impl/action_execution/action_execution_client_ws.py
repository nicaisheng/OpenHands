import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpcore
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
import uvicorn

from openhands.core.config import OpenHandsConfig
from openhands.core.config.mcp_config import (
    MCPConfig,
    MCPSSEServerConfig,
    MCPStdioServerConfig,
)
from openhands.core.exceptions import (
    AgentRuntimeTimeoutError,
)
from openhands.events import EventStream
from openhands.events.action import (
    ActionConfirmationStatus,
    AgentThinkAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.action.action import Action
from openhands.events.action.files import FileEditSource
from openhands.events.action.mcp import MCPAction
from openhands.events.observation import (
    AgentThinkObservation,
    ErrorObservation,
    NullObservation,
    Observation,
    UserRejectObservation,
)
from openhands.events.serialization import event_to_dict, observation_from_dict
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils.request import send_request
from openhands.runtime.utils.system_stats import update_last_execution_time
from openhands.utils.http_session import HttpSession
from openhands.utils.tenacity_stop import stop_if_should_exit


def _is_retryable_error(exception):
    return isinstance(
        exception, (httpx.RemoteProtocolError, httpcore.RemoteProtocolError)
    )


class ActionExecutionClient(Runtime):
    """Base class for runtimes that interact with the action execution server.

    This class now acts as a WebSocket server listening on 0.0.0.0:3001,
    and ActionExecutionServer connects to it as a WebSocket client.

    All instances of this class share a single WebSocket server and event loop
    for efficiency and resource management.
    """

    # Class-level shared components (shared by ALL instances)
    _class_websocket_app: FastAPI | None = None
    _class_websocket_server_thread: threading.Thread | None = None
    _class_websocket_port: int = 3001
    _class_websocket_lock = threading.Lock()
    _class_shared_loop: asyncio.AbstractEventLoop | None = None
    _class_loop_thread: threading.Thread | None = None
    _class_server_started: bool = False
    _class_instance_count: int = 0  # Track number of active instances

    # Registry to map conversation_id to runtime instance (for WebSocket endpoint to find the right instance)
    _class_runtime_registry: dict[str, 'ActionExecutionClient'] = {}

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Any | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    ):
        self.session = HttpSession()
        self.action_semaphore = threading.Semaphore(1)  # Ensure one action at a time
        self._runtime_closed: bool = False
        self._vscode_token: str | None = None  # initial dummy value
        self._last_updated_mcp_stdio_servers: list[MCPStdioServerConfig] = []

        # Instance-level WebSocket connection and pending responses
        self._websocket_connection: WebSocket | None = None
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._connection_ready = threading.Event()  # Signal when WebSocket is connected

        self._log_level = 'info'

        # Register this instance and increment count
        with ActionExecutionClient._class_websocket_lock:
            ActionExecutionClient._class_instance_count += 1

        super().__init__(
            config,
            event_stream,
            llm_registry,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
            user_id,
            git_provider_tokens,
        )

        # Register in the runtime registry after sid is set
        with ActionExecutionClient._class_websocket_lock:
            ActionExecutionClient._class_runtime_registry[self.sid] = self

    @property
    def action_execution_server_url(self) -> str:
        raise NotImplementedError('Action execution server URL is not implemented')

    @classmethod
    def _start_shared_event_loop(cls) -> None:
        """Start a shared event loop in a separate thread for performance optimization.

        This is a class method that starts a single event loop shared by all instances.
        """
        import threading as thread_module

        # Check if already started (thread-safe)
        if cls._class_shared_loop is not None and cls._class_shared_loop.is_running():
            print(f'[ActionExecutionClient] Event loop already running (thread {thread_module.current_thread().name})')
            return  # Already running

        # Double-check with lock to prevent race condition
        with cls._class_websocket_lock:
            if cls._class_shared_loop is not None and cls._class_shared_loop.is_running():
                print(f'[ActionExecutionClient] Event loop already running after lock check (thread {thread_module.current_thread().name})')
                return  # Already running

            print(f'[ActionExecutionClient] Starting shared event loop (thread {thread_module.current_thread().name})')

            def run_loop():
                cls._class_shared_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(cls._class_shared_loop)
                print('[ActionExecutionClient] Shared event loop thread started')
                cls._class_shared_loop.run_forever()

            cls._class_loop_thread = thread_module.Thread(target=run_loop, daemon=True, name='SharedEventLoop')
            cls._class_loop_thread.start()

        # Wait for loop to be ready (outside the lock)
        import time
        max_wait = 5
        waited = 0
        while cls._class_shared_loop is None and waited < max_wait:
            time.sleep(0.01)
            waited += 0.01

        if cls._class_shared_loop is None:
            raise RuntimeError('Failed to start shared event loop')

        print('[ActionExecutionClient] Shared event loop ready')

    @classmethod
    def start_websocket_server(cls) -> None:
        """Start the WebSocket server in a separate thread.

        This is a class method that starts a single server shared by all instances.
        It's safe to call multiple times - it will only start once.
        """
        import threading as thread_module

        # Fast path: check without lock first
        if cls._class_server_started:
            print(f'[ActionExecutionClient] WebSocket server already started (fast path, thread {thread_module.current_thread().name})')
            return  # Server already started

        # Slow path: acquire lock and check again (double-check locking)
        with cls._class_websocket_lock:
            if cls._class_server_started:
                print(f'[ActionExecutionClient] WebSocket server already started (slow path, thread {thread_module.current_thread().name})')
                return  # Server already started (another thread might have started it)

            print(f'[ActionExecutionClient] Starting WebSocket server (thread {thread_module.current_thread().name})')
            # Mark as started before actually starting to prevent other threads from proceeding
            cls._class_server_started = True

        # Start shared event loop first (outside the lock to avoid holding it too long)
        cls._start_shared_event_loop()

        def run_server():
            cls._class_websocket_app = FastAPI()

            @cls._class_websocket_app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()

                # Wait for the first message to get conversation_id
                runtime_instance = None
                conversation_id = None

                try:
                    init_data = await websocket.receive_text()
                    init_message = json.loads(init_data)

                    if init_message.get('type') != 'register':
                        await websocket.close(code=1003, reason="First message must be registration")
                        return

                    conversation_id = init_message.get('conversation_id')

                    # Find the runtime instance for this conversation
                    with cls._class_websocket_lock:
                        # If no conversation_id provided, try to use the first waiting runtime
                        if not conversation_id:
                            # Find first runtime without a connection
                            for sid, runtime in cls._class_runtime_registry.items():
                                if runtime._websocket_connection is None:
                                    conversation_id = sid
                                    runtime_instance = runtime
                                    break

                            if not conversation_id:
                                await websocket.close(code=1003, reason="No conversation_id provided and no waiting runtimes available")
                                return
                        else:
                            runtime_instance = cls._class_runtime_registry.get(conversation_id)
                            if not runtime_instance:
                                await websocket.close(code=1003, reason=f"No runtime found for conversation {conversation_id}")
                                return

                        # Set the WebSocket connection on the runtime instance
                        runtime_instance._websocket_connection = websocket
                        runtime_instance._connection_ready.set()

                    print(f'[ActionExecutionClient] Conversation [{conversation_id}] connected via WebSocket. Total registered: {len(cls._class_runtime_registry)}')

                    # Send acknowledgment with the actual conversation_id
                    await websocket.send_text(json.dumps({'type': 'registered', 'conversation_id': conversation_id}))

                except Exception as e:
                    print(f'[ActionExecutionClient] Error during conversation registration: {e}')
                    await websocket.close(code=1011, reason=str(e))
                    return

                try:
                    while True:
                        # Receive messages from ActionExecutionServer
                        data = await websocket.receive_text()
                        message = json.loads(data)

                        # Handle response messages
                        if message.get('type') == 'response':
                            request_id = message.get('request_id')
                            if runtime_instance and request_id in runtime_instance._pending_responses:
                                future = runtime_instance._pending_responses.pop(request_id)
                                if not future.done():
                                    # Set result in the future's event loop (shared loop)
                                    # This is critical: we're in uvicorn's event loop but the future
                                    # belongs to the shared loop, so we must use call_soon_threadsafe
                                    cls._class_shared_loop.call_soon_threadsafe(
                                        future.set_result, message.get('data')
                                    )

                except WebSocketDisconnect:
                    print(f'[ActionExecutionClient] Conversation [{conversation_id}] disconnected from WebSocket')
                    if runtime_instance:
                        runtime_instance._websocket_connection = None
                        runtime_instance._connection_ready.clear()
                except Exception as e:
                    print(f'[ActionExecutionClient] WebSocket error for conversation [{conversation_id}]: {e}')
                    if runtime_instance:
                        runtime_instance._websocket_connection = None
                        runtime_instance._connection_ready.clear()

            # Run the FastAPI app with uvicorn
            config = uvicorn.Config(
                cls._class_websocket_app,
                host="0.0.0.0",
                port=cls._class_websocket_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            asyncio.run(server.serve())

        cls._class_websocket_server_thread = threading.Thread(target=run_server, daemon=True)
        cls._class_websocket_server_thread.start()
        print(f'[ActionExecutionClient] WebSocket server started on 0.0.0.0:{cls._class_websocket_port}')

    def is_connected(self) -> bool:
        """Check if this runtime's WebSocket is connected."""
        return self._websocket_connection is not None

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for WebSocket connection to be established.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        return self._connection_ready.wait(timeout=timeout)

    async def _send_action_via_websocket(self, action: Action) -> Observation:
        """Send an action to the ActionExecutionServer via WebSocket.

        Args:
            action: The action to send

        Returns:
            Observation from the action execution
        """
        import uuid

        # Check if WebSocket is connected
        if self._websocket_connection is None:
            raise RuntimeError(f'WebSocket not connected for runtime {self.sid}')

        request_id = str(uuid.uuid4())

        # Create a future for the response in the shared event loop
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_responses[request_id] = future

        # Send the action
        message = {
            'type': 'action',
            'request_id': request_id,
            'data': event_to_dict(action)
        }

        try:
            await self._websocket_connection.send_text(json.dumps(message))
            self.log('debug', f'Sent action via WebSocket, request_id: {request_id}')

            # Wait for the response with timeout
            timeout = getattr(action, 'timeout', None) or self.config.sandbox.timeout
            response_data = await asyncio.wait_for(future, timeout=timeout + 5)

            # Convert response back to Observation
            obs = observation_from_dict(response_data)
            obs._cause = action.id  # type: ignore[attr-defined]
            return obs

        except asyncio.TimeoutError:
            self._pending_responses.pop(request_id, None)
            raise AgentRuntimeTimeoutError(
                f'Runtime failed to return response before timeout of {timeout}s'
            )
        except Exception as e:
            self._pending_responses.pop(request_id, None)
            raise RuntimeError(f'Failed to send action via WebSocket: {e}')

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(5) | stop_if_should_exit(),
        wait=wait_exponential(multiplier=1, min=4, max=15),
    )
    def _send_action_server_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Send a request to the action execution server.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to send the request to
            **kwargs: Additional arguments to pass to requests.request()

        Returns:
            Response from the server

        Raises:
            AgentRuntimeError: If the request fails
        """
        return send_request(self.session, method, url, **kwargs)

    def check_if_alive(self) -> None:
        response = self._send_action_server_request(
            'GET',
            f'{self.action_execution_server_url}/alive',
            timeout=5,
        )
        assert response.is_closed

    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the sandbox.

        If path is None, list files in the sandbox's initial working directory (e.g., /workspace).
        """
        try:
            data = {}
            if path is not None:
                data['path'] = path

            response = self._send_action_server_request(
                'POST',
                f'{self.action_execution_server_url}/list_files',
                json=data,
                timeout=10,
            )
            assert response.is_closed
            response_json = response.json()
            assert isinstance(response_json, list)
            return response_json
        except httpx.TimeoutException:
            raise TimeoutError('List files operation timed out')

    def copy_from(self, path: str) -> Path:
        """Zip all files in the sandbox and return as a stream of bytes."""
        try:
            params = {'path': path}
            with self.session.stream(
                'GET',
                f'{self.action_execution_server_url}/download_files',
                params=params,
                timeout=30,
            ) as response:
                with tempfile.NamedTemporaryFile(
                    suffix='.zip', delete=False
                ) as temp_file:
                    for chunk in response.iter_bytes():
                        temp_file.write(chunk)
                    temp_file.flush()
                    return Path(temp_file.name)
        except httpx.TimeoutException:
            raise TimeoutError('Copy operation timed out')

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        if not os.path.exists(host_src):
            raise FileNotFoundError(f'Source file {host_src} does not exist')

        temp_zip_path: str | None = None  # Define temp_zip_path outside the try block

        try:
            params = {'destination': sandbox_dest, 'recursive': str(recursive).lower()}
            file_to_upload = None
            upload_data = {}

            if recursive:
                # Create and write the zip file inside the try block
                with tempfile.NamedTemporaryFile(
                    suffix='.zip', delete=False
                ) as temp_zip:
                    temp_zip_path = temp_zip.name

                try:
                    with ZipFile(temp_zip_path, 'w') as zipf:
                        for root, _, files in os.walk(host_src):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(
                                    file_path, os.path.dirname(host_src)
                                )
                                zipf.write(file_path, arcname)

                    self.log(
                        'debug',
                        f'Opening temporary zip file for upload: {temp_zip_path}',
                    )
                    file_to_upload = open(temp_zip_path, 'rb')
                    upload_data = {'file': file_to_upload}
                except Exception as e:
                    # Ensure temp file is cleaned up if zipping fails
                    if temp_zip_path and os.path.exists(temp_zip_path):
                        os.unlink(temp_zip_path)
                    raise e  # Re-raise the exception after cleanup attempt
            else:
                file_to_upload = open(host_src, 'rb')
                upload_data = {'file': file_to_upload}

            params = {'destination': sandbox_dest, 'recursive': str(recursive).lower()}

            response = self._send_action_server_request(
                'POST',
                f'{self.action_execution_server_url}/upload_file',
                files=upload_data,
                params=params,
                timeout=300,
            )
            self.log(
                'debug',
                f'Copy completed: host:{host_src} -> runtime:{sandbox_dest}. Response: {response.text}',
            )
        finally:
            if file_to_upload:
                file_to_upload.close()

            # Cleanup the temporary zip file if it was created
            if temp_zip_path and os.path.exists(temp_zip_path):
                try:
                    os.unlink(temp_zip_path)
                except Exception as e:
                    self.log(
                        'error',
                        f'Failed to delete temporary zip file {temp_zip_path}: {e}',
                    )

    def get_vscode_token(self) -> str:
        if self.vscode_enabled and self.runtime_initialized:
            if self._vscode_token is not None:  # cached value
                return self._vscode_token
            response = self._send_action_server_request(
                'GET',
                f'{self.action_execution_server_url}/vscode/connection_token',
                timeout=10,
            )
            response_json = response.json()
            assert isinstance(response_json, dict)
            if response_json['token'] is None:
                return ''
            self._vscode_token = response_json['token']
            return response_json['token']
        else:
            return ''

    def send_action_for_execution(self, action: Action) -> Observation:
        if (
            isinstance(action, FileEditAction)
            and action.impl_source == FileEditSource.LLM_BASED_EDIT
        ):
            return self.llm_based_edit(action)

        # set timeout to default if not set
        if action.timeout is None:
            if isinstance(action, CmdRunAction) and action.blocking:
                raise RuntimeError('Blocking command with no timeout set')
            # We don't block the command if this is a default timeout action
            action.set_hard_timeout(self.config.sandbox.timeout, blocking=False)

        with self.action_semaphore:
            if not action.runnable:
                if isinstance(action, AgentThinkAction):
                    return AgentThinkObservation('Your thought has been logged.')
                return NullObservation('')
            if (
                hasattr(action, 'confirmation_state')
                and action.confirmation_state
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                return NullObservation('')
            action_type = action.action  # type: ignore[attr-defined]
            if action_type not in ACTION_TYPE_TO_CLASS:
                raise ValueError(f'Action {action_type} does not exist.')
            if not hasattr(self, action_type):
                return ErrorObservation(
                    f'Action {action_type} is not supported in the current runtime.',
                    error_id='AGENT_ERROR$BAD_ACTION',
                )
            if (
                getattr(action, 'confirmation_state', None)
                == ActionConfirmationStatus.REJECTED
            ):
                return UserRejectObservation(
                    'Action has been rejected by the user! Waiting for further user input.'
                )

            assert action.timeout is not None

            try:
                # Optimize: Only format JSON for logging when needed, without indentation
                if self._log_level == 'debug':
                    json_string = json.dumps(event_to_dict(action), ensure_ascii=False)
                    self.log('debug', f'Sending action via WebSocket: {json_string}')
                else:
                    self.log('info', f'Sending action via WebSocket: {action.action}')

                # Use shared event loop for better performance
                if ActionExecutionClient._class_shared_loop is None:
                    raise RuntimeError('Shared event loop not initialized. Call start_websocket_server() first.')

                # Submit coroutine to shared loop and wait for result
                future = asyncio.run_coroutine_threadsafe(
                    self._send_action_via_websocket(action),
                    ActionExecutionClient._class_shared_loop
                )

                # Wait for result with timeout
                timeout = action.timeout + 5
                obs = future.result(timeout=timeout)

                if self._log_level == 'debug':
                    self.log('debug', f'Received observation from WebSocket: {event_to_dict(obs)}')
                else:
                    self.log('info', f'Received observation from WebSocket: {type(obs).__name__}')

                if getattr(action, 'hidden', False):
                    obs.extras['hidden'] = True  # type: ignore[attr-defined]
                return obs
            except Exception as e:
                self.log('error', f'Error sending action via WebSocket: {e}')
                raise
            finally:
                update_last_execution_time()

    def run(self, action: CmdRunAction) -> Observation:
        return self.send_action_for_execution(action)

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        return self.send_action_for_execution(action)

    def read(self, action: FileReadAction) -> Observation:
        return self.send_action_for_execution(action)

    def write(self, action: FileWriteAction) -> Observation:
        return self.send_action_for_execution(action)

    def edit(self, action: FileEditAction) -> Observation:
        return self.send_action_for_execution(action)

    def browse(self, action: BrowseURLAction) -> Observation:
        return self.send_action_for_execution(action)

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return self.send_action_for_execution(action)

    def get_mcp_config(
        self, extra_stdio_servers: list[MCPStdioServerConfig] | None = None
    ) -> MCPConfig:
        import sys

        # Check if we're on Windows - MCP is disabled on Windows
        if sys.platform == 'win32':
            # Return empty MCP config on Windows
            self.log('debug', 'MCP is disabled on Windows, returning empty config')
            return MCPConfig(sse_servers=[], stdio_servers=[])

        # Add the runtime as another MCP server
        updated_mcp_config = self.config.mcp.model_copy()

        # Get current stdio servers
        current_stdio_servers: list[MCPStdioServerConfig] = list(
            updated_mcp_config.stdio_servers
        )
        if extra_stdio_servers:
            current_stdio_servers.extend(extra_stdio_servers)

        # Check if there are any new servers using the __eq__ operator
        new_servers = [
            server
            for server in current_stdio_servers
            if server not in self._last_updated_mcp_stdio_servers
        ]

        self.log(
            'debug',
            f'adding {len(new_servers)} new stdio servers to MCP config: {new_servers}',
        )

        # Only send update request if there are new servers
        if new_servers:
            # Use a union of current servers and last updated servers for the update
            # This ensures we don't lose any servers that might be missing from either list
            combined_servers = current_stdio_servers.copy()
            for server in self._last_updated_mcp_stdio_servers:
                if server not in combined_servers:
                    combined_servers.append(server)

            stdio_tools = [
                server.model_dump(mode='json') for server in combined_servers
            ]
            stdio_tools.sort(key=lambda x: x.get('name', ''))  # Sort by server name

            self.log(
                'debug',
                f'Updating MCP server with {len(new_servers)} new stdio servers (total: {len(combined_servers)})',
            )
            # response = self._send_action_server_request(
            #     'POST',
            #     f'{self.action_execution_server_url}/update_mcp_server',
            #     json=stdio_tools,
            #     timeout=60,
            # )
            # result = response.json()
            # if response.status_code != 200:
            #     self.log('warning', f'Failed to update MCP server: {response.text}')
            # else:
            #     if result.get('router_error_log'):
            #         self.log(
            #             'warning',
            #             f'Some MCP servers failed to be added: {result["router_error_log"]}',
            #         )

            #     # Update our cached list with combined servers after successful update
            #     self._last_updated_mcp_stdio_servers = combined_servers.copy()
            #     self.log(
            #         'debug',
            #         f'Successfully updated MCP stdio servers, now tracking {len(combined_servers)} servers',
            #     )
            self.log(
                'info',
                f'Updated MCP config: {updated_mcp_config.sse_servers}',
            )
        else:
            self.log('debug', 'No new stdio servers to update')

        if len(self._last_updated_mcp_stdio_servers) > 0:
            # We should always include the runtime as an MCP server whenever there's > 0 stdio servers
            updated_mcp_config.sse_servers.append(
                MCPSSEServerConfig(
                    url=self.action_execution_server_url.rstrip('/') + '/mcp/sse',
                    api_key=self.session_api_key,
                )
            )

        return updated_mcp_config

    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        import sys

        from openhands.events.observation import ErrorObservation

        # Check if we're on Windows - MCP is disabled on Windows
        if sys.platform == 'win32':
            self.log('info', 'MCP functionality is disabled on Windows')
            return ErrorObservation('MCP functionality is not available on Windows')

        # Import here to avoid circular imports
        from openhands.mcp.utils import call_tool_mcp as call_tool_mcp_handler
        from openhands.mcp.utils import create_mcp_clients

        # Get the updated MCP config
        updated_mcp_config = self.get_mcp_config()
        self.log(
            'debug',
            f'Creating MCP clients with servers: {updated_mcp_config.sse_servers}',
        )

        # Create clients for this specific operation
        mcp_clients = await create_mcp_clients(
            updated_mcp_config.sse_servers, updated_mcp_config.shttp_servers, self.sid
        )

        # Call the tool and return the result
        # No need for try/finally since disconnect() is now just resetting state
        result = await call_tool_mcp_handler(mcp_clients, action)
        return result

    def close(self) -> None:
        # Make sure we don't close the session multiple times
        # Can happen in evaluation
        if self._runtime_closed:
            return
        self._runtime_closed = True

        # Close this instance's WebSocket connection
        if self._websocket_connection is not None:
            try:
                asyncio.run(self._websocket_connection.close())
                self.log('info', f'Closed WebSocket connection for runtime {self.sid}')
            except Exception as e:
                self.log('warning', f'Error closing WebSocket connection: {e}')
            self._websocket_connection = None
            self._connection_ready.clear()

        # Unregister from runtime registry and decrement instance count
        should_close_server = False
        with ActionExecutionClient._class_websocket_lock:
            ActionExecutionClient._class_runtime_registry.pop(self.sid, None)
            ActionExecutionClient._class_instance_count -= 1
            # Only close server if this is the last instance
            if ActionExecutionClient._class_instance_count <= 0:
                should_close_server = True

        if should_close_server:
            self.log('info', 'Last runtime instance closing - shutting down shared WebSocket server')

            # Stop shared event loop
            if ActionExecutionClient._class_shared_loop is not None and ActionExecutionClient._class_shared_loop.is_running():
                ActionExecutionClient._class_shared_loop.call_soon_threadsafe(ActionExecutionClient._class_shared_loop.stop)
                self.log('debug', 'Stopped shared event loop')

            # Reset server state
            with ActionExecutionClient._class_websocket_lock:
                ActionExecutionClient._class_server_started = False

            # Note: We can't easily stop the uvicorn server thread,
            # but it's marked as daemon so it will be cleaned up on process exit
        else:
            self.log('info', f'Runtime instance closing - {ActionExecutionClient._class_instance_count} instances remain')

        self.session.close()
