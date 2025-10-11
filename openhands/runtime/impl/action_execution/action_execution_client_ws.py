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
    """

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

        # WebSocket server components
        self._websocket_app: FastAPI | None = None
        self._websocket_server_thread: threading.Thread | None = None
        self._websocket_connection: WebSocket | None = None
        self._websocket_port: int = 3001
        self._websocket_lock = threading.Lock()
        self._pending_responses: dict[str, asyncio.Future] = {}

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

    @property
    def action_execution_server_url(self) -> str:
        raise NotImplementedError('Action execution server URL is not implemented')

    def start_websocket_server(self) -> None:
        """Start the WebSocket server in a separate thread."""
        def run_server():
            self._websocket_app = FastAPI()

            @self._websocket_app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                self.log('info', 'ActionExecutionServer connected via WebSocket')

                with self._websocket_lock:
                    self._websocket_connection = websocket

                try:
                    while True:
                        # Receive messages from ActionExecutionServer
                        data = await websocket.receive_text()
                        message = json.loads(data)

                        # Handle response messages
                        if message.get('type') == 'response':
                            request_id = message.get('request_id')
                            if request_id in self._pending_responses:
                                future = self._pending_responses.pop(request_id)
                                if not future.done():
                                    future.set_result(message.get('data'))

                except WebSocketDisconnect:
                    self.log('warning', 'ActionExecutionServer disconnected from WebSocket')
                    with self._websocket_lock:
                        self._websocket_connection = None
                except Exception as e:
                    self.log('error', f'WebSocket error: {e}')
                    with self._websocket_lock:
                        self._websocket_connection = None

            # Run the FastAPI app with uvicorn
            config = uvicorn.Config(
                self._websocket_app,
                host="0.0.0.0",
                port=self._websocket_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            asyncio.run(server.serve())

        self._websocket_server_thread = threading.Thread(target=run_server, daemon=True)
        self._websocket_server_thread.start()
        self.log('info', f'WebSocket server started on 0.0.0.0:{self._websocket_port}')

    async def _send_action_via_websocket(self, action: Action) -> Observation:
        """Send an action to the ActionExecutionServer via WebSocket."""
        import uuid

        if self._websocket_connection is None:
            raise RuntimeError('WebSocket connection not established')

        request_id = str(uuid.uuid4())

        # Create a future for the response
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
                json_string = json.dumps(event_to_dict(action), ensure_ascii=False, indent=4)
                self.log('info', f'Sending action via WebSocket: {json_string}')

                # Use asyncio to run the WebSocket send operation
                loop = asyncio.new_event_loop()
                try:
                    obs = loop.run_until_complete(self._send_action_via_websocket(action))
                finally:
                    loop.close()

                self.log('info', f'Received observation from WebSocket: {event_to_dict(obs)}')
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

        # Close WebSocket connection
        if self._websocket_connection is not None:
            try:
                asyncio.run(self._websocket_connection.close())
            except Exception as e:
                self.log('warning', f'Error closing WebSocket connection: {e}')
            self._websocket_connection = None

        # Note: We can't easily stop the uvicorn server thread,
        # but it's marked as daemon so it will be cleaned up on process exit

        self.session.close()
