"""This runtime runs the action_execution_server directly on the local machine without cowizard."""

import os
import sys
import threading
from typing import Callable

import httpx
import tenacity

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import AgentRuntimeDisconnectedError
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
    Action,
)
from openhands.events.observation import (
    Observation,
)
from openhands.events.serialization import event_to_dict, observation_from_dict
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.tenacity_stop import stop_if_should_exit


def get_user_info() -> tuple[int, str | None]:
    """Get user ID and username in a cross-platform way."""
    username = os.getenv('USER')
    if sys.platform == 'win32':
        # On Windows, we don't use user IDs the same way
        # Return a default value that won't cause issues
        return 1000, username
    else:
        # On Unix systems, use os.getuid()
        return os.getuid(), username

class CowizardRuntime(ActionExecutionClient):

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable[[str, RuntimeStatus, str], None] | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    ) -> None:
        self.is_windows = sys.platform == 'win32'

        self.config = config
        self._user_id, self._username = get_user_info()

        # Initialize these values to be set in connect()
        self._temp_workspace: str | None = None
        self._execution_server_port = 8000
        self._vscode_port = -1
        self._app_ports: list[int] = []

        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._execution_server_port}'
        self.status_callback = status_callback
        self._log_thread_exit_event = threading.Event()  # Add exit event

        # Initialize the action_execution_server
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

        # If there is an API key in the environment we use this in requests to the runtime
        session_api_key = os.getenv('SESSION_API_KEY')
        self._session_api_key: str | None = None
        if session_api_key:
            self.session.headers['X-Session-API-Key'] = session_api_key
            self._session_api_key = session_api_key

    @property
    def session_api_key(self) -> str | None:
        return self._session_api_key

    @property
    def action_execution_server_url(self) -> str:
        return self.api_url

    async def clone_or_init_repo(
        self,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None,
        selected_repository: str | None,
        selected_branch: str | None,
    ) -> str:
        pass

    async def connect(self) -> None:
        """Start the action_execution_server on the local machine or connect to an existing one."""
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        # API URL for the server
        api_url = f'{self.config.sandbox.local_runtime_url}:{self._execution_server_port}'
        self.api_url = api_url

        self.log('info', f'Waiting for server to become ready at {self.api_url}...')
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        # await call_sync_from_async(self._wait_until_alive)

        if not self.attach_to_existing:
            self.set_runtime_status(RuntimeStatus.READY)
        self._runtime_initialized = True

    @classmethod
    def setup(cls, config: OpenHandsConfig, headless_mode: bool = False):
        pass

    @tenacity.retry(
        wait=tenacity.wait_fixed(2),
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        before_sleep=lambda retry_state: logger.debug(
            f'Waiting for server to be ready... (attempt {retry_state.attempt_number})'
        ),
    )
    def _wait_until_alive(self) -> bool:
        """Wait until the server is ready to accept requests."""
        try:
            response = self.session.get(f'{self.api_url}/alive')
            response.raise_for_status()
            return True
        except Exception as e:
            self.log('debug', f'Server not ready yet: {e}')
            raise

    async def execute_action(self, action: Action) -> Observation:
        """Execute an action by sending it to the server."""
        if not self.runtime_initialized:
            raise AgentRuntimeDisconnectedError('Runtime not initialized')

        with self.action_semaphore:
            try:
                response = await call_sync_from_async(
                    lambda: self.session.post(
                        f'{self.api_url}/execute_action',
                        json={'action': event_to_dict(action)},
                    )
                )

                return observation_from_dict(response.json())
            except httpx.NetworkError:
                raise AgentRuntimeDisconnectedError('Server connection lost')

    def close(self) -> None:
        super().close()

    @classmethod
    async def delete(cls, conversation_id: str) -> None:
        pass

    @property
    def runtime_url(self) -> str:
        runtime_url = os.getenv('RUNTIME_URL')
        if runtime_url:
            return runtime_url

        # TODO: This could be removed if we had a straightforward variable containing the RUNTIME_URL in the K8 env.
        runtime_url_pattern = os.getenv('RUNTIME_URL_PATTERN')
        runtime_id = os.getenv('RUNTIME_ID')
        if runtime_url_pattern and runtime_id:
            runtime_url = runtime_url_pattern.format(runtime_id=runtime_id)
            return runtime_url

        # Fallback to localhost
        return self.config.sandbox.local_runtime_url

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None
        vscode_url = self._create_url('vscode', self._vscode_port)
        return f'{vscode_url}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'

    @property
    def web_hosts(self) -> dict[str, int]:
        hosts: dict[str, int] = {}
        for index, port in enumerate(self._app_ports):
            url = self._create_url(f'work-{index + 1}', port)
            hosts[url] = port
        return hosts
