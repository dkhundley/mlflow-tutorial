"""Local MLflow server helpers for the traditional ML tutorial.

These helpers wrap the command-line server calls used in the notebook. Keeping
them in Python gives the tests a way to start, invoke, and stop local MLflow
servers without shell-specific notebook magics.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from traditional_ml.config import MlflowLocalConfig


@dataclass(frozen=True)
class LocalServer:
    """Metadata for a local server process started by the tutorial.

    The subprocess handle is useful when the current Python process started the
    server. The PID file is useful when a notebook cell failed and only the PID
    file remains for cleanup.
    """

    name: str
    port: int
    pid_file: Path
    log_file: Path
    process: subprocess.Popen[Any]

    @property
    def url(self) -> str:
        # All tutorial servers bind to localhost only.
        return f'http://127.0.0.1:{self.port}'

    def stop(self, timeout_seconds: int = 10) -> None:
        # Provide a small convenience method so notebook cells can call
        # `server.stop()` instead of remembering the PID-file details.
        stop_process(
            process = self.process,
            pid_file = self.pid_file,
            timeout_seconds = timeout_seconds,
        )


def find_free_port() -> int:
    """Find an available local TCP port.

    Tests use dynamic ports to avoid conflicts with a learner's notebook servers
    on the default MLflow ports.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Binding to port 0 asks the OS to choose an available port.
        server_socket.bind(('127.0.0.1', 0))
        return int(server_socket.getsockname()[1])


def start_tracking_server(
    config: MlflowLocalConfig,
    port: int = 5001,
    name: str = 'tracking server',
    pid_file: Path = Path('/tmp/mlflow_local_server.pid'),
    log_file: Path = Path('/tmp/mlflow_local_server.log'),
) -> LocalServer:
    """Start a local MLflow tracking server for the repo-root backend.

    This is the programmatic version of:
    `mlflow server --backend-store-uri ... --default-artifact-root ...`
    """

    config.ensure_local_paths()
    # If the notebook previously failed before reaching its cleanup cell, the
    # PID file may still point at a live process. Clear it before starting again.
    stop_pid_file(pid_file)
    process = _start_process(
        args = [
            sys.executable,
            '-m',
            'mlflow',
            'server',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--backend-store-uri',
            config.backend_store_uri,
            '--default-artifact-root',
            str(config.artifact_root),
            '--serve-artifacts',
        ],
        pid_file = pid_file,
        log_file = log_file,
    )
    # The health endpoint tells us the tracking server is ready before the next
    # cell tries to query model registry metadata through it.
    wait_for_url(f'http://127.0.0.1:{port}/health')
    return LocalServer(
        name = name,
        port = port,
        pid_file = pid_file,
        log_file = log_file,
        process = process,
    )


def start_model_server(
    model_uri: str,
    tracking_uri: str,
    port: int = 5002,
    name: str = 'model server',
    pid_file: Path = Path('/tmp/mlflow_model_server.pid'),
    log_file: Path = Path('/tmp/mlflow_model_server.log'),
) -> LocalServer:
    """Start a local MLflow model serving endpoint.

    The model server needs a tracking URI so model registry URIs such as
    `models:/iris_model/1` can be resolved from the local tracking server.
    """

    stop_pid_file(pid_file)
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = tracking_uri
    process = _start_process(
        args = [
            sys.executable,
            '-m',
            'mlflow',
            'models',
            'serve',
            '-m',
            model_uri,
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--no-conda',
        ],
        pid_file = pid_file,
        log_file = log_file,
        env = env,
    )
    # The scoring server exposes `/ping` once the model has loaded.
    wait_for_url(f'http://127.0.0.1:{port}/ping')
    return LocalServer(
        name = name,
        port = port,
        pid_file = pid_file,
        log_file = log_file,
        process = process,
    )


def invoke_model_endpoint(
    port: int,
    payload: dict[str, Any],
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """Invoke a local MLflow model endpoint and return the JSON response.

    MLflow scoring servers accept JSON payloads at `/invocations`. The payload
    shape depends on the model signature: the Iris model uses `inputs`, while
    the router uses `dataframe_records`.
    """

    response = requests.post(
        f'http://127.0.0.1:{port}/invocations',
        json = payload,
        timeout = timeout_seconds,
    )
    # Surface bad requests immediately instead of returning a partially useful
    # response object to the notebook.
    response.raise_for_status()
    return response.json()


def wait_for_url(
    url: str,
    timeout_seconds: int = 30,
    expected_status: int = 200,
) -> None:
    """Wait until a URL returns the expected status code.

    Local servers take a moment to load their dependencies and model artifacts.
    Polling avoids race conditions where the next request happens too early.
    """

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout = 2)
            if response.status_code == expected_status:
                return
        except requests.RequestException as error:
            last_error = error
        time.sleep(1)

    raise TimeoutError(f'{url} did not return {expected_status}. Last error: {last_error}')


def stop_process(
    process: subprocess.Popen[Any] | None,
    pid_file: Path,
    timeout_seconds: int = 10,
) -> None:
    """Stop a subprocess and remove its PID file.

    This is the clean path used when the current Python process still owns the
    `subprocess.Popen` object.
    """

    if process is not None and process.poll() is None:
        # Try graceful shutdown first so uvicorn/MLflow can release resources.
        process.terminate()
        try:
            process.wait(timeout = timeout_seconds)
        except subprocess.TimeoutExpired:
            # If the process ignores termination, force it so notebook reruns do
            # not leave ports occupied.
            process.kill()
            process.wait(timeout = timeout_seconds)

    if pid_file.exists():
        pid_file.unlink()


def stop_pid_file(pid_file: Path, timeout_seconds: int = 10) -> None:
    """Stop a process by PID file when the Popen object is no longer available.

    This is the recovery path for notebook workflows. If a cell starts a server
    and a later cell errors before cleanup, the next run can still clean up by
    reading the saved PID file.
    """

    if not pid_file.exists():
        return

    pid = int(pid_file.read_text(encoding = 'utf-8').strip())
    try:
        # SIGTERM asks the process to shut down cleanly.
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # A stale PID file is harmless; remove it and continue.
        pid_file.unlink()
        return

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            # Signal 0 does not kill the process; it only checks whether it still
            # exists and whether we have permission to signal it.
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_file.unlink()
            return
        time.sleep(0.5)

    # Fall back to SIGKILL if the process did not exit after SIGTERM.
    os.kill(pid, signal.SIGKILL)
    pid_file.unlink(missing_ok = True)


def _start_process(
    args: list[str],
    pid_file: Path,
    log_file: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    # Keep PID and log locations deterministic so learners can inspect logs if a
    # local server fails to become healthy.
    pid_file.parent.mkdir(parents = True, exist_ok = True)
    log_file.parent.mkdir(parents = True, exist_ok = True)

    # Redirect stdout and stderr to the same log file. MLflow server errors often
    # appear on stderr, and having one file keeps troubleshooting simple.
    log_handle = log_file.open('w', encoding = 'utf-8')
    process = subprocess.Popen(
        args,
        stdout = log_handle,
        stderr = subprocess.STDOUT,
        env = env,
    )
    log_handle.close()
    # Store the PID immediately so a later notebook cell can clean up even if the
    # current Python variables are lost.
    pid_file.write_text(str(process.pid), encoding = 'utf-8')
    return process
