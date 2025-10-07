"""
Pytest configuration for E2E tests.

Provides fixtures for running the Flask test server and accessing test configs.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def market_analyser_server():
    """
    Start the market_analyser Flask server for E2E tests.

    This fixture runs the Flask server in a subprocess and ensures it's ready
    before any tests run. The server is shared across all tests in the session
    for efficiency.

    Yields:
        str: The base URL of the running server (http://localhost:5000)
    """
    # Path to the Flask server script
    server_script = Path(__file__).parent / "fixtures" / "market_analyser.py"

    if not server_script.exists():
        pytest.fail(f"Flask server script not found: {server_script}")

    # Get the venv python path
    import sys

    python_executable = sys.executable

    # Start server process
    process = subprocess.Popen(
        [
            python_executable,
            str(server_script),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "FLASK_ENV": "testing"},
    )

    # Wait for server to be ready
    base_url = "http://localhost:5000"
    max_retries = 20
    retry_delay = 0.5

    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/api/transactions", timeout=2)
            if response.status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            if i == max_retries - 1:
                process.kill()
                stdout, stderr = process.communicate()
                pytest.fail(
                    f"Flask server failed to start after {max_retries} attempts.\n"
                    f"STDOUT: {stdout.decode()}\n"
                    f"STDERR: {stderr.decode()}"
                )
            time.sleep(retry_delay)

    yield base_url

    # Cleanup: terminate the server
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture
def test_configs_dir():
    """
    Return the path to the test configs directory.

    Returns:
        Path: Path to tests/e2e/test_configs/
    """
    return Path(__file__).parent / "test_configs"


@pytest.fixture
def config_file(test_configs_dir):
    """
    Factory fixture to get a specific test config file path.

    Args:
        test_configs_dir: Injected fixture for test configs directory

    Returns:
        callable: Function that takes a config name and returns its path
    """

    def _get_config(name: str) -> Path:
        """Get the path to a specific test config file."""
        config_path = test_configs_dir / name
        if not config_path.exists():
            pytest.fail(f"Test config not found: {config_path}")
        return config_path

    return _get_config


@pytest.fixture
def run_pipeline():
    """
    Factory fixture to run a ShedBoxAI pipeline with a config file.

    Returns:
        callable: Function that takes a config path and returns the pipeline result
    """

    def _run(config_path: Path, verbose: bool = False):
        """
        Run a ShedBoxAI pipeline with the given config.

        Args:
            config_path: Path to the YAML config file
            verbose: Whether to enable verbose logging

        Returns:
            dict: The pipeline execution result
        """
        from shedboxai.pipeline import Pipeline

        pipeline = Pipeline(str(config_path))
        return pipeline.run()

    return _run
