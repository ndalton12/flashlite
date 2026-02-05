"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean environment variables before each test."""
    # Remove any FLASHLITE_ env vars that might interfere
    for key in list(os.environ.keys()):
        if key.startswith("FLASHLITE_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_template_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with test templates."""
    # Create some test templates
    (tmp_path / "greeting.jinja").write_text(
        "Hello, {{ name }}! Welcome to {{ place }}."
    )
    (tmp_path / "system.jinja").write_text(
        "You are a helpful assistant specializing in {{ topic }}."
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.jinja").write_text(
        "Nested template: {{ value }}"
    )
    return tmp_path


@pytest.fixture
def mock_env_file(tmp_path: Path) -> Path:
    """Create a temporary .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
OPENAI_API_KEY=sk-test-key
FLASHLITE_DEFAULT_MODEL=gpt-4o-mini
FLASHLITE_LOG_LEVEL=DEBUG
FLASHLITE_RATE_LIMIT_RPM=60
"""
    )
    return env_file
