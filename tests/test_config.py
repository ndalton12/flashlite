"""Tests for configuration and environment loading."""

import os
from pathlib import Path

import pytest

from flashlite.config import (
    FlashliteConfig,
    load_env_files,
    validate_api_keys,
)
from flashlite.types import ConfigError


class TestFlashliteConfig:
    """Tests for FlashliteConfig."""

    def test_default_config(self) -> None:
        config = FlashliteConfig()

        assert config.default_model is None
        assert config.retry.max_attempts == 3
        assert config.rate_limit.requests_per_minute is None
        assert config.log_requests is False
        assert config.timeout == 600.0

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLASHLITE_DEFAULT_MODEL", "gpt-4o")
        monkeypatch.setenv("FLASHLITE_DEFAULT_TEMPERATURE", "0.7")
        monkeypatch.setenv("FLASHLITE_LOG_LEVEL", "debug")
        monkeypatch.setenv("FLASHLITE_LOG_REQUESTS", "true")
        monkeypatch.setenv("FLASHLITE_RATE_LIMIT_RPM", "60")
        monkeypatch.setenv("FLASHLITE_TIMEOUT", "300")

        config = FlashliteConfig.from_env()

        assert config.default_model == "gpt-4o"
        assert config.default_temperature == 0.7
        assert config.log_level == "DEBUG"
        assert config.log_requests is True
        assert config.rate_limit.requests_per_minute == 60.0
        assert config.timeout == 300.0

    def test_invalid_temperature_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLASHLITE_DEFAULT_TEMPERATURE", "not_a_number")

        with pytest.raises(ConfigError, match="Invalid FLASHLITE_DEFAULT_TEMPERATURE"):
            FlashliteConfig.from_env()

    def test_invalid_max_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLASHLITE_DEFAULT_MAX_TOKENS", "abc")

        with pytest.raises(ConfigError, match="Invalid FLASHLITE_DEFAULT_MAX_TOKENS"):
            FlashliteConfig.from_env()


class TestLoadEnvFiles:
    """Tests for environment file loading."""

    def test_load_single_file(self, mock_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        load_env_files(env_file=mock_env_file)

        assert os.getenv("OPENAI_API_KEY") == "sk-test-key"
        assert os.getenv("FLASHLITE_DEFAULT_MODEL") == "gpt-4o-mini"

    def test_load_multiple_files_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Create two env files
        env1 = tmp_path / ".env"
        env1.write_text("VAR_A=from_env1\nVAR_B=from_env1")

        env2 = tmp_path / ".env.local"
        env2.write_text("VAR_B=from_env2_override")

        monkeypatch.delenv("VAR_A", raising=False)
        monkeypatch.delenv("VAR_B", raising=False)

        load_env_files(env_files=[env1, env2])

        assert os.getenv("VAR_A") == "from_env1"
        assert os.getenv("VAR_B") == "from_env2_override"

    def test_missing_file_is_ignored(self) -> None:
        # Should not raise when file doesn't exist
        load_env_files(env_file=Path("/nonexistent/.env"))


class TestValidateApiKeys:
    """Tests for API key validation."""

    def test_returns_status_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = validate_api_keys()

        assert result["openai"] is True
        assert result["anthropic"] is False

    def test_raises_on_missing_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ConfigError, match="Missing API keys"):
            validate_api_keys(required_providers=["openai", "anthropic"])

    def test_passes_when_required_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Should not raise
        result = validate_api_keys(required_providers=["openai"])
        assert result["openai"] is True
