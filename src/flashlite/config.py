"""Configuration management and environment loading."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .types import ConfigError, RateLimitConfig, RetryConfig


@dataclass
class FlashliteConfig:
    """Main configuration for Flashlite client."""

    # Default model to use if not specified per-request
    default_model: str | None = None

    # Default completion parameters
    default_temperature: float | None = None
    default_max_tokens: int | None = None

    # Retry configuration
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Rate limiting configuration
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Template directory
    template_dir: Path | str | None = None

    # Logging
    log_requests: bool = False
    log_level: str = "INFO"

    # Default kwargs to pass to all completions
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    # Timeout in seconds
    timeout: float = 600.0

    @classmethod
    def from_env(cls) -> "FlashliteConfig":
        """Create config from environment variables."""
        config = cls()

        # Read config from FLASHLITE_ prefixed env vars
        if model := os.getenv("FLASHLITE_DEFAULT_MODEL"):
            config.default_model = model

        if temp := os.getenv("FLASHLITE_DEFAULT_TEMPERATURE"):
            try:
                config.default_temperature = float(temp)
            except ValueError:
                raise ConfigError(f"Invalid FLASHLITE_DEFAULT_TEMPERATURE: {temp}")

        if max_tokens := os.getenv("FLASHLITE_DEFAULT_MAX_TOKENS"):
            try:
                config.default_max_tokens = int(max_tokens)
            except ValueError:
                raise ConfigError(f"Invalid FLASHLITE_DEFAULT_MAX_TOKENS: {max_tokens}")

        if log_level := os.getenv("FLASHLITE_LOG_LEVEL"):
            config.log_level = log_level.upper()

        if os.getenv("FLASHLITE_LOG_REQUESTS", "").lower() in ("1", "true", "yes"):
            config.log_requests = True

        if template_dir := os.getenv("FLASHLITE_TEMPLATE_DIR"):
            config.template_dir = Path(template_dir)

        if rpm := os.getenv("FLASHLITE_RATE_LIMIT_RPM"):
            try:
                config.rate_limit.requests_per_minute = float(rpm)
            except ValueError:
                raise ConfigError(f"Invalid FLASHLITE_RATE_LIMIT_RPM: {rpm}")

        if tpm := os.getenv("FLASHLITE_RATE_LIMIT_TPM"):
            try:
                config.rate_limit.tokens_per_minute = float(tpm)
            except ValueError:
                raise ConfigError(f"Invalid FLASHLITE_RATE_LIMIT_TPM: {tpm}")

        if timeout := os.getenv("FLASHLITE_TIMEOUT"):
            try:
                config.timeout = float(timeout)
            except ValueError:
                raise ConfigError(f"Invalid FLASHLITE_TIMEOUT: {timeout}")

        return config


def load_env_files(
    env_file: str | Path | None = None,
    env_files: list[str | Path] | None = None,
) -> None:
    """
    Load environment variables from .env files.

    Args:
        env_file: Single env file to load
        env_files: Multiple env files to load (later files override earlier)
    """
    files_to_load: list[Path] = []

    if env_files:
        files_to_load.extend(Path(f) for f in env_files)
    elif env_file:
        files_to_load.append(Path(env_file))
    else:
        # Default: try to load .env from current directory
        default_env = Path(".env")
        if default_env.exists():
            files_to_load.append(default_env)

    # Load files in order (later overrides earlier)
    for file_path in files_to_load:
        if file_path.exists():
            load_dotenv(file_path, override=True)


def validate_api_keys(required_providers: list[str] | None = None) -> dict[str, bool]:
    """
    Check which API keys are configured.

    Args:
        required_providers: If provided, raise error if any are missing

    Returns:
        Dict mapping provider names to whether their key is set
    """
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "azure": "AZURE_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }

    results = {}
    for provider, env_var in key_mapping.items():
        results[provider] = bool(os.getenv(env_var))

    if required_providers:
        missing = [p for p in required_providers if not results.get(p)]
        if missing:
            raise ConfigError(
                f"Missing API keys for providers: {', '.join(missing)}. "
                f"Set the corresponding environment variables."
            )

    return results
