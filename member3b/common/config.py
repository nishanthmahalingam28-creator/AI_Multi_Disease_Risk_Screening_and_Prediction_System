"""
Member 3B Configuration Module.

Provides centralized, environment-aware configuration for the unified backend.
Supports local overrides via environment variables with safe defaults.
"""

from typing import Optional, Any
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    DEBUG: bool = False
    ENV: str = "production"
    TESTING: bool = False


def validate_port(port: Any) -> int:
    """
    Validates that a port value is an integer within the valid TCP port range [1, 65535].

    Args:
        port: The port value to validate (int or str).

    Returns:
        int: The validated port integer.

    Raises:
        ValueError: If port is not a valid integer or is outside [1, 65535].
    """
    try:
        p = int(port)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid PORT value '{port}': Port must be an integer between 1 and 65535."
        ) from e

    if not (1 <= p <= 65535):
        raise ValueError(
            f"Invalid PORT value {p}: Port must be between 1 and 65535."
        )
    return p


def get_config(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debug: Optional[bool] = None,
    env: Optional[str] = None,
    testing: Optional[bool] = None,
    strict: bool = False,
) -> AppConfig:
    """
    Constructs an AppConfig instance with values resolved in order of precedence:
    1. Explicit arguments
    2. Environment variables (HOST, PORT, FLASK_DEBUG, FLASK_ENV)
    3. Safe default values

    Args:
        host: Optional explicit host override.
        port: Optional explicit port override.
        debug: Optional explicit debug mode override.
        env: Optional explicit environment name override.
        testing: Optional explicit testing mode flag.
        strict: If True, raises ValueError on invalid port values.
                If False (default), safely falls back to default 5000.

    Returns:
        AppConfig: Immutable configuration object.
    """
    env_host = os.getenv("HOST", "127.0.0.1")

    # Resolve and validate port
    raw_port = port if port is not None else os.getenv("PORT", "5000")
    if strict:
        resolved_port = validate_port(raw_port)
    else:
        try:
            resolved_port = validate_port(raw_port)
        except ValueError:
            resolved_port = 5000

    env_debug_raw = os.getenv("FLASK_DEBUG", os.getenv("DEBUG", "false")).lower()
    env_debug = env_debug_raw in ("1", "true", "yes", "on")

    env_name = os.getenv("FLASK_ENV", "production")

    resolved_host = host if host is not None else env_host
    resolved_debug = debug if debug is not None else env_debug
    resolved_env = env if env is not None else env_name
    resolved_testing = testing if testing is not None else False

    return AppConfig(
        HOST=resolved_host,
        PORT=resolved_port,
        DEBUG=resolved_debug,
        ENV=resolved_env,
        TESTING=resolved_testing,
    )

