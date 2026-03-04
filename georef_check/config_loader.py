"""
Configuration loader for georef_check.
Supports YAML config files with CLI override capability.
"""

import yaml
from pathlib import Path
from argparse import Namespace
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = "configs/georef_check.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, returns empty dict.

    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config_with_args(config: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
    """
    Merge configuration with CLI arguments.
    CLI arguments override config values.

    Args:
        config: Configuration dictionary from YAML.
        args: Parsed command-line arguments.

    Returns:
        Merged configuration dictionary.
    """
    result = config.copy()

    # Get all CLI arguments that were explicitly provided (not None)
    for key, value in vars(args).items():
        if value is not None:
            result[key] = value

    return result


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get configuration value with fallback to default.

    Args:
        config: Configuration dictionary.
        key: Configuration key.
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    return config.get(key, default)


def get_default_config_path() -> str:
    """
    Get the default config path relative to this module.

    Returns:
        Path to default config file.
    """
    return str(Path(__file__).parent.parent / DEFAULT_CONFIG_PATH)
