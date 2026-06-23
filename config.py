"""
config.py - Configuration loader
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "brain": {
        "n_neurons": 2000,
        "n_areas": 3,
        "cap_fraction": 0.01,
        "plasticity_rate": 0.10,
        "connection_density": 0.05,
        "random_seed": 42,
    },
    "patterns": {
        "n_patterns": 5,
        "pattern_size": 100,
        "active_fraction": 0.30,
    },
    "training": {
        "n_rounds": 150,
        "completion_rounds": 5,
    },
    "visualization": {
        "max_heatmap_neurons": 500,
        "network_display_neurons": 40,
    },
}

def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    config = {section: dict(values) for section, values in _DEFAULTS.items()}
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file %s not found; using built-in defaults.", path)
        return config
    try:
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        for section, values in loaded.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
    except Exception as exc:
        logger.warning("Failed to parse %s (%s); using built-in defaults.", path, exc)
    return config