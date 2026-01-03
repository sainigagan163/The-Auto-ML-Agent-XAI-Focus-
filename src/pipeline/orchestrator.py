"""Orchestration utilities for the AutoML workflow."""
from __future__ import annotations

from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load pipeline configuration from YAML."""
    if not config_path.exists():
        message = f"Config file not found: {config_path}"
        raise FileNotFoundError(message)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return config


def run_pipeline(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Placeholder for the orchestration flow."""
    config = load_config(config_path)
    dataset_path = config.get("dataset_path", "")
    target_column = config.get("target_column", "")
    automl_budget = config.get("automl_budget", {})

    print("AutoML pipeline configured.")
    print(f"Dataset: {dataset_path}")
    print(f"Target column: {target_column}")
    print(f"AutoML budget: {automl_budget}")
