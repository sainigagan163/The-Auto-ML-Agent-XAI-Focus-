"""Project entrypoint for running orchestration."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import DEFAULT_CONFIG_PATH, run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline entrypoint."""
    parser = argparse.ArgumentParser(description="Run the AutoML pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the AutoML pipeline using configuration from CLI arguments."""
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
