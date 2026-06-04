#!/usr/bin/env python3
"""Backward-compatible alias for run_experiment (default config: pilot)."""

from run_experiment import main

if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(root / "configs/pilot_config.yaml")])
    main()
