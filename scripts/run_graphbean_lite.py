#!/usr/bin/env python3
"""Deprecated shim for scripts/run_ridge_feature_decoder.py.

Overnight step id graphbean_lite is a frozen historical ops wire name.
This path is not the front-door CLI.
"""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "scripts/run_graphbean_lite.py is deprecated; use scripts/run_ridge_feature_decoder.py",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("run_ridge_feature_decoder.py")),
        run_name="__main__",
    )
