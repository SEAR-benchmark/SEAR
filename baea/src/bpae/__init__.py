"""Backward-compatible import alias for the renamed :mod:`baea` package."""
from __future__ import annotations

import importlib
import sys

from baea import *  # noqa: F401,F403
from baea import __all__

for _name in (
    "acoustic", "agent", "binding", "cache", "cli", "policies",
    "reference", "tools", "two_stage_binding",
):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"baea.{_name}")
