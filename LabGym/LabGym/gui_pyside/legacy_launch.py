"""Launch the legacy wx LabGym GUI in a separate process (no shared event loop).

Phase 8: the default ``LabGym`` entry point is the PySide workbench. Legacy
wx must be started with ``--legacy-wx`` (or ``LABGYM_LEGACY_WX=1``).
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Optional


def launch_legacy_labgym(extra_args: Optional[List[str]] = None) -> subprocess.Popen:
    """Spawn classic wx GUI: ``python -m LabGym --legacy-wx`` [extra_args]."""
    cmd = [sys.executable, "-m", "LabGym", "--legacy-wx"]
    if extra_args:
        # Avoid nesting --legacy-wx twice if caller passes it.
        filtered = [a for a in extra_args if a not in ("--legacy-wx", "--legacy_wx")]
        cmd.extend(filtered)
    return subprocess.Popen(cmd, close_fds=True)


def launch_standalone_annotator() -> subprocess.Popen:
    cmd = [sys.executable, "-m", "LabGym.annotator"]
    return subprocess.Popen(cmd, close_fds=True)
