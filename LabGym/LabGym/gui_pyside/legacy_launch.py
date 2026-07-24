"""Launch the legacy wx LabGym GUI in a separate process (no shared event loop)."""

from __future__ import annotations

import subprocess
import sys
from typing import List, Optional


def launch_legacy_labgym(extra_args: Optional[List[str]] = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "LabGym"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.Popen(cmd, close_fds=True)


def launch_standalone_annotator() -> subprocess.Popen:
    cmd = [sys.executable, "-m", "LabGym.annotator"]
    return subprocess.Popen(cmd, close_fds=True)
