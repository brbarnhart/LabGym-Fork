"""LabGym multi-animal behavior video annotator (PySide6).

Frame-accurate ethogram annotation for generating LabGym categorizer training data.
"""

from .core.data_models import (
    SCHEMA_VERSION,
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
    TracksRef,
)
from .core.annotation_manager import AnnotationManager

__all__ = [
    "SCHEMA_VERSION",
    "AnnotationSession",
    "AnnotationManager",
    "Behavior",
    "Bout",
    "Subject",
    "TracksRef",
    "main",
]


def main() -> None:
    """Launch the LabGym Behavior Annotator GUI."""
    from .__main__ import main as _main

    _main()
