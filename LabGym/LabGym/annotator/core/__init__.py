"""Annotator core (data models, managers, video, export)."""

from .annotation_manager import AnnotationManager
from .data_models import (
    SCHEMA_VERSION,
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
    TracksRef,
)
from .example_generator import ExampleGenerator
from .metrics_calculator import MetricsCalculator
from .tracklets_bridge import (
    LoadedTracklets,
    apply_subjects_to_session,
    load_tracklets_for_annotator,
    overlays_at_video_frame,
)
from .video_handler import VideoHandler, VideoMetadata

__all__ = [
    "SCHEMA_VERSION",
    "AnnotationManager",
    "AnnotationSession",
    "Behavior",
    "Bout",
    "Subject",
    "TracksRef",
    "ExampleGenerator",
    "MetricsCalculator",
    "VideoHandler",
    "VideoMetadata",
    "LoadedTracklets",
    "apply_subjects_to_session",
    "load_tracklets_for_annotator",
    "overlays_at_video_frame",
]
