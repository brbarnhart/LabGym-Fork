"""Geometry for the preprocess crop window.

Edges are exclusive pixel bounds on a resized frame. The kept region is
rows ``[top, bottom)`` and columns ``[left, right)``, the same slice
``preprocess_video`` writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class FrameSize:
    """Width and height of a frame in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class CropEdges:
    """Exclusive left, top, right, and bottom edges of a crop window."""

    left: int
    top: int
    right: int
    bottom: int

    def valid(self) -> bool:
        """Return True when the kept region has a positive width and height."""
        return self.right > self.left and self.bottom > self.top

    @property
    def width(self) -> int:
        """Kept width in pixels (``right - left``)."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Kept height in pixels (``bottom - top``)."""
        return self.bottom - self.top


@dataclass(frozen=True)
class RememberedFrame:
    """Native video size and the resized size the crop edges belong to."""

    native: FrameSize
    resized: FrameSize


@dataclass(frozen=True)
class CropSnapshot:
    """Edges, crop checkbox, and remembered frame for one undo step."""

    edges: CropEdges
    crop_enabled: bool
    remembered: Optional[RememberedFrame]


class HitZone(Enum):
    """Part of the crop rectangle under the pointer, in widget pixels."""

    OUTSIDE = "outside"
    INSIDE = "inside"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


_LEFT_ZONES = (HitZone.LEFT, HitZone.TOP_LEFT, HitZone.BOTTOM_LEFT)
_RIGHT_ZONES = (HitZone.RIGHT, HitZone.TOP_RIGHT, HitZone.BOTTOM_RIGHT)
_TOP_ZONES = (HitZone.TOP, HitZone.TOP_LEFT, HitZone.TOP_RIGHT)
_BOTTOM_ZONES = (HitZone.BOTTOM, HitZone.BOTTOM_LEFT, HitZone.BOTTOM_RIGHT)


def resized_frame_size(
    native_width: int,
    native_height: int,
    framewidth: Optional[int],
) -> FrameSize:
    """Return the frame size after the preprocess width resize.

    Height uses the same truncation as ``preprocess_video``:
    ``int(framewidth * height / width)``. A missing or non-positive
    ``framewidth`` keeps the native size.

    Args:
        native_width: Source frame width in pixels.
        native_height: Source frame height in pixels.
        framewidth: Target width, or None / 0 to keep the native size.

    Returns:
        The resized width and height. Both are 0 when the native size is empty.
    """
    if native_width <= 0 or native_height <= 0:
        return FrameSize(0, 0)
    if framewidth is None or int(framewidth) <= 0:
        return FrameSize(int(native_width), int(native_height))
    width = int(framewidth)
    height = int(int(framewidth) * int(native_height) / int(native_width))
    return FrameSize(width, max(height, 1))


def scale_edges(edges: CropEdges, old: FrameSize, new: FrameSize) -> CropEdges:
    """Scale edges so they cover the same fraction of ``old`` on ``new``.

    Args:
        edges: Crop edges measured on ``old``.
        old: Frame size the edges currently belong to.
        new: Frame size to measure the edges on.

    Returns:
        Integer edges on ``new``. ``edges`` unchanged when either size is empty.
    """
    if old.width <= 0 or old.height <= 0 or new.width <= 0 or new.height <= 0:
        return edges

    def _sx(value: int) -> int:
        return int(round(value * new.width / old.width))

    def _sy(value: int) -> int:
        return int(round(value * new.height / old.height))

    return CropEdges(
        _sx(edges.left),
        _sy(edges.top),
        _sx(edges.right),
        _sy(edges.bottom),
    )


def intersect_edges(edges: CropEdges, frame: FrameSize) -> Optional[CropEdges]:
    """Return the part of ``edges`` that lies on ``frame``.

    Args:
        edges: Stored crop window. Edges may fall outside ``frame``.
        frame: Frame the crop is drawn or written on.

    Returns:
        The overlapping crop, or None when the overlap is empty or ``edges``
        is not a valid crop window.
    """
    if frame.width <= 0 or frame.height <= 0 or not edges.valid():
        return None
    clamped = CropEdges(
        max(0, edges.left),
        max(0, edges.top),
        min(frame.width, edges.right),
        min(frame.height, edges.bottom),
    )
    if not clamped.valid():
        return None
    return clamped


def shift_inside(edges: CropEdges, frame: FrameSize) -> CropEdges:
    """Slide ``edges`` inside ``frame``, shrinking it only when it is larger.

    Args:
        edges: A valid crop window.
        frame: Frame that must contain the result.

    Returns:
        A crop inside ``frame`` with the same size when that size fits.
        ``edges`` unchanged when it is not a valid crop or ``frame`` is empty.
    """
    if not edges.valid() or frame.width <= 0 or frame.height <= 0:
        return edges
    if edges.width >= frame.width:
        left, right = 0, frame.width
    else:
        left = min(max(0, edges.left), frame.width - edges.width)
        right = left + edges.width
    if edges.height >= frame.height:
        top, bottom = 0, frame.height
    else:
        top = min(max(0, edges.top), frame.height - edges.height)
        bottom = top + edges.height
    return CropEdges(left, top, right, bottom)


def edges_from_points(x0: int, y0: int, x1: int, y1: int) -> CropEdges:
    """Build an exclusive rectangle spanning two image boundary points.

    Args:
        x0: First x boundary, in ``0..width``.
        y0: First y boundary, in ``0..height``.
        x1: Second x boundary.
        y1: Second y boundary.

    Returns:
        Edges with ``right >= left`` and ``bottom >= top``. A zero-length
        drag yields an empty (invalid) crop.
    """
    left, right = sorted((int(x0), int(x1)))
    top, bottom = sorted((int(y0), int(y1)))
    return CropEdges(left, top, right, bottom)


def drag_moved(
    start: Tuple[float, float],
    end: Tuple[float, float],
    minimum_px: float = 4,
) -> bool:
    """Return True when the pointer moved at least ``minimum_px`` on one axis.

    Args:
        start: Pointer position where the drag began, in widget pixels.
        end: Pointer position where the drag ended, in widget pixels.
        minimum_px: Smallest movement that counts as a drag.
    """
    return abs(end[0] - start[0]) >= minimum_px or abs(end[1] - start[1]) >= minimum_px


def nudge_edges(
    edges: CropEdges,
    dx: int,
    dy: int,
    frame: FrameSize,
) -> Optional[CropEdges]:
    """Shift the on-frame part of ``edges`` by ``dx`` and ``dy``.

    A crop that hangs off ``frame`` is adopted as its on-frame part first.
    The result stays inside ``frame``.

    Args:
        edges: Stored crop window.
        dx: Horizontal shift in image pixels.
        dy: Vertical shift in image pixels.
        frame: Picture the gesture is editing.

    Returns:
        The crop to store, or None when ``edges`` does not overlap ``frame``.
    """
    base = intersect_edges(edges, frame)
    if base is None:
        return None
    moved = CropEdges(base.left + dx, base.top + dy, base.right + dx, base.bottom + dy)
    return shift_inside(moved, frame)


def resize_edges(
    edges: CropEdges,
    zone: HitZone,
    x: int,
    y: int,
    frame: FrameSize,
) -> Optional[CropEdges]:
    """Drag one edge or corner of the on-frame crop.

    Args:
        edges: Stored crop window.
        zone: Edge or corner being dragged.
        x: New horizontal boundary in image pixels.
        y: New vertical boundary in image pixels.
        frame: Picture the gesture is editing.

    Returns:
        A valid crop inside ``frame``, or None when ``edges`` does not
        overlap ``frame``.
    """
    base = intersect_edges(edges, frame)
    if base is None or frame.width <= 0 or frame.height <= 0:
        return None
    left, top, right, bottom = base.left, base.top, base.right, base.bottom
    x = min(max(0, int(x)), frame.width)
    y = min(max(0, int(y)), frame.height)
    if zone in _LEFT_ZONES:
        left = min(x, right - 1)
    if zone in _RIGHT_ZONES:
        right = max(x, left + 1)
    if zone in _TOP_ZONES:
        top = min(y, bottom - 1)
    if zone in _BOTTOM_ZONES:
        bottom = max(y, top + 1)
    left = min(max(0, left), frame.width - 1)
    right = min(max(left + 1, right), frame.width)
    top = min(max(0, top), frame.height - 1)
    bottom = min(max(top + 1, bottom), frame.height)
    return CropEdges(left, top, right, bottom)


@dataclass(frozen=True)
class ImageFit:
    """Letterboxed placement of an image inside a widget."""

    scale: float
    offset_x: float
    offset_y: float
    display_width: float
    display_height: float


def fit_image(image_w: int, image_h: int, widget_w: int, widget_h: int) -> ImageFit:
    """Scale ``image_w`` × ``image_h`` uniformly so it fits in the widget.

    Args:
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        widget_w: Widget width in pixels.
        widget_h: Widget height in pixels.

    Returns:
        Scale and top-left offset of the letterboxed image. An empty input
        yields a zero display size.
    """
    if image_w <= 0 or image_h <= 0 or widget_w <= 0 or widget_h <= 0:
        return ImageFit(1.0, 0.0, 0.0, 0.0, 0.0)
    scale = min(widget_w / image_w, widget_h / image_h)
    display_w = image_w * scale
    display_h = image_h * scale
    return ImageFit(
        scale,
        (widget_w - display_w) / 2.0,
        (widget_h - display_h) / 2.0,
        display_w,
        display_h,
    )


def image_to_widget(x: float, y: float, fit: ImageFit) -> Tuple[float, float]:
    """Map an image-boundary point to widget pixels."""
    return (fit.offset_x + x * fit.scale, fit.offset_y + y * fit.scale)


def widget_to_image(
    x: float,
    y: float,
    fit: ImageFit,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int]]:
    """Map a widget point to an image boundary.

    Args:
        x: Widget x in pixels.
        y: Widget y in pixels.
        fit: Placement of the image in the widget.
        image_w: Image width. The returned x is in ``0..image_w``.
        image_h: Image height. The returned y is in ``0..image_h``.

    Returns:
        Rounded image boundaries, or None when the point is in the letterbox
        margin or the image is not displayed.
    """
    if fit.display_width <= 0 or fit.display_height <= 0 or fit.scale == 0:
        return None
    if x < fit.offset_x or y < fit.offset_y:
        return None
    if x > fit.offset_x + fit.display_width or y > fit.offset_y + fit.display_height:
        return None
    ix = int(round((x - fit.offset_x) / fit.scale))
    iy = int(round((y - fit.offset_y) / fit.scale))
    return (min(max(0, ix), image_w), min(max(0, iy), image_h))


def widget_to_image_clamped(
    x: float,
    y: float,
    fit: ImageFit,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int]]:
    """Map a widget point to an image boundary, clamping into the image.

    Used while a drag is already active so the pointer can reach the frame
    edge from the letterbox margin.

    Args:
        x: Widget x in pixels.
        y: Widget y in pixels.
        fit: Placement of the image in the widget.
        image_w: Image width.
        image_h: Image height.

    Returns:
        Image boundaries, or None when the image is not displayed.
    """
    if fit.display_width <= 0 or fit.display_height <= 0:
        return None
    cx = min(max(x, fit.offset_x), fit.offset_x + fit.display_width)
    cy = min(max(y, fit.offset_y), fit.offset_y + fit.display_height)
    return widget_to_image(cx, cy, fit, image_w, image_h)


def _near(px: float, py: float, cx: float, cy: float, threshold: float) -> bool:
    return abs(px - cx) <= threshold and abs(py - cy) <= threshold


def _near_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    threshold: float,
) -> bool:
    if abs(y1 - y2) <= 1e-6:
        if abs(py - y1) > threshold:
            return False
        lo, hi = sorted((x1, x2))
        return lo - threshold <= px <= hi + threshold
    if abs(px - x1) > threshold:
        return False
    lo, hi = sorted((y1, y2))
    return lo - threshold <= py <= hi + threshold


def hit_test(
    x: float,
    y: float,
    edges: CropEdges,
    fit: ImageFit,
    threshold: float = 8,
) -> HitZone:
    """Return the crop-rectangle zone under a widget point.

    Corners win over edges, then edges over the interior. An invalid crop
    is entirely outside.

    Args:
        x: Widget x in pixels.
        y: Widget y in pixels.
        edges: Crop window in image boundaries. Ignored when invalid.
        fit: Placement of the image in the widget.
        threshold: Hit slop around edges and corners, in widget pixels.
    """
    if not edges.valid():
        return HitZone.OUTSIDE
    left, top = image_to_widget(edges.left, edges.top, fit)
    right, bottom = image_to_widget(edges.right, edges.bottom, fit)
    corners = (
        (HitZone.TOP_LEFT, left, top),
        (HitZone.TOP_RIGHT, right, top),
        (HitZone.BOTTOM_LEFT, left, bottom),
        (HitZone.BOTTOM_RIGHT, right, bottom),
    )
    for zone, cx, cy in corners:
        if _near(x, y, cx, cy, threshold):
            return zone
    if _near_segment(x, y, left, top, right, top, threshold):
        return HitZone.TOP
    if _near_segment(x, y, left, bottom, right, bottom, threshold):
        return HitZone.BOTTOM
    if _near_segment(x, y, left, top, left, bottom, threshold):
        return HitZone.LEFT
    if _near_segment(x, y, right, top, right, bottom, threshold):
        return HitZone.RIGHT
    if left <= x <= right and top <= y <= bottom:
        return HitZone.INSIDE
    return HitZone.OUTSIDE


def mismatch_lines(
    remembered: FrameSize,
    videos: Sequence[Tuple[str, FrameSize]],
) -> Tuple[Tuple[str, FrameSize], ...]:
    """Return checked videos whose resized size differs from ``remembered``.

    Args:
        remembered: Resized size the crop window was set on.
        videos: ``(path, resized size)`` for each checked video.

    Returns:
        The entries whose size is not ``remembered``, in the same order.
    """
    return tuple((path, size) for path, size in videos if size != remembered)


class CropWindowState:
    """Crop edges, the crop checkbox, the remembered frame, and one undo step.

    Typing does not record an undo step. Drawing, moving, resizing, nudging,
    and clearing do. Changing frame width scales the edges only when a frame
    size is already remembered, and drops the undo step.
    """

    def __init__(self) -> None:
        self.edges = CropEdges(0, 0, 0, 0)
        self.crop_enabled = False
        self.remembered: Optional[RememberedFrame] = None
        self._undo: Optional[CropSnapshot] = None

    def snapshot(self) -> CropSnapshot:
        """Return the current edges, checkbox, and remembered frame."""
        return CropSnapshot(self.edges, self.crop_enabled, self.remembered)

    def note_preview(self, native: FrameSize, resized: FrameSize) -> bool:
        """Remember ``resized`` when a valid crop has no frame size yet.

        Opening the editor on a preview uses this. Highlighting another video
        does not replace a size that is already stored.

        Args:
            native: Native size of the preview video.
            resized: Size of the preview after the frame-width resize.

        Returns:
            True when the remembered size was set.
        """
        if self.remembered is None and self.edges.valid():
            self.remembered = RememberedFrame(native, resized)
            return True
        return False

    def commit_gesture(
        self,
        edges: CropEdges,
        native: FrameSize,
        resized: FrameSize,
        before: Optional[CropSnapshot] = None,
    ) -> None:
        """Store a draw, move, resize, or nudge and remember ``resized``.

        A valid result checks Crop frames. ``before`` is the state to restore
        on undo, so a live preview can change ``edges`` before the gesture
        ends. When ``before`` is omitted, the state at call time is used.
        Nothing is recorded when the gesture leaves the state unchanged.

        Args:
            edges: Crop window written by the gesture. It lies on the picture.
            native: Native size of the picture being edited.
            resized: Resized size of that picture.
            before: State from before the gesture started.
        """
        baseline = before if before is not None else self.snapshot()
        self.edges = edges
        if edges.valid():
            self.crop_enabled = True
        self.remembered = RememberedFrame(native, resized)
        if self.snapshot() != baseline:
            self._undo = baseline

    def type_edges(
        self,
        edges: CropEdges,
        preview: Optional[Tuple[FrameSize, FrameSize]] = None,
    ) -> None:
        """Store edges typed into the fields.

        Does not change the crop checkbox or the undo step. When ``preview``
        is set, the crop window now belongs to that preview.

        Args:
            edges: Numbers from the edge fields. They may extend past the preview.
            preview: ``(native, resized)`` of the preview on screen, or None
                when no preview is showing.
        """
        self.edges = edges
        if preview is not None:
            native, resized = preview
            self.remembered = RememberedFrame(native, resized)

    def set_crop_enabled(self, enabled: bool) -> None:
        """Set whether preprocessing applies the crop window.

        The rectangle and the remembered size stay as they are.
        """
        self.crop_enabled = bool(enabled)

    def clear(self) -> None:
        """Forget the crop window and record that state for undo."""
        before = self.snapshot()
        self.edges = CropEdges(0, 0, 0, 0)
        self.crop_enabled = False
        self.remembered = None
        if self.snapshot() != before:
            self._undo = before

    def undo(self) -> bool:
        """Restore the previous crop window. One step, then the step is gone.

        Returns:
            True when a step was restored.
        """
        if self._undo is None:
            return False
        step = self._undo
        self.edges = step.edges
        self.crop_enabled = step.crop_enabled
        self.remembered = step.remembered
        self._undo = None
        return True

    def apply_framewidth(self, framewidth: Optional[int]) -> bool:
        """Scale the crop window onto the frame width of its remembered video.

        Uses the native size stored with the crop, not whichever video is
        highlighted. Does nothing when no size is remembered. Drops the undo
        step when the edges change, because that step was measured on the
        previous size.

        Args:
            framewidth: Target width, or None / 0 for the native size.

        Returns:
            True when the edges or remembered size changed.
        """
        if self.remembered is None:
            return False
        new_resized = resized_frame_size(
            self.remembered.native.width,
            self.remembered.native.height,
            framewidth,
        )
        if new_resized == self.remembered.resized:
            return False
        self.edges = scale_edges(self.edges, self.remembered.resized, new_resized)
        self.remembered = RememberedFrame(self.remembered.native, new_resized)
        self._undo = None
        return True
