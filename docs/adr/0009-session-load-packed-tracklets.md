# Packed tracklets on session load

Status: accepted

**Session load** (Review IDs or Annotate ethogram) must not expand every frame’s outlines into Python polygons before the tab is usable. We keep the existing on-disk `*_tracklets.npz` layout. After load, geometry lives as packed arrays; a frame’s outlines are sliced when that frame is drawn. Review IDs loads the **visible animal kind** first; Annotate still loads every kind needed for overlays. A full mutable clone waits until the first switch-list edit. Play/seek must not feel worse than today; this decision does not change the video decoder.

## Considered options

- **Packed arrays + per-frame slice (chosen)** — same npz files; drop the all-frames Python expansion and clone-at-load; outlines for frame 0 as soon as that slice exists. Save / remap / hard-case extract may still walk the whole video then.
- **Keep full in-memory contour lists; only move load off the UI thread** — tab still waits on the whole-video expand; 20–30s session load remains.
- **Expand the whole video on first Play** — makes Load look fast, then puts the wait on the first Play; undoes the load-time gain.
- **Change the on-disk format or memory-map npz** — npz is a zip; mmap is not a free win. Format change is a later, separate decision.
- **Make Play smoother (sequential video read) in the same change** — rejected as a hitch of its own; playback must not regress, but decoder behavior stays as it is.

## Consequences

- ADR-0006 is unchanged: **raw tracklets** stay immutable, **switch markers** remain the editable identity source of truth, **remapped tracklets** publish only on Review IDs save, **accepted identities** still gate Annotate / generate / Process videos.
- After session load, callers must not assume every `contours[id][frame]` polygon already exists. Drawing, seeking, and Play pull the current frame from packed arrays (in RAM, not from disk per tick).
- Review IDs may show one animal kind before others exist in memory; changing the kind combo is a further load, cancellable like any session load.
- Clone-for-remap is an edit-time cost, not a load-time cost. Save still rebuilds remapped geometry from raw, never from public files (ADR-0006).
- A failed video open or corrupt npz must not wedge the shell: metadata from package JSON can remain; the user can pick another video.
- **Project swap** (open / Edit Project) is out of this ADR: it must not perform a session load, and it does not scan identity packages for every video up front.
