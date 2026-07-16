Feature: Offline ID Switch Correction for LabGym Detector (Option A – Rule-Based + Appearance Matching)
1. Project Context
LabGym uses a Detectron2-based Detector for multi-animal instance segmentation and tracking. A common failure mode occurs during/after close-proximity interactions (e.g., wrestling in mice): identities switch when animals separate.
We want to add an offline post-processing refinement step that corrects these ID switches using appearance similarity and simple consistency rules.
2. Goal
Implement a modular offline refinement module that:

Detects close-proximity (contact) events
Extracts appearance embeddings for animals just before and just after contact
Re-assigns identities using cosine similarity + Hungarian matching
Enforces basic constraints (no two animals share the same ID at the same time, spatial continuity)
Produces corrected tracklets

The feature should be optional and controlled by a configuration flag.
3. High-Level Architecture
textMain Tracking Pass (existing LabGym Detector)
        ↓
Contact Event Detection
        ↓
Appearance Embedding Extraction
        ↓
Offline Refinement (per contact event)
        ↓
Corrected Tracklets Output
4. Core Modules to Implement
4.1 Contact Event Detector

Detect frames/periods where animals are in close proximity.
Criteria (configurable):
Mask overlap ratio above a threshold, or
Center-to-center distance below a multiple of average animal size

Output: List of contact events, each containing:
start_frame, end_frame
list of involved track IDs during the event


4.2 Appearance Embedding Extractor

For a given detection (mask or tight crop), produce a fixed-size embedding vector.
Recommended starting point:
Use a small pretrained CNN backbone (ResNet18 or ResNet34)
Global average pooling → 512 or 2048-dimensional embedding

Optional: Allow fine-tuning later with contrastive/triplet loss on mouse identity data
Cache embeddings for efficiency

4.3 Offline Refiner (Main Logic – Option A)
For each contact event:

Collect the last reliable detections of involved animals before the contact starts.
Collect the first reliable detections of animals after the contact ends.
Compute pairwise cosine similarity between pre-contact and post-contact embeddings.
Build a cost matrix (1 – cosine_similarity).
Solve the assignment problem using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
Apply the resulting ID remapping to all detections after the contact event.
Enforce constraints:
No two animals can have the same ID in the same frame
Prefer spatially smooth trajectories (optional soft constraint)


4.4 Tracklet Manager / Data Structures

Represent tracklets as:
Dictionary or class containing per-frame detections
Each detection: frame_id, track_id, mask/bbox, embedding, confidence

Support efficient querying of detections before/after a given frame for specific track IDs

5. Integration with LabGym

Add the refinement as a post-processing step after the main Detector tracking is finished.
Preferred location: Analysis Module or a new post-processing utility.
Configuration options (YAML/JSON or existing LabGym config style):YAMLoffline_refinement:
  enabled: true
  contact_overlap_threshold: 0.3
  contact_distance_threshold: 1.5   # times average animal size
  embedding_dim: 512
  similarity_threshold: 0.6
  window_before: 15   # frames
  window_after: 15

6. Implementation Order (Recommended for the Agent)

Define data structures for tracklets and contact events.
Implement Contact Event Detector.
Implement Appearance Embedding Extractor (start with frozen pretrained backbone).
Implement the core Offline Refiner (Hungarian matching + ID remapping).
Add configuration loading and enable/disable flag.
Integrate into the main LabGym tracking pipeline as a post-processing step.
Add basic logging and visualization of corrected vs original IDs around contact events.
Write unit tests for the matching logic and constraint enforcement.

7. Testing Plan

Create or use a small set of videos containing wrestling/close-contact events with known ground-truth IDs.
Metrics to report:
Number of ID switches before vs after refinement
IDF1 / MOTA improvement (if possible)
Qualitative visualization of ID trajectories around contact events

Edge cases:
More than two animals involved
Very short contact
Animals that remain partially overlapping after separation


8. Success Criteria

The offline refinement correctly recovers the majority of identity switches caused by wrestling events.
The feature can be toggled on/off without breaking existing LabGym functionality.
Runtime overhead is acceptable for typical analysis videos.
Code is modular so that more advanced offline methods (graph optimization, learned matching) can be added later as alternative strategies.

9. Notes for the Agent

Prefer minimal invasive changes to the existing LabGym codebase.
Use existing Detectron2 masks for accurate animal crops when extracting embeddings.
Keep dependencies light (scipy for Hungarian matching is fine).
Make the ReID model easily swappable (so it can later be replaced with a fine-tuned mouse-specific model).
