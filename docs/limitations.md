# limitations.md

Known trade-offs of the current design. These are accepted limitations, not bugs, unless explicitly marked.

## 1. X-Axis Sorting Instability

- **What it is**: the multi-person selector picks the top 2 persons and sorts them by X axis to keep a stable identity. When two persons overlap horizontally (e.g. crossing or hugging), their X coordinates can swap, causing person 1 and person 2 to flip in the feature vector across consecutive frames.
- **Impact**: short-term feature noise inside the 30-frame window; in the worst case, partial degradation of the **normalized bbox center distance** signal during overlap.
- **Why accepted**: a robust identity tracker would substantially increase complexity and latency; the GRU is somewhat tolerant to short flips at 10 FPS.
- **Status**: **known limitation**, not a bug.
- **Future improvement**: a lightweight identity tracker (e.g. IoU-based or ByteTrack) could be slotted in front of the multi-person selector.

## 2. Video-Level Label Noise

- **What it is**: labels in the Real Life Violence Situations Dataset are assigned **per video**, not per window. A "Violence" video can contain calm pre/post moments that propagate the Violence label to low-motion 30-frame windows.
- **Impact**: noisy positive examples during training; harder convergence and less reliable threshold tuning.
- **Mitigation in baseline**: the **motion filter** with **θ = 0.05** removes low-motion Violence windows before training.
- **Why accepted**: the dataset only ships video-level labels; manual window-level relabeling is out of scope.
- **Status**: **known limitation**, not a bug.
- **Future improvement**: weak supervision, multi-instance learning, or a curated subset with window-level labels.

## 3. Zero-Person Frames

- **What it is**: frames in which YOLOv8n-Pose detects no person. The 69-dim feature vector has no skeleton to encode.
- **Impact**: such frames need an explicit handling rule; otherwise the FIFO buffer holds undefined data and decisions become unreliable.
- **Status**: handling strategy is **TBD** but mandatory; same rule must be used offline and online. Until the rule is locked, this is a **bug-class risk**, not a benign limitation.
- **Future improvement**: explicit zero-fill with a "no skeleton" flag dimension, or buffer-pause semantics during person-less stretches.

## 4. Single-Source Dataset / Domain Shift

- **What it is**: the model is trained on a single Kaggle dataset sourced predominantly from YouTube clips. Camera angles, resolutions, lighting, and behaviors are skewed toward that source.
- **Impact**: out-of-distribution drops on real surveillance feeds (low light, top-down camera, fish-eye lenses, low FPS sources).
- **Why accepted**: the project's scope is to validate the pose-based approach end-to-end; broader generalization is a follow-up.
- **Status**: **known limitation**, not a bug.
- **Future improvement**: multi-source training data, domain adaptation, or fine-tuning on target-deployment footage.

## 5. Pose-Only Representation

- **What it is**: the model never sees pixels; it sees only normalized skeletal coordinates plus an inter-person distance feature.
- **Impact**: actions that look like Violence in pixels but not in skeletons (e.g. weapons, blood) are invisible to the model. Conversely, vigorous non-Violent activities (sports, dance) can superficially resemble Violence skeletons.
- **Why accepted**: this is the explicit design choice; pose features are robust, lightweight, and privacy-friendlier.
- **Status**: **known limitation by design**.

## 6. Two-Person Cap

- **What it is**: the multi-person selector keeps only the top 2 persons.
- **Impact**: scenes with more than 2 active participants lose information about additional people.
- **Why accepted**: keeps the per-frame feature vector at a fixed **69 dims** and bounds latency.
- **Status**: **known limitation by design**.

## 7. Fixed 30-Frame Window

- **What it is**: every decision is based on exactly **30 frames** at 10 FPS (~3 seconds).
- **Impact**: events shorter than the window are diluted; longer events are inferred from consecutive overlapping windows.
- **Why accepted**: a fixed length keeps the GRU input shape stable and the FIFO buffer simple.
- **Status**: **known limitation by design**.

## 8. Sub-Threshold Keypoints

- **What it is**: keypoints with confidence `< 0.5` are flagged unreliable. The exact replacement strategy (zero-fill, last-known, interpolation) is **TBD**.
- **Impact**: until locked, two implementations could disagree on the same input.
- **Status**: **bug-class risk** until the strategy is locked in `decision_log.md`. Same rule must be applied offline and online.

## 9. Initial Threshold Is Not Empirical

- **What it is**: the **0.7** initial decision threshold is a starting point, not the result of an evaluation sweep.
- **Impact**: the deployed operating point should be re-tuned per use case using val-split metrics.
- **Status**: **known limitation by design**; see `evaluation.md` §5 and §7.

## 10. Known Limitations vs. Bugs

| Item | Type |
|---|---|
| X-axis sorting instability | known limitation |
| Video-level label noise | known limitation (mitigated by motion filter) |
| Zero-person frames | **bug-class risk** until rule is locked |
| Single-source domain shift | known limitation |
| Pose-only representation | known limitation by design |
| Two-person cap | known limitation by design |
| Fixed 30-frame window | known limitation by design |
| Sub-threshold keypoint replacement | **bug-class risk** until rule is locked |
| Non-empirical initial threshold | known limitation by design |

## 11. Future Improvements (Non-Exhaustive)

- Add a lightweight identity tracker upstream of the multi-person selector.
- Replace video-level labels with window-level labels on a curated subset.
- Lock and document zero-person and sub-threshold-keypoint handling.
- Multi-source training data and / or domain adaptation.
- Optional pixel-side branch (e.g. weapon detection) fused with the pose-side decision.
- Temporal smoothing across consecutive Violence decisions in online inference.

## 12. Assumptions / Open Questions

- Whether identity tracking is in scope for v2: **TBD.**
- Whether the project ever leaves the pose-only design: **TBD.**
- Acceptable false-alarm budget for deployment: **Not specified in the source PDF.**
