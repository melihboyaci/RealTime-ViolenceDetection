# inference.md

Real-time and offline inference behavior.

## 1. Live Source Flow

The same per-frame preprocessor and feature builder used in offline preprocessing are reused. The online pipeline differs only in: (a) no FPS resampling, (b) a **FIFO buffer of 30 frames** instead of a sliding-window dataset, (c) no motion filter, (d) no backpropagation.

```mermaid
flowchart TD
    S[Live source\n(file or webcam)] --> R[Read next frame]
    R --> P[Frame preprocessor\nCLAHE? -> Gaussian 3x3 -> 640x640 -> BGR->RGB]
    P --> Y[YOLOv8n-Pose\n17 COCO keypoints]
    Y --> KF[Keypoint filter\nconf < 0.5 unreliable]
    KF --> MP[Multi-person selector\ntop 2, X-sort,\nnormalized bbox center distance]
    MP --> N[Skeletal normalizer\nhip centering + shoulder-hip scaling]
    N --> FV[69-dim feature vector]
    FV --> B[FIFO buffer length 30]
    B -->|len < 30| W[Warm-up: no decision]
    B -->|len = 30| F[GRU forward pass]
    F --> SG[Sigmoid score p]
    SG --> T{p >= 0.7?}
    T -- yes --> V[Decision: Violence]
    T -- no  --> NV[Decision: NonViolence]
    V --> VIS[Visualizer overlay]
    NV --> VIS
    W --> VIS
    VIS --> R
```

## 2. FIFO Buffer Behavior

- Capacity: **30 feature vectors** (one per processed frame).
- Type: ring / deque, length-bounded.
- On every new frame:
  1. Append the new 69-dim vector.
  2. If `len(buffer) > 30`, drop the oldest.
  3. If `len(buffer) == 30`, run a forward pass on the stacked tensor `(1, 30, 69)`.
  4. If `len(buffer) < 30`, emit a "warm-up" status; **no decision** is produced.

## 3. Per-Frame Processing Order

Strict, identical to the offline per-frame stage:

1. Conditional CLAHE on dark frames.
2. Gaussian blur 3×3.
3. Resize to 640×640.
4. BGR → RGB.
5. YOLOv8n-Pose, 17 COCO keypoints per person.
6. Drop / mark keypoints with confidence `< 0.5`.
7. Pick top-2 persons, **sort by X axis**, compute normalized bbox center distance.
8. Hip centering + shoulder–hip scaling per person.
9. Build the 69-dim feature vector.
10. Push into the FIFO buffer.

## 4. Decision Threshold Logic

- Default: `p ≥ 0.7` → **Violence**, else **NonViolence**.
- The threshold is **not** a model hyperparameter; it is a deployment knob.
- It may be retuned per use case using val-split metrics (see `evaluation.md`).
- Threshold tuning **must not** use the test split.

## 5. Visualization Rules

> Concrete UI is **TBD**. Recommended baseline:

- Always show the **current decision** as a textual label (e.g. top-left).
- Show the **sigmoid probability** to two decimals.
- Show a **warm-up** indicator while the buffer has fewer than 30 vectors.
- Optionally overlay the active skeletons (top-2 persons, X-sorted).
- Use a clearly distinct color/state for **Violence** vs **NonViolence**.

## 6. What to Display On Screen

| Element | Required | Notes |
|---|---|---|
| Decision label | yes | `Violence` / `NonViolence` / `warming up` |
| Sigmoid probability | recommended | `0.00`–`1.00` |
| Threshold value in use | recommended | for transparency |
| Skeleton overlay | optional | top-2 only |
| FPS / latency counter | optional | development aid |

## 7. Handling Insufficient Buffer Length

- While `len(buffer) < 30`: emit status `warming up`, render the optional skeleton overlay if available, **do not call** the GRU.
- This prevents both undefined behavior and noisy decisions made on padded sequences.
- Padding shorter buffers to length 30 is **not** done by default. *(If a future variant pads, it must document the padding strategy and re-run threshold ablation.)*

## 8. Timing / Latency Considerations

- The dominant cost per frame is **YOLOv8n-Pose**; the GRU forward on `(1, 30, 69)` is negligible by comparison.
- Concrete latency / FPS targets: **Not specified in the source PDF**.
- Recommended (TBD): measure per-stage time on the target hardware before promising any FPS number.
- The GRU forward is only invoked when the buffer is full, so warm-up frames are cheaper.

## 9. Failure Behavior — No Person Detected

- Frames with **0 persons**:
  - The pipeline must not crash.
  - Feature vector handling: **TBD** (e.g. zero vector or skip-and-pause-buffer). The chosen rule must match the offline pipeline exactly.
  - Visualizer should display a "no person" state alongside the current decision (or "warming up" if applicable).
- Frames with **1 person only**:
  - Person-2 placeholder rule is **TBD**.
  - The interaction feature (normalized bbox center distance) handling is **TBD**.

## 10. Pseudo-Code

```python
buffer = deque(maxlen=30)
THRESHOLD = 0.7

for frame in stream:
    pre = preprocess(frame)              # CLAHE? -> Gaussian 3x3 -> 640x640 -> BGR->RGB
    persons = yolov8n_pose(pre)          # 17 COCO keypoints per person
    persons = filter_keypoints(persons, conf_thresh=0.5)
    p1, p2 = select_top2_x_sorted(persons)   # may include placeholders if <2 persons
    inter_dist = normalized_bbox_center_distance(p1, p2)
    p1n = normalize_skeleton(p1)         # hip centering + shoulder-hip scaling
    p2n = normalize_skeleton(p2)
    fvec = build_feature_vector(p1n, p2n, inter_dist)   # 69-dim

    buffer.append(fvec)

    if len(buffer) < 30:
        render(frame, status="warming up")
        continue

    seq = stack(buffer)                  # shape (30, 69)
    p = model(seq.unsqueeze(0))          # sigmoid, shape (1,)
    decision = "Violence" if p.item() >= THRESHOLD else "NonViolence"
    render(frame, status=decision, prob=p.item(), threshold=THRESHOLD)
```

## 11. Offline Inference (Batch Mode)

- Used for evaluation and ablations on the test split.
- Same per-frame pipeline, but instead of a FIFO, the offline sequence builder produces all 30-frame windows for a video, the model scores them in batches, and metrics are aggregated per window.
- The motion filter is **not applied** during evaluation; only during training data preparation.

## 12. Assumptions / Open Questions

- Whether consecutive Violence decisions are smoothed (e.g. require N consecutive positives): **TBD.**
- Whether warm-up frames are visually distinct from NonViolence: **TBD.**
- Behavior with partial occlusion that yields 2 persons but many sub-threshold keypoints: **TBD.**
- Exact UI / overlay specification: **TBD.**
