# specs.md

Formal, numbered requirements for the violence detection system. Implementations and reviews must reference these IDs.

## 1. Functional Requirements (FR)

- **FR-1** The system shall classify a 30-frame skeletal sequence window into one of two classes: **Violence** or **NonViolence**.
- **FR-2** The system shall provide an **offline preprocessing** pipeline that reads raw video files and outputs cached `.npy` feature artifacts.
- **FR-3** The system shall provide an **online inference** pipeline that consumes a live source (file or webcam) and emits per-window classifications.
- **FR-4** The offline pipeline shall standardize input video to **10 FPS**.
- **FR-5** The frame preprocessor shall apply, in this order: **conditional CLAHE on dark frames → Gaussian blur 3×3 → resize 640×640 → BGR→RGB**.
- **FR-6** Pose estimation shall use **YOLOv8n-Pose** producing **COCO 17 keypoints** per person.
- **FR-7** Keypoints with confidence below **0.5** shall be treated as unreliable.
- **FR-8** The multi-person selector shall pick the **top 2** detected people, **sort them by X axis**, and compute a **normalized bbox center distance** as an interaction feature.
- **FR-9** The per-frame feature vector shall be **69-dimensional**.
- **FR-10** Skeletal normalization shall apply **hip centering** and **shoulder–hip scaling**.
- **FR-11** Sequences shall be built using a **30-frame sliding window**.
- **FR-12** Violence sequences shall be filtered by a **motion filter with θ = 0.05** to reduce label noise.
- **FR-13** The dataset split shall be **stratified 70 / 15 / 15** train / val / test.
- **FR-14** The model shall be a **GRU sequence classifier** with a **sigmoid** output.
- **FR-15** Training shall use **BCELoss** and the **Adam** optimizer.
- **FR-16** Training shall run for **at most 100 epochs**, with **batch size 32** and **early stopping on val_loss**.
- **FR-17** The training procedure shall save the **best checkpoint by lowest val_loss**.
- **FR-18** Online inference shall maintain a **FIFO buffer of 30 frames** of feature vectors.
- **FR-19** The decision threshold shall default to **0.7** on the sigmoid output.
- **FR-20** Evaluation shall report **confusion matrix, precision, recall, F1, and AUC-ROC** on the **isolated test split**.

## 2. Non-Functional Requirements (NFR)

- **NFR-1** Reproducibility: all hyperparameters used by a training run shall be persisted alongside the checkpoint. *(Exact format TBD.)*
- **NFR-2** Determinism: random seeds for splits shall be fixed where supported. *(Exact seed TBD.)*
- **NFR-3** Online inference shall run in real time on a live source. *(Concrete FPS budget: Not specified in the source PDF.)*
- **NFR-4** Offline and online phases shall be **strictly separated**: offline must not depend on a live source; online must not write to the training dataset.
- **NFR-5** All cached artifacts shall be addressable by a stable, content-derived path (e.g. `<split>/<class>/<video_id>.npy`). *(Exact scheme TBD.)*
- **NFR-6** Code shall use the same per-frame feature builder in both offline and online phases.

## 3. Data Requirements (DR)

- **DR-1** The source dataset shall be the **Real Life Violence Situations Dataset** from Kaggle (**2000 videos** total).
- **DR-2** The split shall be **stratified 70 / 15 / 15**.
- **DR-3** Each preprocessed video shall yield a `(N, 69)` feature array, where `N` is the number of sampled frames at 10 FPS.
- **DR-4** Each sequence window shall have shape `(30, 69)`.
- **DR-5** Class labels shall be encoded as binary: `Violence = 1`, `NonViolence = 0`. *(Exact integer mapping TBD if not in source PDF.)*
- **DR-6** The motion filter shall apply **only** to Violence windows.

## 4. Model Requirements (MR)

- **MR-1** The model shall be a GRU-based sequence classifier.
- **MR-2** The model input shape shall be `(batch, 30, 69)`.
- **MR-3** The model output shall be a single sigmoid probability `p ∈ [0, 1]` representing the probability of class **Violence**.
- **MR-4** The loss function shall be **BCELoss**.
- **MR-5** The optimizer shall be **Adam**.
- **MR-6** GRU layer count, hidden size, dropout, and learning rate are **Not specified in the source PDF** and shall be documented in `decision_log.md` once chosen.

## 5. Inference Requirements (IR)

- **IR-1** The online pipeline shall reuse the offline per-frame preprocessor and feature builder verbatim.
- **IR-2** A **FIFO buffer of length 30** shall hold the most recent feature vectors.
- **IR-3** The model shall be invoked once per new frame **after** the buffer is full.
- **IR-4** A sigmoid score `≥ 0.7` (initial threshold) shall classify the current window as **Violence**.
- **IR-5** When the buffer is not yet full, the system shall **not** emit a Violence/NonViolence decision; behavior in this state is documented in `inference.md`.
- **IR-6** When no person is detected in the current frame, the system shall handle the case explicitly (see `inference.md`).
- **IR-7** The inference UI shall display the current decision and may display the sigmoid probability. *(Exact UI rules TBD.)*

## 6. Performance Constraints (PC)

- **PC-1** The system shall be designed for real-time operation on a live source. *(Hardware target: Not specified in the source PDF.)*
- **PC-2** Per-frame end-to-end latency budget: **Not specified in the source PDF.**
- **PC-3** Memory footprint constraints: **Not specified in the source PDF.**
- **PC-4** The offline pipeline shall not block the live pipeline.

## 7. Failure Handling Requirements (FH)

- **FH-1** **Zero-person frames** shall not crash the pipeline. The exact placeholder strategy (zero-fill or skip) is **TBD**; once chosen, it shall be applied identically offline and online.
- **FH-2** **Single-person frames** (only 1 of 2 detected) shall not crash the pipeline. Person 2 placeholder strategy is **TBD**.
- **FH-3** **Sub-threshold keypoints** (`< 0.5`) shall be treated as unreliable. The exact replacement value is **TBD**.
- **FH-4** **Short videos** that produce fewer than 30 sampled frames shall be excluded from sequence assembly with a logged reason.
- **FH-5** **Corrupt or unreadable videos** shall be skipped with a logged warning, never aborting the pipeline.
- **FH-6** **Online buffer underflow** (fewer than 30 frames) shall not produce a decision; see `inference.md`.

## 8. Acceptance Criteria (AC)

- **AC-1** Running the offline pipeline on the full dataset produces one `.npy` per usable video, conforming to DR-3.
- **AC-2** The sequence dataset contains windows of shape `(30, 69)`, with motion-filtering applied only to Violence (FR-12).
- **AC-3** A training run with the locked hyperparameters (FR-15, FR-16) reaches early stopping and saves a best-by-lowest-val_loss checkpoint (FR-17).
- **AC-4** Evaluation on the test split reports all five metrics in FR-20.
- **AC-5** The four ablations (threshold, interaction-feature, normalization, motion-filter θ) listed in `evaluation.md` are runnable and produce comparable reports.
- **AC-6** Online inference on a sample source emits per-window decisions using FIFO 30 + threshold 0.7 without crashing on zero-person frames.

## 9. Assumptions / Open Questions

- Concrete latency / FPS targets: **Not specified in the source PDF.**
- Final on-disk artifact layout: **TBD.**
- UI specification for the inference overlay: **TBD.**
- Exact placeholder values for missing keypoints / persons: **TBD.**
