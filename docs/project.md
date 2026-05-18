# project.md

## Türkçe Özet

İskelet (poz) tabanlı, gerçek zamanlı bir **şiddet tespit** sistemidir. Ham video, **çevrimdışı ön işleme** (offline preprocessing) ile çıkarılan iskelet özelliklerine dönüştürülür; ardından **çevrimiçi çıkarım** (online inference) aşamasında bir GRU sınıflandırıcısı ile her 30 karelik pencere için **Violence / NonViolence** kararı üretilir. Stack: OpenCV, YOLOv8n-Pose, PyTorch, GRU.

---

## 1. Problem Statement

Surveillance and live-stream platforms need a low-latency way to flag physical violence in video. Pixel-based action recognition models are heavy, sensitive to appearance (clothing, lighting, background), and prone to spurious correlations. The project addresses this by reducing each frame to a compact **skeletal feature vector** and classifying short temporal **sequence windows** of skeletal motion as **Violence** or **NonViolence**.

## 2. Goal

Build an end-to-end pipeline that:

1. Standardizes raw videos and extracts pose-based features **offline**.
2. Trains a **GRU sequence classifier** on fixed-length skeletal sequences.
3. Runs **online inference** on a live stream (file or webcam) using a **FIFO buffer of 30 frames** and a configurable **decision threshold** (initial value `0.7`).
4. Produces a per-window binary label: **Violence** or **NonViolence**.

## 3. High-Level Architecture Summary

The system is split into two strictly separated phases:

- **Offline preprocessing** — runs once over the dataset, produces cached `.npy` artifacts:
  raw video → 10 FPS sampling → frame preprocessing (conditional CLAHE → Gaussian 3×3 → resize 640×640 → BGR→RGB) → **YOLOv8n-Pose** (COCO 17 keypoints) → keypoint confidence filter (`≥ 0.5`) → multi-person selection (top 2, X-sorted, normalized bbox center distance) → **69-dim feature vector** per frame → hip-centering + shoulder–hip scaling → 30-frame sliding **sequence windows** → motion-based filtering for Violence windows (θ = 0.05).
- **Online inference** — runs in real time on a live source: same per-frame preprocessing + pose extraction + 69-dim feature vector → push into a **FIFO buffer of 30 frames** → GRU forward pass → sigmoid score → threshold decision → on-screen visualization.

## 4. Tech Stack

| Layer | Choice |
|---|---|
| Video I/O & frame ops | OpenCV |
| Pose estimation | YOLOv8n-Pose (COCO 17 keypoints) |
| Modeling / training | PyTorch |
| Sequence model | GRU (sigmoid head, BCELoss, Adam) |
| Dataset | Real Life Violence Situations Dataset (Kaggle), 2000 videos |

## 5. What Makes This Project Non-Trivial

- **Two-phase design**: cached pose features decouple heavy vision work from fast sequence learning.
- **Multi-person handling**: top-2 selection with X-axis sorting and a normalized inter-person distance feature, despite known instability during overlap.
- **Label-noise mitigation**: video-level Violence labels are noisy at the window level, so a **motion filter** (θ = 0.05) drops low-motion Violence windows before training.
- **Pose-only representation**: 69-dim per-frame vector and skeletal normalization (hip centering + shoulder–hip scaling) make the model robust to appearance and resolution.
- **Real-time constraint**: a 30-frame FIFO buffer + GRU enables continuous classification on a live feed.

## 6. Expected Outputs

- Cached per-video `.npy` feature arrays produced by the offline pipeline.
- 30-frame **sequence window** datasets for train / val / test (stratified 70 / 15 / 15).
- A trained GRU checkpoint (best model selected by **lowest val_loss**).
- Evaluation artifacts: confusion matrix, precision, recall, F1, AUC-ROC, plus ablations.
- Online inference loop producing a per-window **Violence / NonViolence** label and an on-screen overlay.

## 7. Executive Summary

This project converts raw video into a binary **Violence / NonViolence** decision using a pose-first design. **Offline preprocessing** standardizes videos to 10 FPS, applies conditional CLAHE, Gaussian 3×3 blur, 640×640 resize, and BGR→RGB conversion, then runs **YOLOv8n-Pose** to obtain COCO 17 keypoints, filters keypoints under `0.5` confidence, picks the top 2 people sorted by X axis, encodes a normalized inter-person distance, and emits a **69-dimensional feature vector** per frame normalized via hip centering and shoulder–hip scaling. Frames are grouped into **30-frame sliding sequence windows**; Violence windows pass through a motion filter (θ = 0.05) to suppress label noise. A **GRU classifier** (sigmoid output, BCELoss, Adam) is trained for up to 100 epochs with batch size 32 and early stopping on val_loss, with the best model saved by lowest val_loss. **Online inference** mirrors the per-frame pipeline, maintains a **FIFO buffer of 30 frames**, and emits a thresholded decision (initial threshold `0.7`).

## 8. Assumptions / Open Questions

- Inference hardware target (GPU/CPU, FPS budget): **Not specified in the source PDF.**
- Exact GRU layer counts, hidden size, dropout, and parameter count: **Not specified in the source PDF.**
- Scheduler (if any) beyond Adam: **Not specified in the source PDF.**
- Visualization details (colors, labels, font sizes) on the inference overlay: **TBD.**
