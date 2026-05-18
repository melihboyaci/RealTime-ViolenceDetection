# README.md

## Türkçe Özet

Bu depo, **iskelet (poz) tabanlı, gerçek zamanlı şiddet tespiti** projesinin dokümantasyonunu içerir. Sistem iki aşamadan oluşur: **çevrimdışı ön işleme** (offline preprocessing) ham videodan kareleri 10 FPS'e indirir, koşullu CLAHE + 3×3 Gauss + 640×640 yeniden boyutlandırma + BGR→RGB uygular, **YOLOv8n-Pose** ile 17 COCO eklem noktasını çıkarır, en üst 2 kişiyi X eksenine göre sıralar, **69 boyutlu** kare öznitelik vektörü üretir ve **30 karelik kayan pencerelere** böler; Şiddet pencereleri **θ = 0.05** hareket filtresinden geçer. **Çevrimiçi çıkarım** (online inference) aşamasında aynı kare hattı kullanılır, **30 kare uzunluğunda FIFO** tampon dolduğunda **GRU** modeli sigmoid skoru üretir, **eşik = 0.7** ile **Violence / NonViolence** kararı verilir. Kaynak veri kümesi: **Real Life Violence Situations Dataset** (Kaggle, 2000 video). Bölme: **stratifiye 70 / 15 / 15**. Yığın: OpenCV, YOLOv8n-Pose, PyTorch, GRU.

---

## 1. Overview

A real-time violence detection system based on **skeletal pose analysis**. Each frame is reduced to a 69-dimensional pose feature vector; sequences of 30 such vectors are classified as **Violence** or **NonViolence** by a GRU. The pipeline is split into **offline preprocessing** (cached `.npy` features) and **online inference** (FIFO buffer of 30 frames + threshold).

## 2. Project Goal

Produce a binary **Violence / NonViolence** classifier that runs in real time on a live video source, trained on cached pose features extracted offline from the **Real Life Violence Situations Dataset** (Kaggle, **2000 videos**), using a **GRU** sequence model with **BCELoss + Adam**, with the deployment threshold initially set to **0.7**.

## 3. Main Modules

| Module | Phase | Doc |
|---|---|---|
| Frame preprocessor (CLAHE? → 3×3 → 640×640 → BGR→RGB) | offline + online | `architecture.md`, `data_pipeline.md` |
| Pose extractor (YOLOv8n-Pose, 17 COCO keypoints) | offline + online | `data_pipeline.md` |
| Multi-person selector (top 2, X-sort, normalized bbox center distance) | offline + online | `data_pipeline.md`, `architecture.md` |
| Feature builder (69-dim, hip centering + shoulder–hip scaling) | offline + online | `data_pipeline.md` |
| Sequence builder (30-frame sliding window) | offline | `data_pipeline.md` |
| Motion filter (θ = 0.05, Violence only) | offline | `data_pipeline.md`, `decision_log.md` |
| GRU classifier (sigmoid, BCELoss, Adam) | training + online | `model.md`, `training.md` |
| Trainer (max 100 epochs, batch 32, early stop on val_loss) | training | `training.md` |
| Evaluator (CM, precision, recall, F1, AUC-ROC) | evaluation | `evaluation.md` |
| FIFO + decision (FIFO 30, threshold 0.7) | online | `inference.md` |

## 4. How the Docs Are Organized

| File | Purpose |
|---|---|
| `project.md` | Vision, problem, goal, exec summary |
| `state.md` | Current state, locked decisions, open questions |
| `project-plan.md` | Phased roadmap and checklist |
| `specs.md` | Numbered functional / non-functional / data / model / inference requirements |
| `architecture.md` | Component boundaries + Mermaid flow diagrams |
| `data_pipeline.md` | Full data lifecycle, normalization, sequencing, motion filter |
| `model.md` | GRU specification and ablation hooks |
| `training.md` | Training playbook and reproducibility notes |
| `inference.md` | Real-time + offline inference, FIFO behavior, pseudo-code |
| `evaluation.md` | Metrics, threshold analysis, four-ablation suite |
| `limitations.md` | Accepted trade-offs, known issues vs. bugs |
| `ai_instructions.md` | **Required reading** for any AI agent |
| `decision_log.md` | Authoritative locked decisions table |
| `README.md` | This file |

## 5. Recommended Reading Order for New Contributors

1. `README.md` (you are here)
2. `ai_instructions.md` — non-negotiable rules for agents and humans
3. `project.md` — what the system does and why
4. `state.md` — current locked decisions and open questions
5. `decision_log.md` — the source of truth for "what is fixed"
6. `architecture.md` — overall flow
7. The doc closest to your task: `data_pipeline.md`, `model.md`, `training.md`, `inference.md`, `evaluation.md`, `limitations.md`, or `specs.md`

## 6. Suggested Next Steps for Implementation

> Detailed phasing lives in `project-plan.md`. The short version:

1. **P0** — Bootstrap the repo, install OpenCV / Ultralytics (YOLOv8n-Pose) / PyTorch, place the dataset.
2. **P1** — Implement the offline pipeline up to per-video `(N, 69)` `.npy` artifacts.
3. **P2** — Build the **30-frame sliding window** dataset; apply **motion filter θ = 0.05** to Violence windows; persist the **stratified 70 / 15 / 15** split.
4. **P3** — Implement the GRU classifier (`(batch, 30, 69)` → sigmoid).
5. **P4** — Train with **BCELoss + Adam, batch 32, max 100 epochs, early stopping on val_loss, best by lowest val_loss**.
6. **P5** — Evaluate on the **isolated test split**: confusion matrix, precision, recall, F1, AUC-ROC.
7. **P6** — Build the online inference loop with **FIFO 30 frames** and **threshold 0.7**.
8. **P7** — Run the four ablations (threshold, interaction-feature, normalization, motion-filter θ); tune the deployment threshold on the val split per use case.

## 7. Doc-Bundle Status

This documentation bundle reflects only what is in the source PDF. Anything missing is marked **TBD** or **"Not specified in the source PDF."** Future contributors must keep that discipline (see `ai_instructions.md` §11).
