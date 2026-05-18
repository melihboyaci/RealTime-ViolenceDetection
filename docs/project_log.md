# Project Log

---

### [2026-05-18 12:25:00] — Antigravity / CLI Execution

- **Action/Task:** Created the `agent_workflow.md` rules document to orchestrate AI agent activities.
- **Files Affected:** `docs/rules/agent_workflow.md`
- **Details/Decisions:** Defined a strict 6-step execution loop that mandates agents to read `ai_instructions.md`, check `state.md`, update `project-plan.md`, enforce `coding_standards.md`, and log to `project_log.md` via `logging_standarts.md` for every task.
- **Issues & Resolutions:** None

---

### [2026-05-18 14:16:00] — P0 Project Bootstrap

- **Action/Task:** Created repository skeleton, config, requirements, and Python virtual environment.
- **Files Affected:** `configs/config.py`, `configs/__init__.py`, `src/__init__.py`, `requirements.txt`, `data/`, `models/`, `notebooks/`
- **Details/Decisions:** All locked constants from decision_log.md encoded in config.py. Venv created with PyTorch 2.12.0 (CPU), OpenCV 4.13.0, Ultralytics 8.4.51. CUDA PyTorch to be installed for training phase.
- **Issues & Resolutions:** Local PyTorch is CPU-only; will install CUDA version before training.

---

### [2026-05-18 14:20:00] — P1+P2 Kaggle Preprocessing Notebook

- **Action/Task:** Created complete Kaggle Notebook for offline preprocessing pipeline.
- **Files Affected:** `notebooks/kaggle_preprocessing.ipynb` (14 cells)
- **Details/Decisions:** Full pipeline: FPS sampling (10 FPS) → CLAHE → GaussianBlur → Resize → BGR→RGB → YOLOv8n-Pose → Multi-person B+ → Hip centering → Shoulder-hip scaling → 69-dim .npy → Stratified split 70/15/15 → Sliding window (30, stride 15) → Motion filter (θ=0.05 Violence only) → Sequences. Includes NonViolence undersampling for train balance.
- **Issues & Resolutions:** Kaggle free tier GPU requires phone verification — user completed verification.

---

### [2026-05-18 14:30:00] — P3+P4+P5+P6 Local Pipeline Code

- **Action/Task:** Created GRU model, training, evaluation, and inference scripts.
- **Files Affected:** `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py`
- **Details/Decisions:** ViolenceGRU: 115,777 params, 2-layer GRU (128→64), Dropout 0.3, Dense 32, Sigmoid. Smoke test passed: (32,30,69)→(32,1). Training: BCELoss, Adam, ReduceLROnPlateau, gradient clipping, early stopping. Evaluation: CM, P, R, F1, AUC-ROC. Inference: FIFO-30 buffer, webcam/video support, on-screen overlay.
- **Issues & Resolutions:** None

---

### [2026-05-18 15:04:00] — Kaggle Notebook Compatibility Fix

- **Action/Task:** Fixed Kaggle import compatibility issue in `notebooks/kaggle_preprocessing.ipynb`.
- **Files Affected:** `notebooks/kaggle_preprocessing.ipynb`
- **Details/Decisions:** Kaggle's notebook parser fails to read nbformat 4.5 (nbformat_minor=5) with cell `id` fields. Downgraded to nbformat_minor=4 and removed all 8 cell `id` keys. Notebook now has 8 clean cells (1 markdown + 7 code) with proper structure for Kaggle import.
- **Issues & Resolutions:** Kaggle was showing incomplete/malformed cells; fixed by removing `"id"` fields and setting `nbformat_minor": 4`.

---

### [2026-05-18 18:04:00] — P4 Training Execution

- **Action/Task:** Ran GRU model training on preprocessed sequences (CPU).
- **Files Affected:** `models/best_model.pt`, `models/training_log.csv`, `data/sequences/{train,val,test}/`
- **Details/Decisions:** Moved .npy files into split subdirectories to match `dataset.py` expectations. Training ran 18 epochs before early stopping (patience=10). Best val_loss=0.4837 at epoch 8. Val accuracy peaked at ~79.9%. LR reduced from 1e-3 to 5e-4 at epoch 14. Gradient clipping active. No outlier clipping applied to data.
- **Issues & Resolutions:** Overfitting observed from epoch 9 onwards (train_loss kept decreasing but val_loss plateaued/increased). Early stopping triggered correctly at epoch 18. Best checkpoint saved at epoch 8.

---

### [2026-05-18 18:15:00] — P5 Evaluation + P7.1 Threshold Ablation

- **Action/Task:** Evaluated best model on test split. Ran threshold ablation (0.3–0.9). Tested 4x data augmentation (jitter, temporal shift, reverse) — no improvement, reverted to original data and retrained.
- **Files Affected:** `models/best_model.pt`, `models/evaluation_results.txt`, `models/training_curves.png`, `scripts/threshold_ablation.py`, `scripts/augment_and_retrain.py`
- **Details/Decisions:** Final model (best val_loss=0.4854, epoch 7). Test results at threshold=0.4: Precision=0.7435, Recall=0.9314, F1=0.8269, AUC-ROC=0.9210, Accuracy=85.4%. Threshold ablation showed best F1 at t=0.4 (not locked 0.7). Augmented training (13K sequences) showed worse overfitting with no F1 gain — reverted to original 3283 sequences.
- **Issues & Resolutions:** Data augmentation (Gaussian jitter + temporal shift + reverse) caused faster overfitting without improving generalization. Root cause: augmented samples too similar to originals. Reverted to original data.

---

### [2026-05-18 18:45:00] — P6 Inference Test + Docs Finalization

- **Action/Task:** Tested real-time inference via webcam; updated deployment threshold and all project documentation.
- **Files Affected:** `configs/config.py`, `docs/evaluation.md`, `docs/decision_log.md`, `docs/project-plan.md`
- **Details/Decisions:** DECISION_THRESHOLD updated from 0.7 to 0.4 (val-tuned). Installed `ultralytics` locally for inference. Webcam inference verified working (FIFO buffer fills, decisions rendered on screen). `evaluation.md` results table filled with actual metrics. `project-plan.md` P7.1 marked done. `decision_log.md` §3 updated with deployed threshold rationale.
- **Issues & Resolutions:** `ultralytics` was not installed locally (only used on Kaggle previously); installed via pip. Inference runs on CPU (~5-10 FPS with YOLOv8n-Pose).
