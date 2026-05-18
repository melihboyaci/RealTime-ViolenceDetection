# Project Log

---
### [2026-05-18 12:25:00] — Antigravity / CLI Execution
* **Action/Task:** Created the `agent_workflow.md` rules document to orchestrate AI agent activities.
* **Files Affected:** `docs/rules/agent_workflow.md`
* **Details/Decisions:** Defined a strict 6-step execution loop that mandates agents to read `ai_instructions.md`, check `state.md`, update `project-plan.md`, enforce `coding_standards.md`, and log to `project_log.md` via `logging_standarts.md` for every task.
* **Issues & Resolutions:** None

---
### [2026-05-18 14:16:00] — P0 Project Bootstrap
* **Action/Task:** Created repository skeleton, config, requirements, and Python virtual environment.
* **Files Affected:** `configs/config.py`, `configs/__init__.py`, `src/__init__.py`, `requirements.txt`, `data/`, `models/`, `notebooks/`
* **Details/Decisions:** All locked constants from decision_log.md encoded in config.py. Venv created with PyTorch 2.12.0 (CPU), OpenCV 4.13.0, Ultralytics 8.4.51. CUDA PyTorch to be installed for training phase.
* **Issues & Resolutions:** Local PyTorch is CPU-only; will install CUDA version before training.

---
### [2026-05-18 14:20:00] — P1+P2 Kaggle Preprocessing Notebook
* **Action/Task:** Created complete Kaggle Notebook for offline preprocessing pipeline.
* **Files Affected:** `notebooks/kaggle_preprocessing.ipynb` (14 cells)
* **Details/Decisions:** Full pipeline: FPS sampling (10 FPS) → CLAHE → GaussianBlur → Resize → BGR→RGB → YOLOv8n-Pose → Multi-person B+ → Hip centering → Shoulder-hip scaling → 69-dim .npy → Stratified split 70/15/15 → Sliding window (30, stride 15) → Motion filter (θ=0.05 Violence only) → Sequences. Includes NonViolence undersampling for train balance.
* **Issues & Resolutions:** Kaggle free tier GPU requires phone verification — user completed verification.

---
### [2026-05-18 14:30:00] — P3+P4+P5+P6 Local Pipeline Code
* **Action/Task:** Created GRU model, training, evaluation, and inference scripts.
* **Files Affected:** `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py`
* **Details/Decisions:** ViolenceGRU: 115,777 params, 2-layer GRU (128→64), Dropout 0.3, Dense 32, Sigmoid. Smoke test passed: (32,30,69)→(32,1). Training: BCELoss, Adam, ReduceLROnPlateau, gradient clipping, early stopping. Evaluation: CM, P, R, F1, AUC-ROC. Inference: FIFO-30 buffer, webcam/video support, on-screen overlay.
* **Issues & Resolutions:** None