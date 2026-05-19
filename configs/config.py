"""
Project-wide configuration constants for RealTime-ViolenceDetection.

All locked decisions from decision_log.md (D1–D20) are encoded here.
Do NOT change any value without first updating docs/decision_log.md
and re-running the relevant ablation.
"""


# ── D1: FPS Standardization ─────────────────────────────────────
TARGET_FPS = 10

# ── D2: Frame Preprocessing ─────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_BRIGHTNESS_THRESHOLD = 50
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 0
RESIZE_DIM = (640, 640)

# ── D3: Pose Model ──────────────────────────────────────────────
POSE_MODEL_NAME = "yolov8n-pose.pt"
NUM_KEYPOINTS = 17

# ── D12: Keypoint Confidence ────────────────────────────────────
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# ── D4 / D5 / D6: Multi-Person Strategy ────────────────────────
MAX_PERSONS = 2

# ── D13: Feature Vector ────────────────────────────────────────
FEATURE_DIM = 69
SKELETON_DIM = 34  # 17 keypoints × 2 coordinates per person

# ── D7: Sequence Window ────────────────────────────────────────
SEQUENCE_LENGTH = 30
SLIDING_WINDOW_STRIDE_TRAIN = 15
SLIDING_WINDOW_STRIDE_INFERENCE = 1

# ── D11: Motion Filter (Violence Only) ─────────────────────────
MOTION_FILTER_THETA = 0.05

# ── D9: Split Strategy ─────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_RANDOM_STATE = 42

# ── D8: GRU Model Architecture ─────────────────────────────────
GRU_INPUT_SIZE = FEATURE_DIM  # 69
GRU_HIDDEN_SIZE_1 = 128
GRU_HIDDEN_SIZE_2 = 64
GRU_NUM_LAYERS_1 = 1
GRU_NUM_LAYERS_2 = 1
DROPOUT_RATE = 0.3
DENSE_UNITS = 32

# ── D15 / D16 / D17: Training ──────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0

# ── LR Scheduler ────────────────────────────────────────────────
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

# ── D10 / D18: Inference ───────────────────────────────────────
FIFO_BUFFER_LENGTH = SEQUENCE_LENGTH  # 30
SUSPICIOUS_THRESHOLD = 0.35  # p >= 0.35 → Suspicious (yellow)
DECISION_THRESHOLD = 0.45    # p >= 0.45 → Violence (red)

# ── Temporal Smoothing ─────────────────────────────────────────
SMOOTHING_WINDOW = 3         # Son N kararı tut
SMOOTHING_MIN_COUNT = 2      # N'den en az kaçı Violence/Suspicious olmalı

# ── Entry Suppression ──────────────────────────────────────────
ENTRY_SUPPRESSION_FRAMES = 30  # Yeni kişi belirdiğinde N frame boyunca karar verme

# ── Class Labels ────────────────────────────────────────────────
VIOLENCE_LABEL = 1
NONVIOLENCE_LABEL = 0

# ── Paths (Lokal) ──────────────────────────────────────────────
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
FEATURES_DIR = f"{DATA_DIR}/features"
SPLITS_DIR = f"{DATA_DIR}/splits"
SEQUENCES_DIR = f"{DATA_DIR}/sequences"
MODELS_DIR = "models"
BEST_MODEL_PATH = f"{MODELS_DIR}/best_model.pt"

# ── Paths (Kaggle Notebook) ────────────────────────────────────
KAGGLE_INPUT_DIR = "/kaggle/input/real-life-violence-situations-dataset"
KAGGLE_VIOLENCE_DIR = f"{KAGGLE_INPUT_DIR}/Real Life Violence Dataset/Violence"
KAGGLE_NONVIOLENCE_DIR = f"{KAGGLE_INPUT_DIR}/Real Life Violence Dataset/NonViolence"
KAGGLE_OUTPUT_DIR = "/kaggle/working"

# ── Torso Height Zero-Division Guard ───────────────────────────
TORSO_HEIGHT_EPSILON = 1e-6
