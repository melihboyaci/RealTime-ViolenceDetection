"""
Shared preprocessing functions used by both Kaggle Notebook (offline)
and real-time inference (online).

Pipeline order (decision_log.md D2):
    Conditional CLAHE → GaussianBlur 3×3 → Resize 640×640 → BGR→RGB
    → YOLOv8n-Pose → Multi-Person B+ → Hip Centering → Shoulder-Hip Scaling
    → 69-dim Feature Vector
"""

import os
import math
import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────
TARGET_FPS = 10
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_BRIGHTNESS_THRESHOLD = 50
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 0
RESIZE_DIM = (640, 640)
NUM_KEYPOINTS = 17
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
FEATURE_DIM = 69
SKELETON_DIM = 34
TORSO_HEIGHT_EPSILON = 1e-6
SEQUENCE_LENGTH = 30
SLIDING_WINDOW_STRIDE = 15
MOTION_FILTER_THETA = 0.05


# ── Frame Preprocessing (D2) ─────────────────────────────────

def check_low_brightness(frame, threshold=CLAHE_BRIGHTNESS_THRESHOLD):
    """Frame'in ortalama parlaklığı threshold altındaysa True döner."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold


def apply_clahe(frame):
    """LAB renk uzayında L kanalına CLAHE uygular."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame):
    """
    Locked preprocessing chain:
    CLAHE (conditional) → GaussianBlur 3×3 → Resize 640×640 → BGR→RGB
    """
    if check_low_brightness(frame):
        frame = apply_clahe(frame)
    frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    frame = cv2.resize(frame, RESIZE_DIM)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# ── Pose Extraction (D3) ─────────────────────────────────────

def extract_pose(frame_rgb, model):
    """
    YOLOv8n-Pose ile frame'den keypoint çıkarır.

    Returns:
        List[dict]: her kişi için {keypoints(17,3), bbox, bbox_area, bbox_center, detection_conf}
    """
    results = model(frame_rgb, verbose=False)
    persons = []

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kps_data = results[0].keypoints.data.cpu().numpy()
        boxes_data = results[0].boxes.data.cpu().numpy()

        for i in range(len(kps_data)):
            kps = kps_data[i]
            box = boxes_data[i]
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            persons.append({
                'keypoints': kps,
                'bbox': box[:4],
                'bbox_area': bbox_area,
                'bbox_center': (cx, cy),
                'detection_conf': box[4],
            })

    return persons


# ── Multi-Person Selection (D4/D5) ───────────────────────────

def select_top2_persons(persons):
    """Top-2 kişi seçimi (bbox alanına göre) + X-axis sort."""
    if len(persons) == 0:
        return None, None
    if len(persons) == 1:
        return persons[0], None
    sorted_by_area = sorted(persons, key=lambda p: p['bbox_area'], reverse=True)
    top2 = sorted_by_area[:2]
    top2_sorted = sorted(top2, key=lambda p: p['bbox_center'][0])
    return top2_sorted[0], top2_sorted[1]


# ── Keypoint Filtering (D12) ─────────────────────────────────

def filter_keypoints(keypoints):
    """Confidence < 0.5 olan keypoint'leri sıfırlar."""
    filtered = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    for i in range(NUM_KEYPOINTS):
        if keypoints[i, 2] >= KEYPOINT_CONFIDENCE_THRESHOLD:
            filtered[i, 0] = keypoints[i, 0]
            filtered[i, 1] = keypoints[i, 1]
    return filtered


# ── Skeleton Normalization (D14) ──────────────────────────────

def normalize_skeleton(kps_2d):
    """Hip centering + shoulder-hip scaling."""
    hip_mx = (kps_2d[11, 0] + kps_2d[12, 0]) / 2
    hip_my = (kps_2d[11, 1] + kps_2d[12, 1]) / 2

    for i in range(NUM_KEYPOINTS):
        if kps_2d[i, 0] == 0.0 and kps_2d[i, 1] == 0.0:
            kps_2d[i, 0] = hip_mx
            kps_2d[i, 1] = hip_my

    centered = kps_2d.copy()
    centered[:, 0] -= hip_mx
    centered[:, 1] -= hip_my

    sh_my = (kps_2d[5, 1] + kps_2d[6, 1]) / 2
    torso_h = abs(hip_my - sh_my)

    if torso_h > TORSO_HEIGHT_EPSILON:
        centered /= torso_h
    else:
        return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    return centered


# ── Feature Vector Builder (D13) ─────────────────────────────

def build_feature_vector(person1, person2, frame_width):
    """69-dim vektör: [skeleton1(34)] + [skeleton2(34)] + [norm_distance(1)]."""
    if person1 is not None:
        sk1 = normalize_skeleton(filter_keypoints(person1['keypoints'])).flatten()
    else:
        sk1 = np.zeros(SKELETON_DIM, dtype=np.float32)

    if person2 is not None:
        sk2 = normalize_skeleton(filter_keypoints(person2['keypoints'])).flatten()
    else:
        sk2 = np.zeros(SKELETON_DIM, dtype=np.float32)

    if person1 is not None and person2 is not None:
        d = math.dist(person1['bbox_center'], person2['bbox_center'])
        nd = d / frame_width if frame_width > 0 else 1.0
    elif person1 is not None:
        nd = 1.0
    else:
        nd = 0.0

    return np.concatenate([sk1, sk2, np.array([nd], dtype=np.float32)])


# ── Video Processing ──────────────────────────────────────────

def process_single_video(video_path, model):
    """
    Tek videoyu uçtan uca işler: FPS sampling → preprocess → pose → feature.

    Returns:
        (np.array shape (n_frames, 69), None) on success
        (None, error_string) on failure
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Video açılamadı"

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        return None, f"Geçersiz FPS={native_fps} veya frame_count={total_frames}"

    frame_interval = max(1, round(native_fps / TARGET_FPS))
    feature_vectors = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            preprocessed = preprocess_frame(frame)
            persons = extract_pose(preprocessed, model)
            p1, p2 = select_top2_persons(persons)
            fv = build_feature_vector(p1, p2, RESIZE_DIM[0])
            feature_vectors.append(fv)

        frame_idx += 1

    cap.release()

    if len(feature_vectors) == 0:
        return None, "Hiç frame işlenemedi"

    return np.array(feature_vectors, dtype=np.float32), None


# ── Sequence Generation ───────────────────────────────────────

def create_sliding_windows(features, window_size=SEQUENCE_LENGTH, stride=SLIDING_WINDOW_STRIDE):
    """(n_frames, 69) → list of (30, 69) windows."""
    n_frames = features.shape[0]
    windows = []

    if n_frames < window_size:
        padded = np.zeros((window_size, FEATURE_DIM), dtype=np.float32)
        padded[:n_frames] = features
        windows.append(padded)
    else:
        for start in range(0, n_frames - window_size + 1, stride):
            windows.append(features[start:start + window_size])

    return windows


def compute_motion_score(window):
    """Penceredeki ardışık frame çiftleri arası ortalama L2 mesafesi."""
    diffs = np.diff(window, axis=0)
    return np.linalg.norm(diffs, axis=1).mean()


def apply_motion_filter(windows, theta=MOTION_FILTER_THETA):
    """Düşük hareketli Violence pencerelerini eler (D11)."""
    return [w for w in windows if compute_motion_score(w) >= theta]
