"""
Real-time inference pipeline with FIFO buffer and on-screen overlay.

Locked decisions:
    D10/D18: FIFO buffer of 30 frames, threshold 0.7
    D2:      Same preprocessing chain as offline
    D3:      YOLOv8n-Pose, 17 COCO keypoints
    D13:     69-dim feature vector
"""

import os
import sys
import math
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    BEST_MODEL_PATH,
    DECISION_THRESHOLD,
    FIFO_BUFFER_LENGTH,
    POSE_MODEL_NAME,
    NUM_KEYPOINTS,
    KEYPOINT_CONFIDENCE_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CLAHE_BRIGHTNESS_THRESHOLD,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    RESIZE_DIM,
    FEATURE_DIM,
    SKELETON_DIM,
    TORSO_HEIGHT_EPSILON,
)
from src.model import ViolenceGRU


# ── Frame Preprocessing (identical to Kaggle offline) ────────

def check_low_brightness(frame, threshold=CLAHE_BRIGHTNESS_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame):
    if check_low_brightness(frame):
        frame = apply_clahe(frame)
    frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    frame = cv2.resize(frame, RESIZE_DIM)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# ── Pose Extraction ──────────────────────────────────────────

def extract_pose(frame_rgb, model):
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


# ── Multi-Person + Normalization + Feature Vector ────────────

def select_top2_persons(persons):
    if len(persons) == 0:
        return None, None
    if len(persons) == 1:
        return persons[0], None
    sorted_by_area = sorted(persons, key=lambda p: p['bbox_area'], reverse=True)
    top2 = sorted_by_area[:2]
    top2_sorted = sorted(top2, key=lambda p: p['bbox_center'][0])
    return top2_sorted[0], top2_sorted[1]


def filter_keypoints(keypoints):
    filtered = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    for i in range(NUM_KEYPOINTS):
        if keypoints[i, 2] >= KEYPOINT_CONFIDENCE_THRESHOLD:
            filtered[i, 0] = keypoints[i, 0]
            filtered[i, 1] = keypoints[i, 1]
    return filtered


def normalize_skeleton(kps_2d):
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


def build_feature_vector(p1, p2, frame_w):
    if p1 is not None:
        sk1 = normalize_skeleton(filter_keypoints(p1['keypoints'])).flatten()
    else:
        sk1 = np.zeros(SKELETON_DIM, dtype=np.float32)
    if p2 is not None:
        sk2 = normalize_skeleton(filter_keypoints(p2['keypoints'])).flatten()
    else:
        sk2 = np.zeros(SKELETON_DIM, dtype=np.float32)
    if p1 is not None and p2 is not None:
        d = math.dist(p1['bbox_center'], p2['bbox_center'])
        nd = d / frame_w if frame_w > 0 else 1.0
    elif p1 is not None:
        nd = 1.0
    else:
        nd = 0.0
    return np.concatenate([sk1, sk2, np.array([nd], dtype=np.float32)])


# ── Visualization ────────────────────────────────────────────

def draw_overlay(frame, decision, probability, threshold, buffer_len, fps):
    h, w = frame.shape[:2]

    if decision == "Violence":
        color = (0, 0, 255)   # Red
    elif decision == "NonViolence":
        color = (0, 200, 0)   # Green
    else:
        color = (200, 200, 0) # Yellow (warming up)

    # Decision label
    cv2.putText(frame, decision, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Probability bar
    if probability is not None:
        bar_w = 200
        bar_h = 20
        bar_x = 10
        bar_y = 60
        cv2.rectangle(frame, (bar_x, bar_y),
                       (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * probability)
        cv2.rectangle(frame, (bar_x, bar_y),
                       (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"P={probability:.2f} (T={threshold})",
                     (bar_x + bar_w + 10, bar_y + 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Buffer status
    cv2.putText(frame, f"Buffer: {buffer_len}/{FIFO_BUFFER_LENGTH}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ── Main Inference Loop ──────────────────────────────────────

def run_inference(source=0, threshold=DECISION_THRESHOLD):
    """
    Run real-time inference on a video source.

    Args:
        source: 0 for webcam, or path to video file
        threshold: decision threshold (default 0.7)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GRU model
    model = ViolenceGRU().to(device)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"GRU model loaded: {BEST_MODEL_PATH}")

    # Load YOLO pose model
    pose_model = YOLO(POSE_MODEL_NAME)
    print(f"Pose model loaded: {POSE_MODEL_NAME}")

    # FIFO buffer
    buffer = deque(maxlen=FIFO_BUFFER_LENGTH)

    # Video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    print(f"\nInference started (source={source}, threshold={threshold})")
    print("Press 'q' to quit.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # ── Per-frame pipeline ────────────────────────────────
        preprocessed = preprocess_frame(frame)
        persons = extract_pose(preprocessed, pose_model)
        p1, p2 = select_top2_persons(persons)
        fv = build_feature_vector(p1, p2, RESIZE_DIM[0])

        buffer.append(fv)

        # ── Decision ──────────────────────────────────────────
        if len(buffer) < FIFO_BUFFER_LENGTH:
            decision = "Warming Up"
            probability = None
        else:
            seq = np.array(list(buffer), dtype=np.float32)  # (30, 69)
            tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # (1, 30, 69)

            with torch.no_grad():
                prob = model(tensor).item()

            probability = prob
            decision = "Violence" if prob >= threshold else "NonViolence"

        # ── Visualize ─────────────────────────────────────────
        display = draw_overlay(
            frame.copy(), decision, probability, threshold,
            len(buffer), fps
        )

        cv2.imshow("Violence Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Violence Detection")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD,
                        help=f"Decision threshold (default: {DECISION_THRESHOLD})")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_inference(source=source, threshold=args.threshold)
