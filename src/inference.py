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
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    BEST_MODEL_PATH,
    DECISION_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
    FIFO_BUFFER_LENGTH,
    POSE_MODEL_NAME,
    RESIZE_DIM,
    SMOOTHING_WINDOW,
    SMOOTHING_MIN_COUNT,
    ENTRY_SUPPRESSION_FRAMES,
    KEYPOINT_CONFIDENCE_THRESHOLD,
    MIN_VALID_KEYPOINTS_FOR_INFERENCE,
    MIN_TORSO_KEYPOINTS_FOR_INFERENCE,
    MIN_PERSON_BBOX_AREA_RATIO,
)
from src.model import ViolenceGRU
from src.preprocessing import (
    preprocess_frame,
    extract_pose,
    select_top2_persons,
    build_feature_vector,
)
from src.overlay_panel import build_info_panel, compose_display


# ── Pose Visualization ─────────────────────────────────────

# COCO 17 keypoint skeleton connections (parent -> child)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),       # Nose to eyes
    (1, 3), (2, 4),       # Eyes to ears
    (0, 5), (0, 6),       # Nose to shoulders
    (5, 7), (7, 9),       # Left arm
    (6, 8), (8, 10),      # Right arm
    (5, 6),               # Shoulders
    (5, 11), (6, 12),     # Shoulders to hips
    (11, 12),             # Hips
    (11, 13), (13, 15),   # Left leg
    (12, 14), (14, 16),   # Right leg
]

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

TORSO_KEYPOINT_INDICES = (5, 6, 11, 12)


def is_valid_person_pose(person, frame_shape):
    keypoints = person["keypoints"]
    valid_keypoint_count = int(
        np.sum(keypoints[:, 2] >= KEYPOINT_CONFIDENCE_THRESHOLD)
    )
    torso_keypoint_count = int(
        np.sum(
            keypoints[list(TORSO_KEYPOINT_INDICES), 2]
            >= KEYPOINT_CONFIDENCE_THRESHOLD
        )
    )

    frame_area = frame_shape[0] * frame_shape[1]
    bbox_area_ratio = person["bbox_area"] / frame_area if frame_area > 0 else 0.0

    return (
        valid_keypoint_count >= MIN_VALID_KEYPOINTS_FOR_INFERENCE
        and torso_keypoint_count >= MIN_TORSO_KEYPOINTS_FOR_INFERENCE
        and bbox_area_ratio >= MIN_PERSON_BBOX_AREA_RATIO
    )


def filter_valid_persons(persons, frame_shape):
    return [person for person in persons if is_valid_person_pose(person, frame_shape)]


def draw_skeleton_on_frame(frame, persons, decision):
    """Draw pose keypoints, skeleton connections, and bounding boxes on frame."""
    if not persons:
        return frame

    # Color based on decision
    if decision == "Violence":
        pose_color = (0, 0, 255)      # Red
        skeleton_color = (0, 100, 255) # Light red
    elif decision == "Suspicious":
        pose_color = (0, 200, 255)    # Orange
        skeleton_color = (0, 150, 255) # Light orange
    else:
        pose_color = (0, 255, 0)      # Green
        skeleton_color = (0, 200, 100) # Light green

    # Draw for each detected person
    for person_idx, person in enumerate(persons[:2]):  # Top 2 persons
        kps = person['keypoints']  # (17, 3) array [x, y, conf]
        bbox = person['bbox']      # [x1, y1, x2, y2, conf]

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), pose_color, 2)
        label = f"P{person_idx + 1}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)

        # Draw skeleton connections first (lines behind circles)
        valid_kps = {}  # Store valid keypoints for drawing
        for parent_idx, child_idx in SKELETON_CONNECTIONS:
            parent_kp = kps[parent_idx]
            child_kp = kps[child_idx]

            # Both keypoints must be valid (confidence > 0.5)
            if parent_kp[2] > 0.5 and child_kp[2] > 0.5:
                pt1 = (int(parent_kp[0]), int(parent_kp[1]))
                pt2 = (int(child_kp[0]), int(child_kp[1]))
                cv2.line(frame, pt1, pt2, skeleton_color, 2)
                valid_kps[parent_idx] = pt1
                valid_kps[child_idx] = pt2

        # Draw keypoint circles
        for i, kp in enumerate(kps):
            if kp[2] > 0.5:  # Confidence threshold
                x, y = int(kp[0]), int(kp[1])
                # Different size for nose (index 0)
                radius = 5 if i == 0 else 3
                cv2.circle(frame, (x, y), radius, pose_color, -1)

    return frame


# ── Visualization ────────────────────────────────────────────

def draw_overlay(frame, decision, probability, threshold, buffer_len, fps):
    h, w = frame.shape[:2]

    if decision == "Violence":
        color = (0, 0, 255)   # Red
    elif decision == "Suspicious":
        color = (0, 200, 255) # Yellow/Orange
    elif decision == "NonViolence":
        color = (0, 200, 0)   # Green
    else:
        color = (200, 200, 0) # Cyan (warming up)

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


def draw_pose_quality(frame, valid_count, total_count):
    h = frame.shape[0]
    cv2.putText(
        frame,
        f"Valid poses: {valid_count}/{total_count}",
        (10, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    return frame


# ── Main Inference Loop ──────────────────────────────────────

def run_inference(source=0, threshold=DECISION_THRESHOLD, save_dir=None, display=True):
    """
    Run real-time inference on a video source.

    Args:
        source: 0 for webcam, or path to video file
        threshold: decision threshold (default 0.6)
        save_dir: directory to save collected samples (None = no collection)
        display: show OpenCV preview window if True
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

    # Temporal smoothing history
    decision_history = deque(maxlen=SMOOTHING_WINDOW)

    # Entry suppression state
    prev_person_count = 0
    suppression_counter = 0

    # Panel decision history log (last 10 with timestamps)
    decision_history_log = deque(maxlen=10)

    # Sample collection setup
    if save_dir:
        os.makedirs(os.path.join(save_dir, "violence"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "nonviolence"), exist_ok=True)
        print(f"Sample collection enabled: {save_dir}")
        print("  Press 'v' to save current buffer as Violence")
        print("  Press 'n' to save current buffer as NonViolence")

    # Video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    print(f"\nInference started (source={source}, threshold={threshold})")
    print(f"Smoothing: {SMOOTHING_MIN_COUNT}/{SMOOTHING_WINDOW} windows required")
    print(f"Entry suppression: {ENTRY_SUPPRESSION_FRAMES} frames")
    if display:
        print("Press 'q' to quit.\n")
    else:
        print("Display disabled. Press Ctrl+C to stop.\n")

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
        valid_persons = filter_valid_persons(persons, preprocessed.shape)
        p1, p2 = select_top2_persons(valid_persons)
        has_valid_pose = p1 is not None

        # ── Entry Suppression ─────────────────────────────────
        current_person_count = len(valid_persons)
        if prev_person_count == 0 and current_person_count > 0:
            suppression_counter = ENTRY_SUPPRESSION_FRAMES
            buffer.clear()
            decision_history.clear()
        prev_person_count = current_person_count

        if suppression_counter > 0:
            suppression_counter -= 1

        if has_valid_pose:
            fv = build_feature_vector(p1, p2, RESIZE_DIM[0])
            buffer.append(fv)
        else:
            buffer.clear()
            decision_history.clear()

        # ── Decision ──────────────────────────────────────────
        if not has_valid_pose:
            decision = "No Valid Pose"
            probability = None
        elif suppression_counter > 0:
            decision = "Warming Up"
            probability = None
        elif len(buffer) < FIFO_BUFFER_LENGTH:
            decision = "Warming Up"
            probability = None
        else:
            seq = np.array(list(buffer), dtype=np.float32)  # (30, 69)
            tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # (1, 30, 69)

            with torch.no_grad():
                prob = model(tensor).item()

            probability = prob

            # Raw decision from model
            if prob >= threshold:
                raw_decision = "Violence"
            elif prob >= SUSPICIOUS_THRESHOLD:
                raw_decision = "Suspicious"
            else:
                raw_decision = "NonViolence"

            # Temporal smoothing
            decision_history.append(raw_decision)
            violence_count = sum(
                1 for d in decision_history if d in ("Violence", "Suspicious")
            )

            if violence_count >= SMOOTHING_MIN_COUNT:
                # Confirm: check if any were full Violence
                full_violence = sum(1 for d in decision_history if d == "Violence")
                if full_violence >= SMOOTHING_MIN_COUNT:
                    decision = "Violence"
                else:
                    decision = "Suspicious"
            else:
                decision = "NonViolence"

        # ── Log decision for panel history (only on change) ───
        if not decision_history_log or decision_history_log[-1][1] != decision:
            decision_history_log.append((datetime.now(), decision))

        if display:
            # ── Visualize ─────────────────────────────────────
            # 1. Draw pose skeleton on original frame
            frame_with_skeleton = draw_skeleton_on_frame(
                frame.copy(), valid_persons, decision
            )

            # 2. Build info panel (pass up to 2 valid persons)
            info_panel = build_info_panel(
                frame_height=frame_with_skeleton.shape[0],
                decision=decision,
                probability=probability,
                threshold=threshold,
                buffer_len=len(buffer),
                buffer_max=FIFO_BUFFER_LENGTH,
                valid_poses=len(valid_persons),
                total_poses=len(persons),
                fps=fps,
                persons=valid_persons[:2],
                decision_history=decision_history_log,
            )

            # 3. Compose frame + panel side by side
            display_frame = compose_display(frame_with_skeleton, info_panel)

            try:
                cv2.imshow("Violence Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as error:
                print(f"OpenCV display unavailable: {error}")
                print("Continuing in no-display mode.")
                display = False
                key = 255

            # ── Key handling ──────────────────────────────────
            if key == ord('q'):
                break
            elif save_dir and len(buffer) == FIFO_BUFFER_LENGTH:
                if key == ord('v'):
                    _save_sample(buffer, save_dir, "violence")
                elif key == ord('n'):
                    _save_sample(buffer, save_dir, "nonviolence")
        elif probability is not None:
            print(
                f"decision={decision:12s} prob={probability:.3f} "
                f"buffer={len(buffer)}/{FIFO_BUFFER_LENGTH} "
                f"valid_poses={len(valid_persons)}/{len(persons)} fps={fps:.1f}"
            )
        elif not display:
            print(
                f"decision={decision:12s} prob=None "
                f"buffer={len(buffer)}/{FIFO_BUFFER_LENGTH} "
                f"valid_poses={len(valid_persons)}/{len(persons)} fps={fps:.1f}"
            )

    cap.release()
    if display:
        cv2.destroyAllWindows()
    print("Inference stopped.")


def _save_sample(buffer, save_dir, label):
    """Save current buffer as a labeled .npy sample for fine-tuning."""
    seq = np.array(list(buffer), dtype=np.float32)
    label_dir = os.path.join(save_dir, label)
    existing = len([f for f in os.listdir(label_dir) if f.endswith('.npy')])
    filename = f"sample_{existing:04d}.npy"
    filepath = os.path.join(label_dir, filename)
    np.save(filepath, seq)
    print(f"  [SAVED] {label}/{filename} shape={seq.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Violence Detection")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD,
                        help=f"Decision threshold (default: {DECISION_THRESHOLD})")
    parser.add_argument("--collect", type=str, default=None,
                        help="Directory to save collected samples for fine-tuning")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without OpenCV preview window")
    args = parser.parse_args()

    if isinstance(args.source, str) and args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    run_inference(
        source=source,
        threshold=args.threshold,
        save_dir=args.collect,
        display=not args.no_display,
    )
