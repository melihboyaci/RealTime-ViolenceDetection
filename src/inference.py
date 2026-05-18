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
)
from src.model import ViolenceGRU
from src.preprocessing import (
    preprocess_frame,
    extract_pose,
    select_top2_persons,
    build_feature_vector,
)


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


# ── Main Inference Loop ──────────────────────────────────────

def run_inference(source=0, threshold=DECISION_THRESHOLD, save_dir=None):
    """
    Run real-time inference on a video source.

    Args:
        source: 0 for webcam, or path to video file
        threshold: decision threshold (default 0.6)
        save_dir: directory to save collected samples (None = no collection)
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

        # ── Entry Suppression ─────────────────────────────────
        current_person_count = len(persons)
        if prev_person_count == 0 and current_person_count > 0:
            suppression_counter = ENTRY_SUPPRESSION_FRAMES
            buffer.clear()
            decision_history.clear()
        prev_person_count = current_person_count

        if suppression_counter > 0:
            suppression_counter -= 1

        buffer.append(fv)

        # ── Decision ──────────────────────────────────────────
        if suppression_counter > 0:
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

        # ── Visualize ─────────────────────────────────────────
        display = draw_overlay(
            frame.copy(), decision, probability, threshold,
            len(buffer), fps
        )

        cv2.imshow("Violence Detection", display)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif save_dir and len(buffer) == FIFO_BUFFER_LENGTH:
            if key == ord('v'):
                _save_sample(buffer, save_dir, "violence")
            elif key == ord('n'):
                _save_sample(buffer, save_dir, "nonviolence")

    cap.release()
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
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_inference(source=source, threshold=args.threshold, save_dir=args.collect)
