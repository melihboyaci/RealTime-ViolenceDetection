"""
Fixed-slot OpenCV info panel for real-time violence detection.

Renders a dark sidebar panel (380px wide) alongside the camera frame.
Every section occupies an absolute, pre-calculated pixel slot so the
layout NEVER shifts regardless of content changes.

Slot map (480px total height):
    y=  0  DECISION         h=58
    y= 58  PROBABILITY      h=64
    y=122  STATUS           h=66
    y=188  POSE QUALITY     h=140  (two persons side-by-side)
    y=328  HISTORY          h=90   (last 5 decisions, fixed rows)
    y=418  LEGEND + MODEL   h=62   (single compact footer)
"""

from datetime import datetime

import cv2
import numpy as np

# ── Color Palette (BGR) ──────────────────────────────────────

COLORS = {
    "Violence":     (0,   0,   255),
    "Suspicious":   (0,   200, 255),
    "NonViolence":  (0,   200, 0),
    "No Valid Pose":(200, 200, 0),
    "Warming Up":   (200, 200, 0),
}

PANEL_BG            = (22,  22,  22)
SECTION_BG          = (35,  35,  35)
SECTION_TITLE_COLOR = (160, 160, 160)
TEXT_COLOR          = (220, 220, 220)
DIM_TEXT_COLOR      = (110, 110, 110)
BAR_BG_COLOR        = (70,  70,  70)
DIVIDER_COLOR       = (55,  55,  55)
THRESHOLD_COLOR     = (255, 255, 255)
VALID_KP_COLOR      = (0,   210, 0)
INVALID_KP_COLOR    = (70,  70,  70)

# ── Fixed slot boundaries ─────────────────────────────────────
# All y-values are absolute pixel positions in the panel.

SLOT_DECISION   = (0,   58)
SLOT_PROB       = (58,  122)
SLOT_STATUS     = (122, 188)
SLOT_POSE       = (188, 328)
SLOT_HISTORY    = (328, 418)
SLOT_FOOTER     = (418, 480)

PANEL_WIDTH     = 380
PANEL_HEIGHT    = 480   # matches standard 480p camera height

HISTORY_ROWS    = 5     # fixed number of history rows rendered

# ── Keypoint diagram positions (relative, 70×120 bounding box) ─

DIAGRAM_POSITIONS = {
    0:  (35, 8),    # nose
    1:  (30, 6),    # left_eye
    2:  (40, 6),    # right_eye
    3:  (26, 10),   # left_ear
    4:  (44, 10),   # right_ear
    5:  (24, 30),   # left_shoulder
    6:  (46, 30),   # right_shoulder
    7:  (16, 48),   # left_elbow
    8:  (54, 48),   # right_elbow
    9:  (10, 66),   # left_wrist
    10: (60, 66),   # right_wrist
    11: (28, 66),   # left_hip
    12: (42, 66),   # right_hip
    13: (24, 88),   # left_knee
    14: (46, 88),   # right_knee
    15: (21, 110),  # left_ankle
    16: (49, 110),  # right_ankle
}

DIAGRAM_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KEYPOINT_GROUPS = {
    "face":  [0, 1, 2, 3, 4],
    "torso": [5, 6, 11, 12],
    "arms":  [7, 8, 9, 10],
    "legs":  [13, 14, 15, 16],
}


# ── Helpers ──────────────────────────────────────────────────

def _fill_slot(panel, slot_y, slot_h, bg=SECTION_BG):
    """Fill a slot with its background color."""
    cv2.rectangle(panel, (0, slot_y), (panel.shape[1], slot_y + slot_h),
                  bg, -1)


def _divider(panel, slot_y):
    """Draw a 1-pixel horizontal divider at the top of a slot."""
    cv2.line(panel, (0, slot_y), (panel.shape[1], slot_y), DIVIDER_COLOR, 1)


def _title(panel, text, slot_y, margin_top=10):
    """Draw a small section title inside a slot."""
    cv2.putText(panel, text,
                (14, slot_y + margin_top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, SECTION_TITLE_COLOR, 1,
                cv2.LINE_AA)


# ── Slot drawing functions ────────────────────────────────────

def _draw_decision(panel, decision):
    """Slot 0: large coloured decision label."""
    sy, sh = SLOT_DECISION
    _fill_slot(panel, sy, sh, SECTION_BG)
    color = COLORS.get(decision, (180, 180, 180))
    # Coloured left accent bar
    cv2.rectangle(panel, (0, sy), (5, sy + sh), color, -1)
    cv2.putText(panel, decision,
                (18, sy + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)


def _draw_probability(panel, probability, threshold):
    """Slot 1: probability bar with threshold marker and text inside bar."""
    sy, _ = SLOT_PROB
    _fill_slot(panel, sy, SLOT_PROB[1] - sy, PANEL_BG)
    _divider(panel, sy)
    _title(panel, "PROBABILITY", sy)

    bar_x = 14
    bar_y = sy + 20
    bar_w = PANEL_WIDTH - 28
    bar_h = 20

    # Background
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), BAR_BG_COLOR, -1)

    if probability is not None:
        probability = max(0.0, min(1.0, probability))
        fill_w = int(bar_w * probability)
        if probability >= threshold:
            bar_color = COLORS["Violence"]
        elif probability >= 0.35:
            bar_color = COLORS["Suspicious"]
        else:
            bar_color = COLORS["NonViolence"]
        cv2.rectangle(panel, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        # Text inside bar (always visible)
        prob_text = f"{probability:.3f}"
        cv2.putText(panel, prob_text,
                    (bar_x + 6, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Threshold marker
    thresh_x = bar_x + int(bar_w * threshold)
    cv2.line(panel, (thresh_x, bar_y - 4),
             (thresh_x, bar_y + bar_h + 4), THRESHOLD_COLOR, 2)

    # Threshold label
    cv2.putText(panel, f"Threshold: {threshold:.2f}",
                (bar_x, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM_TEXT_COLOR, 1,
                cv2.LINE_AA)


def _draw_status(panel, buffer_len, buffer_max, valid_poses,
                 total_poses, fps):
    """Slot 2: buffer, valid poses, FPS."""
    sy, sh = SLOT_STATUS
    _fill_slot(panel, sy, sh, SECTION_BG)
    _divider(panel, sy)
    _title(panel, "STATUS", sy)

    items = [
        (f"Buffer  {buffer_len}/{buffer_max}", 20),
        (f"Poses   {valid_poses}/{total_poses}", 38),
        (f"FPS     {fps:.1f}", 56),
    ]
    for text, dy in items:
        cv2.putText(panel, text,
                    (14, sy + dy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1,
                    cv2.LINE_AA)


def _draw_single_figure(panel, person, ox, oy):
    """Draw one stick-figure + group summary at (ox, oy) offset."""
    if person is None:
        cv2.putText(panel, "No person",
                    (ox + 8, oy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM_TEXT_COLOR, 1,
                    cv2.LINE_AA)
        return

    keypoints = person["keypoints"]

    # Skeleton connections
    for pi, ci in DIAGRAM_CONNECTIONS:
        px, py = DIAGRAM_POSITIONS[pi][0] + ox, DIAGRAM_POSITIONS[pi][1] + oy
        cx, cy = DIAGRAM_POSITIONS[ci][0] + ox, DIAGRAM_POSITIONS[ci][1] + oy
        valid = (keypoints[pi, 2] >= 0.5 and keypoints[ci, 2] >= 0.5)
        cv2.line(panel, (px, py), (cx, cy),
                 (170, 170, 170) if valid else (50, 50, 50), 1)

    # Keypoint circles
    for idx, (dx, dy) in DIAGRAM_POSITIONS.items():
        pt = (dx + ox, dy + oy)
        is_valid = keypoints[idx, 2] >= 0.5
        if is_valid:
            cv2.circle(panel, pt, 3, VALID_KP_COLOR, -1)
        else:
            cv2.circle(panel, pt, 3, INVALID_KP_COLOR, 1)

    # Group summary to the RIGHT of the figure (avoids bleeding below slot)
    summary_x = ox + 76
    summary_y = oy + 28
    for group, indices in KEYPOINT_GROUPS.items():
        cnt = sum(1 for i in indices if keypoints[i, 2] >= 0.5)
        total = len(indices)
        if cnt == total:
            c = VALID_KP_COLOR
        elif cnt > 0:
            c = (0, 180, 255)
        else:
            c = INVALID_KP_COLOR
        cv2.putText(panel, f"{group[:4]}:{cnt}/{total}",
                    (summary_x, summary_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, c, 1, cv2.LINE_AA)
        summary_y += 16


def _draw_pose_quality(panel, persons):
    """Slot 3: two stick-figures side-by-side."""
    sy, sh = SLOT_POSE
    _fill_slot(panel, sy, sh, PANEL_BG)
    _divider(panel, sy)
    _title(panel, "POSE QUALITY", sy)

    # Person labels
    for col, label in enumerate(["P1", "P2"]):
        lx = 14 + col * 183
        cv2.putText(panel, label,
                    (lx, sy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1,
                    cv2.LINE_AA)

    # Vertical divider between the two figures
    mid_x = PANEL_WIDTH // 2
    cv2.line(panel, (mid_x, sy + 16), (mid_x, sy + sh - 4),
             DIVIDER_COLOR, 1)

    # Draw figures
    p1 = persons[0] if len(persons) > 0 else None
    p2 = persons[1] if len(persons) > 1 else None
    _draw_single_figure(panel, p1, ox=18,          oy=sy + 30)
    _draw_single_figure(panel, p2, ox=mid_x + 8,   oy=sy + 30)


def _draw_history(panel, history):
    """Slot 4: last HISTORY_ROWS decisions, always HISTORY_ROWS rows."""
    sy, sh = SLOT_HISTORY
    _fill_slot(panel, sy, sh, SECTION_BG)
    _divider(panel, sy)
    _title(panel, "HISTORY", sy)

    row_h = (sh - 18) // HISTORY_ROWS
    entries = list(history)[-HISTORY_ROWS:]
    # Pad to always fill HISTORY_ROWS rows
    while len(entries) < HISTORY_ROWS:
        entries.insert(0, None)

    for row_idx, entry in enumerate(entries):
        ry = sy + 18 + row_idx * row_h
        if entry is None:
            continue  # leave empty rows blank — cleaner than near-invisible dashes
        timestamp, decision = entry
        color = COLORS.get(decision, (180, 180, 180))
        time_str = timestamp.strftime("%H:%M:%S")
        cv2.putText(panel, time_str,
                    (14, ry + row_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM_TEXT_COLOR, 1,
                    cv2.LINE_AA)
        cv2.putText(panel, decision,
                    (90, ry + row_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    cv2.LINE_AA)


def _draw_footer(panel):
    """Slot 5: compact legend + model info on two lines."""
    sy, sh = SLOT_FOOTER
    _fill_slot(panel, sy, sh, PANEL_BG)
    _divider(panel, sy)

    # Legend dots inline
    legend = [
        ("NV", COLORS["NonViolence"]),
        ("SU", COLORS["Suspicious"]),
        ("VI", COLORS["Violence"]),
        ("NP", COLORS["No Valid Pose"]),
    ]
    lx = 14
    for abbr, color in legend:
        cv2.circle(panel, (lx + 5, sy + 16), 5, color, -1)
        cv2.putText(panel, abbr, (lx + 14, sy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1,
                    cv2.LINE_AA)
        lx += 60

    cv2.putText(panel, "YOLOv8n-Pose | GRU 128->64 | 30x69",
                (14, sy + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, DIM_TEXT_COLOR, 1,
                cv2.LINE_AA)
    cv2.putText(panel, "Q:Quit  V:Save-Violence  N:Save-NonViolence",
                (14, sy + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, DIM_TEXT_COLOR, 1,
                cv2.LINE_AA)


# ── Public API ────────────────────────────────────────────────

def build_info_panel(frame_height, decision, probability, threshold,
                     buffer_len, buffer_max, valid_poses, total_poses,
                     fps, persons, decision_history, panel_width=PANEL_WIDTH):
    """
    Build the complete fixed-slot info panel.

    Args:
        frame_height: height of the camera frame (panel is resized to match)
        decision: current decision string
        probability: float or None
        threshold: float decision boundary
        buffer_len: current FIFO fill count
        buffer_max: FIFO capacity (30)
        valid_poses: count of valid persons this frame
        total_poses: count of all detected persons this frame
        fps: current frames-per-second
        persons: list of up to 2 person dicts (keypoints, bbox, bbox_area…)
        decision_history: deque of (datetime, decision_str) tuples
        panel_width: panel width in pixels (default 380)

    Returns:
        numpy array (PANEL_HEIGHT × panel_width × 3)
    """
    panel = np.full((PANEL_HEIGHT, panel_width, 3), PANEL_BG, dtype=np.uint8)

    _draw_decision(panel, decision)
    _draw_probability(panel, probability, threshold)
    _draw_status(panel, buffer_len, buffer_max, valid_poses, total_poses, fps)
    _draw_pose_quality(panel, list(persons) if persons else [])
    _draw_history(panel, decision_history)
    _draw_footer(panel)

    return panel


def compose_display(frame, panel):
    """Horizontally stack camera frame and info panel, matching heights."""
    frame_h = frame.shape[0]
    panel_h = panel.shape[0]
    if frame_h != panel_h:
        panel = cv2.resize(panel, (panel.shape[1], frame_h),
                           interpolation=cv2.INTER_NEAREST)
    return np.hstack([frame, panel])
