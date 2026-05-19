"""
Generate ablation notebooks from the baseline preprocessing notebook.

P7.3 — Normalization variants:
  1. norm_none: no normalization (raw keypoints after filtering)
  2. norm_hip_only: hip centering only (no shoulder-hip scaling)
  3. norm_scale_only: shoulder-hip scaling only (no hip centering)

P7.4 — Motion-filter θ variants:
  4. motion_0.00: θ = 0.00 (disabled)
  5. motion_0.025: θ = 0.025
  6. motion_0.075: θ = 0.075
  7. motion_0.10: θ = 0.10
"""

import json
import copy
import os

BASE_NOTEBOOK = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'kaggle_preprocessing.ipynb')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

# ── Normalization variants (replace normalize_skeleton in MODULE_CODE) ──

NORM_NONE = '''
def normalize_skeleton(kps_2d):
    """P7.3 Ablation: NO normalization — return filtered keypoints as-is."""
    return kps_2d
'''

NORM_HIP_ONLY = '''
def normalize_skeleton(kps_2d):
    """P7.3 Ablation: Hip centering ONLY (no shoulder-hip scaling)."""
    hip_mx = (kps_2d[11, 0] + kps_2d[12, 0]) / 2
    hip_my = (kps_2d[11, 1] + kps_2d[12, 1]) / 2

    for i in range(NUM_KEYPOINTS):
        if kps_2d[i, 0] == 0.0 and kps_2d[i, 1] == 0.0:
            kps_2d[i, 0] = hip_mx
            kps_2d[i, 1] = hip_my

    centered = kps_2d.copy()
    centered[:, 0] -= hip_mx
    centered[:, 1] -= hip_my
    return centered
'''

NORM_SCALE_ONLY = '''
def normalize_skeleton(kps_2d):
    """P7.3 Ablation: Shoulder-hip scaling ONLY (no hip centering)."""
    hip_mx = (kps_2d[11, 0] + kps_2d[12, 0]) / 2
    hip_my = (kps_2d[11, 1] + kps_2d[12, 1]) / 2

    for i in range(NUM_KEYPOINTS):
        if kps_2d[i, 0] == 0.0 and kps_2d[i, 1] == 0.0:
            kps_2d[i, 0] = hip_mx
            kps_2d[i, 1] = hip_my

    sh_my = (kps_2d[5, 1] + kps_2d[6, 1]) / 2
    torso_h = abs(hip_my - sh_my)

    if torso_h > TORSO_HEIGHT_EPSILON:
        kps_2d /= torso_h
    else:
        return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    return kps_2d
'''

BASELINE_NORM = '''def normalize_skeleton(kps_2d):
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

    return centered'''


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {os.path.basename(path)}")


def get_cell_source(cell):
    """Get cell source as a single string."""
    if isinstance(cell['source'], list):
        return ''.join(cell['source'])
    return cell['source']


def set_cell_source(cell, text):
    """Set cell source as a list of lines."""
    cell['source'] = text.split('\n')
    # Rejoin as list with newlines (except last line)
    lines = text.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]


def make_norm_variant(nb, variant_name, norm_func_code, description):
    """Replace normalize_skeleton in Cell 1 (MODULE_CODE) and update markdown."""
    nb = copy.deepcopy(nb)

    # Update markdown title
    md_src = get_cell_source(nb['cells'][0])
    md_src = md_src.replace(
        "**Amac:** Ham videoları uçtan uca işleyip eğitime hazır `(30, 69)` sekans dosyalarına dönüştürmek.",
        f"**P7.3 Ablation: {description}**\\n\\nBu notebook, normalizasyon ablasyonu için değiştirilmiş preprocessing pipeline çalıştırır."
    )
    set_cell_source(nb['cells'][0], md_src)

    # Replace normalize_skeleton in Cell 1 MODULE_CODE
    cell1_src = get_cell_source(nb['cells'][1])
    cell1_src = cell1_src.replace(BASELINE_NORM, norm_func_code.strip())
    set_cell_source(nb['cells'][1], cell1_src)

    return nb


def make_motion_variant(nb, theta_value):
    """Replace MOTION_FILTER_THETA in Cell 1 (MODULE_CODE) and update markdown."""
    nb = copy.deepcopy(nb)

    # Update markdown title
    md_src = get_cell_source(nb['cells'][0])
    md_src = md_src.replace(
        "**Amac:** Ham videoları uçtan uca işleyip eğitime hazır `(30, 69)` sekans dosyalarına dönüştürmek.",
        f"**P7.4 Ablation: Motion Filter θ = {theta_value}**\\n\\nBu notebook, motion filter eşiği ablasyonu için değiştirilmiş preprocessing pipeline çalıştırır."
    )
    set_cell_source(nb['cells'][0], md_src)

    # Replace MOTION_FILTER_THETA in Cell 1 MODULE_CODE
    cell1_src = get_cell_source(nb['cells'][1])
    cell1_src = cell1_src.replace(
        "MOTION_FILTER_THETA = 0.05",
        f"MOTION_FILTER_THETA = {theta_value}"
    )
    set_cell_source(nb['cells'][1], cell1_src)

    return nb


def main():
    print("Loading baseline notebook...")
    nb = load_notebook(BASE_NOTEBOOK)

    print("\nGenerating P7.3 Normalization ablation notebooks:")
    variants = [
        ("norm_none", NORM_NONE, "No Normalization (raw keypoints)"),
        ("norm_hip_only", NORM_HIP_ONLY, "Hip Centering Only (no scaling)"),
        ("norm_scale_only", NORM_SCALE_ONLY, "Shoulder-Hip Scaling Only (no centering)"),
    ]
    for name, func, desc in variants:
        out_nb = make_norm_variant(nb, name, func, desc)
        out_path = os.path.join(OUTPUT_DIR, f"kaggle_ablation_{name}.ipynb")
        save_notebook(out_nb, out_path)

    print("\nGenerating P7.4 Motion-filter θ ablation notebooks:")
    thetas = [0.00, 0.025, 0.075, 0.10]
    for theta in thetas:
        out_nb = make_motion_variant(nb, theta)
        theta_str = f"{theta:.3f}".replace('.', '_')
        out_path = os.path.join(OUTPUT_DIR, f"kaggle_ablation_motion_{theta_str}.ipynb")
        save_notebook(out_nb, out_path)

    print(f"\nDone! {len(variants) + len(thetas)} notebooks generated in notebooks/")


if __name__ == "__main__":
    main()
