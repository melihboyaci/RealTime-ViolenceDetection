"""
Builds the Kaggle preprocessing notebook with proper multi-line source format.
Each source line is a separate string in the array — Kaggle compatible.
"""
import json
import os

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'notebooks', 'kaggle_preprocessing.ipynb'
)

# Read the preprocessing module source
PREPROC_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'preprocessing.py'
)
with open(PREPROC_PATH, 'r', encoding='utf-8') as f:
    preproc_source = f.read()


def make_code_cell(source_str):
    lines = source_str.split('\n')
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + '\n')
        else:
            if line:  # skip trailing empty
                source_lines.append(line)
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }


def make_md_cell(source_str):
    lines = source_str.split('\n')
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + '\n')
        else:
            if line:
                source_lines.append(line)
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines
    }


cells = []

# ─── Cell 0: Markdown ────────────────────────────────────────
cells.append(make_md_cell(
"""# Kaggle Preprocessing Notebook

**Amac:** 2000 ham videoyu uctan uca isleyip egitime hazir `(30, 69)` sekans dosyalarina donusturmek.

### Kullanim:
1. Kaggle > Create > New Notebook > Accelerator: **GPU T4 x2**
2. Add Data > `real life violence situations dataset`
3. Hucreleri sirasiyla calistirin
4. Output sekmesinden `.npy` dosyalarini indirin

### Hucre Haritasi:
| # | Ne Yapar |
|---|---|
| 1 | preprocessing.py modulunu Kaggle ortamina yazar |
| 2 | Kutuphane import + sabitler |
| 3 | Dataset klasorlerini kesfeder |
| 4 | Stratified split (70/15/15) |
| 5 | Tum videolari isler > per-video .npy |
| 6 | Rastgele 5 .npy dogrulama |
| 7 | Sliding window + motion filter > sekanslar |
| 8 | Final dogrulama + ozet |"""
))

# ─── Cell 1: Write preprocessing.py ─────────────────────────
# Escape the module source for embedding in the notebook
escaped_source = preproc_source.replace('\\', '\\\\').replace("'", "\\'")

cell1_code = f'''# HUCRE 1: preprocessing.py modulunu Kaggle ortamina yaz
# Bu hucre tum preprocessing fonksiyonlarini /kaggle/working/preprocessing.py olarak yazar.
# Sonraki hucrelerde bu modulden import yapilir.

!pip install ultralytics -q

MODULE_CODE = \'\'\'
{preproc_source}
\'\'\'

with open('/kaggle/working/preprocessing.py', 'w') as f:
    f.write(MODULE_CODE)

print("preprocessing.py yazildi -> /kaggle/working/preprocessing.py")'''

cells.append(make_code_cell(cell1_code))

# ─── Cell 2: Imports ─────────────────────────────────────────
cells.append(make_code_cell(
"""# HUCRE 2: Import ve Sabitler
# preprocessing.py modulunden fonksiyonlari import eder.
# Notebook'a ozel sabitler (path, split oranlari) burada tanimlanir.

import os
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/kaggle/working')
from preprocessing import (
    process_single_video, create_sliding_windows,
    apply_motion_filter, FEATURE_DIM, SEQUENCE_LENGTH
)

# Notebook-only constants
POSE_MODEL_NAME = "yolov8n-pose.pt"
KAGGLE_INPUT = "/kaggle/input/real-life-violence-situations-dataset"
OUTPUT = "/kaggle/working"
VIOLENCE_LABEL = 1
NONVIOLENCE_LABEL = 0
SPLIT_RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# YOLO model yukle
pose_model = YOLO(POSE_MODEL_NAME)

print("Import tamamlandi. YOLO modeli yuklendi.")"""
))

# ─── Cell 3: Dataset discovery ───────────────────────────────
cells.append(make_code_cell(
'''# HUCRE 3: Dataset Kesfi
# Kaggle input dizininde Violence ve NonViolence klasorlerini bulur.
# Video sayilarini dogrular (beklenen: 1000+1000=2000).

for root, dirs, files in os.walk(KAGGLE_INPUT):
    if "Violence" in dirs or "NonViolence" in dirs:
        VIOLENCE_DIR = os.path.join(root, "Violence")
        NONVIOLENCE_DIR = os.path.join(root, "NonViolence")
        break

violence_videos = sorted(glob(os.path.join(VIOLENCE_DIR, "*")))
nonviolence_videos = sorted(glob(os.path.join(NONVIOLENCE_DIR, "*")))

print(f"Violence:    {len(violence_videos)} video  ->  {VIOLENCE_DIR}")
print(f"NonViolence: {len(nonviolence_videos)} video  ->  {NONVIOLENCE_DIR}")
print(f"Toplam:      {len(violence_videos) + len(nonviolence_videos)}")'''
))

# ─── Cell 4: Stratified split ────────────────────────────────
cells.append(make_code_cell(
'''# HUCRE 4: Stratified Split (70 / 15 / 15)
# Videolari train/val/test'e boler (sinif orani korunur).
# split_map dict'i olusturur: {video_path: (split_name, label)}

all_videos = violence_videos + nonviolence_videos
all_labels = [VIOLENCE_LABEL]*len(violence_videos) + [NONVIOLENCE_LABEL]*len(nonviolence_videos)

train_val_f, test_f, train_val_l, test_l = train_test_split(
    all_videos, all_labels, test_size=TEST_RATIO,
    stratify=all_labels, random_state=SPLIT_RANDOM_STATE)

val_adj = VAL_RATIO / (1 - TEST_RATIO)
train_f, val_f, train_l, val_l = train_test_split(
    train_val_f, train_val_l, test_size=val_adj,
    stratify=train_val_l, random_state=SPLIT_RANDOM_STATE)

split_map = {}
for f, l in zip(train_f, train_l): split_map[f] = ('train', l)
for f, l in zip(val_f, val_l):     split_map[f] = ('val', l)
for f, l in zip(test_f, test_l):   split_map[f] = ('test', l)

for name, files, labels in [('train',train_f,train_l),('val',val_f,val_l),('test',test_f,test_l)]:
    v = sum(1 for l in labels if l==1)
    nv = len(labels) - v
    print(f"{name:5s}: {len(files):4d} video  (V={v}, NV={nv})")
    pd.DataFrame({'filepath':files,'label':labels,'split':name}).to_csv(
        os.path.join(OUTPUT, f"{name}.csv"), index=False)'''
))

# ─── Cell 5: Process all videos ──────────────────────────────
cells.append(make_code_cell(
'''# HUCRE 5: Tum Videolari Isle > Per-Video .npy
# Her videoyu process_single_video() ile isler:
#   FPS sampling (10fps) > preprocess > YOLOv8n-Pose > multi-person
#   > normalize > 69-dim feature vector > .npy kaydet

for split in ['train', 'val', 'test']:
    for label in ['violence', 'nonviolence']:
        os.makedirs(os.path.join(OUTPUT, 'features', split, label), exist_ok=True)

process_log = []
skipped_videos = []

for video_path in tqdm(all_videos, desc="Video Isleme"):
    split_name, label = split_map[video_path]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    label_str = 'violence' if label == VIOLENCE_LABEL else 'nonviolence'
    npy_path = os.path.join(OUTPUT, 'features', split_name, label_str, f"{video_name}.npy")

    features, error = process_single_video(video_path, pose_model)

    if features is None:
        skipped_videos.append({'video': video_path, 'error': error})
        continue

    np.save(npy_path, features)
    process_log.append({'video': video_name, 'split': split_name,
                        'label': label_str, 'frames': features.shape[0]})

print(f"\\nIslenen: {len(process_log)} | Atlanan: {len(skipped_videos)}")
if skipped_videos:
    for s in skipped_videos[:5]:
        print(f"  SKIP: {s['video']}: {s['error']}")'''
))

# ─── Cell 6: Validation ──────────────────────────────────────
cells.append(make_code_cell(
'''# HUCRE 6: Dogrulama - Rastgele 5 .npy Kontrol
# Uretilen .npy dosyalarinin shape, dtype, NaN/Inf ve deger araligini kontrol eder.

npy_files = glob(os.path.join(OUTPUT, 'features', '**', '*.npy'), recursive=True)
print(f"Toplam .npy: {len(npy_files)}")

for f in random.sample(npy_files, min(5, len(npy_files))):
    arr = np.load(f)
    print(f"  {os.path.basename(f):40s} shape={str(arr.shape):12s} "
          f"min={arr.min():.2f}  max={arr.max():.2f}  "
          f"NaN={np.isnan(arr).any()}  Inf={np.isinf(arr).any()}")'''
))

# ─── Cell 7: Sliding window + motion filter ──────────────────
cells.append(make_code_cell(
'''# HUCRE 7: Sliding Window + Motion Filter > Sekanslar
# Her per-video .npy'den 30-frame pencereler olusturur.
# Violence: motion filter (theta=0.05), NonViolence: filtresiz.
# Train: undersampling ile sinif dengesi saglanir.

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT, 'sequences', split), exist_ok=True)

    v_seqs, nv_seqs = [], []

    for f in glob(os.path.join(OUTPUT, 'features', split, 'violence', '*.npy')):
        windows = create_sliding_windows(np.load(f))
        v_seqs.extend(apply_motion_filter(windows))

    for f in glob(os.path.join(OUTPUT, 'features', split, 'nonviolence', '*.npy')):
        nv_seqs.extend(create_sliding_windows(np.load(f)))

    if split == 'train' and len(nv_seqs) > len(v_seqs):
        random.seed(42)
        nv_seqs = random.sample(nv_seqs, len(v_seqs))

    X = np.array(v_seqs + nv_seqs, dtype=np.float32)
    y = np.array([1]*len(v_seqs) + [0]*len(nv_seqs), dtype=np.float32)

    np.save(os.path.join(OUTPUT, 'sequences', split, f'X_{split}.npy'), X)
    np.save(os.path.join(OUTPUT, 'sequences', split, f'y_{split}.npy'), y)

    print(f"{split:5s}: V={len(v_seqs):5d}  NV={len(nv_seqs):5d}  "
          f"Total={len(X):5d}  X={X.shape}")'''
))

# ─── Cell 8: Final validation ────────────────────────────────
cells.append(make_code_cell(
'''# HUCRE 8: Final Dogrulama + Indirme Ozeti
# Her split icin X/y shape, NaN/Inf ve sinif dagilimini kontrol eder.

print("=" * 55)
for split in ['train', 'val', 'test']:
    X = np.load(os.path.join(OUTPUT, 'sequences', split, f'X_{split}.npy'))
    y = np.load(os.path.join(OUTPUT, 'sequences', split, f'y_{split}.npy'))
    print(f"\\n{split.upper()}")
    print(f"  X={X.shape}  y={y.shape}")
    print(f"  V={int((y==1).sum())}  NV={int((y==0).sum())}")
    print(f"  NaN={np.isnan(X).any()}  Inf={np.isinf(X).any()}")
    print(f"  min={X.min():.3f}  max={X.max():.3f}")

if process_log:
    pd.DataFrame(process_log).to_csv(os.path.join(OUTPUT, 'preprocessing_log.csv'), index=False)
if skipped_videos:
    pd.DataFrame(skipped_videos).to_csv(os.path.join(OUTPUT, 'skipped_videos.csv'), index=False)

print("\\n" + "=" * 55)
print("Indirilecek dosyalar (Output sekmesi):")
for split in ['train', 'val', 'test']:
    for fname in [f'X_{split}.npy', f'y_{split}.npy']:
        p = os.path.join(OUTPUT, 'sequences', split, fname)
        mb = os.path.getsize(p) / 1024**2
        print(f"  sequences/{split}/{fname}  ({mb:.1f} MB)")
print("\\nPreprocessing tamamlandi!")'''
))

# ─── Write notebook ──────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'cells': cells
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook yazildi: {OUTPUT_PATH}")
print(f"Toplam hucre: {len(cells)}")
