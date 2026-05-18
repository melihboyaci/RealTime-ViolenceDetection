# DATA.md
# Veri Yönetimi, Dataset Detayları ve Doğrulama Kılavuzu

**Proje:** Real-Time Violence Detection via Skeletal Pose Analysis  
**Dataset:** Real Life Violence Situations Dataset  
**Kaynak:** Kaggle — `mohamedmustafa/real-life-violence-situations-dataset`  
**Versiyon:** 1.0.0

---

## İçindekiler

1. [Dataset Genel Bilgisi](#1-dataset-genel-bilgisi)
2. [İndirme ve Klasör Yapısı](#2-i̇ndirme-ve-klasör-yapısı)
3. [Ham Veri İstatistikleri](#3-ham-veri-i̇statistikleri)
4. [Split Stratejisi ve Dosya Listesi](#4-split-stratejisi-ve-dosya-listesi)
5. [Preprocessing Çıktı Yapısı (.npy)](#5-preprocessing-çıktı-yapısı-npy)
6. [Motion Filter Sonrası Beklenen Sekans Sayıları](#6-motion-filter-sonrası-beklenen-sekans-sayıları)
7. [Veri Doğrulama Checklist'i](#7-veri-doğrulama-checklisṫi)
8. [Bilinen Veri Sorunları](#8-bilinen-veri-sorunları)
9. [Karar Günlüğü](#9-karar-günlüğü)

---

## 1. Dataset Genel Bilgisi

| Özellik | Değer |
|---|---|
| Dataset Adı | Real Life Violence Situations Dataset |
| Kaggle URL | `https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset` |
| Toplam Video | 2000 |
| Violence Sınıfı | 1000 video — gerçek sokak kavgaları |
| NonViolence Sınıfı | 1000 video — spor, yemek, yürüyüş, sohbet |
| Kaynak Platform | YouTube |
| Lisans | CC BY 4.0 |
| Çekim Koşulları | Değişken — farklı FPS, çözünürlük, aydınlatma, kamera açısı |
| Etiketleme Granülaritesi | Video-level (frame-level etiket yok) |
| Format | .mp4, .avi (karışık) |

### Neden Bu Dataset?

- **Denge:** 1000/1000 — sınıf dengesizliği problemi yok
- **Çeşitlilik:** NonViolence sınıfında spor, yemek, dans gibi çeşitli aktiviteler → model "hareket var = Violence" kısayoluna gidemez
- **Gerçek dünya:** YouTube kayıtları → değişken ışık, çözünürlük, açı — laboratuvar verisi değil
- **Boyut:** 1000 Violence videosu, UCF-Crime'ın fighting kategorisinden (~150 klip) çok daha fazla

---

## 2. İndirme ve Klasör Yapısı

### İndirme (Kaggle CLI)

```bash
# Kaggle API kurulu değilse
pip install kaggle

# API token ~/.kaggle/kaggle.json konumunda olmalı
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset

# Zip'i aç
unzip real-life-violence-situations-dataset.zip -d data/raw/
```

### Beklenen Ham Klasör Yapısı

```
project/
└── data/
    └── raw/
        └── Real Life Violence Dataset/
            ├── Violence/
            │   ├── V_1.mp4
            │   ├── V_2.mp4
            │   └── ... (1000 video)
            └── NonViolence/
                ├── NV_1.mp4
                ├── NV_2.mp4
                └── ... (1000 video)
```

### Tam Proje Klasör Yapısı

```
project/
├── data/
│   ├── raw/                          ← Ham videolar (dokunulmaz)
│   │   └── Real Life Violence Dataset/
│   │       ├── Violence/
│   │       └── NonViolence/
│   ├── features/                     ← Preprocessing çıktıları (.npy)
│   │   ├── train/
│   │   │   ├── violence/
│   │   │   └── nonviolence/
│   │   ├── val/
│   │   │   ├── violence/
│   │   │   └── nonviolence/
│   │   └── test/
│   │       ├── violence/
│   │       └── nonviolence/
│   └── splits/                       ← Split dosyaları
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── docs/
├── models/
└── src/
```

> **Kural:** `data/raw/` klasörüne hiçbir script yazma işlemi yapmamalıdır. Ham videolar salt okunur kaynak olarak kalır. Tüm işlem çıktıları `data/features/` altına yazılır.

---

## 3. Ham Veri İstatistikleri

### Sınıf Dağılımı

```
Violence    : 1000 video  (%50.0)
NonViolence : 1000 video  (%50.0)
Toplam      : 2000 video
```

Veri seti dengeli — ek class weighting veya oversampling gerekmez.  
Motion filter sonrası Violence sekans sayısı azalır; bu noktada undersampling uygulanır (bkz. Bölüm 6).

### Video Karakteristikleri (Beklenen Aralıklar)

| Özellik | Beklenen Aralık | Not |
|---|---|---|
| Süre | 1–10 saniye | Çoğu 3–7 sn |
| Native FPS | 24–60 FPS | Standartlaştırılacak → 10 FPS |
| Çözünürlük | 240p–1080p | Resize → 640×640 |
| Format | .mp4, .avi | OpenCV her ikisini okur |

> Bu değerler tahminidir. Preprocessing sırasında her videonun gerçek FPS ve çözünürlüğü loglanmalıdır. Aykırı değerler (0 FPS, 0 frame) bozuk video işaretidir — atla ve logla.

---

## 4. Split Stratejisi ve Dosya Listesi

### Oranlar

```
Train :  %70 → 700 Violence + 700 NonViolence = 1400 video
Val   :  %15 → 150 Violence + 150 NonViolence =  300 video
Test  :  %15 → 150 Violence + 150 NonViolence =  300 video
```

### Stratified Split — Neden Zorunlu?

Rastgele bölme şans eseri bir bölümde fazla Violence toplayabilir.  
`sklearn.model_selection.train_test_split(stratify=labels)` sınıf oranını garanti eder.

```python
from sklearn.model_selection import train_test_split

violence_files    = sorted(glob("data/raw/.../Violence/*.mp4"))
nonviolence_files = sorted(glob("data/raw/.../NonViolence/*.mp4"))

all_files  = violence_files + nonviolence_files
all_labels = [1]*1000 + [0]*1000

# Önce test ayır
train_val_files, test_files, train_val_labels, test_labels = train_test_split(
    all_files, all_labels,
    test_size=0.15,
    stratify=all_labels,
    random_state=42
)

# Sonra val ayır
train_files, val_files, train_labels, val_labels = train_test_split(
    train_val_files, train_val_labels,
    test_size=0.15/0.85,          # 0.85'in %17.6'sı ≈ toplamın %15'i
    stratify=train_val_labels,
    random_state=42
)
```

`random_state=42` — reproducibility için sabittir. Değiştirme.

### CSV Format (data/splits/)

Her split için bir CSV dosyası üretilir:

```
# train.csv
filepath,label,split
data/raw/.../Violence/V_1.mp4,1,train
data/raw/.../NonViolence/NV_1.mp4,0,train
...
```

```
# val.csv ve test.csv — aynı format
```

CSV dosyaları split'in tekrar üretilmesini engeller.  
Bir kez üretilip `data/splits/` altına kaydedilir, bir daha çalıştırılmaz.

### Test Seti İzolasyon Kuralı

```
test.csv içeriği eğitim boyunca açılmaz.
Hiperparametre kararları yalnızca val.csv üzerinden alınır.
Test seti yalnızca final raporlama için bir kez açılır.
```

Bu kural ihlal edilirse test seti artık bağımsız değerlendirme için geçersizdir.

---

## 5. Preprocessing Çıktı Yapısı (.npy)

### Per-Video .npy Dosyası

Her video için preprocessing pipeline çalıştıktan sonra:

```
Shape  : (n_frames, 69)
dtype  : float32
İçerik : Her satır bir frame'in 69-boyutlu normalize özellik vektörü

69 özelliğin dökümü:
  [0:34]  → Kişi 1: 17 keypoint × (x, y) — hip-centered, shoulder-hip scaled
  [34:68] → Kişi 2: 17 keypoint × (x, y) — aynı normalizasyon, yoksa sıfır
  [68]    → normalize bbox merkez mesafesi — frame_width'e bölünmüş
```

### Dosya Adlandırma Kuralı

```
{split}_{label}_{orijinal_dosya_adı}.npy

Örnekler:
  train_violence_V_042.npy
  train_nonviolence_NV_017.npy
  val_violence_V_712.npy
  test_nonviolence_NV_891.npy
```

### Kayıt Konumu

```
data/features/
├── train/
│   ├── violence/      → 700 .npy dosyası
│   └── nonviolence/   → 700 .npy dosyası
├── val/
│   ├── violence/      → 150 .npy dosyası
│   └── nonviolence/   → 150 .npy dosyası
└── test/
    ├── violence/      → 150 .npy dosyası
    └── nonviolence/   → 150 .npy dosyası
```

### Bozuk Video Kenar Durumları

```
Durum                          Davranış
─────────────────────────────────────────────────────
Video açılamıyor               Atla, preprocessing.log'a yaz
Native FPS = 0 veya None       Atla, preprocessing.log'a yaz
frame_count < 30               Sıfır padding → (30, 69) yap, kaydet
YOLOv8 hiç kişi bulamazsa     Sıfır vektörü yaz, devam et
torso_height < 1e-6            Sıfır vektörü yaz, devam et
```

Tüm atlamalar `data/preprocessing.log` dosyasına yazılır.  
Log dosyası preprocessing bittikten sonra gözden geçirilir.

---

## 6. Motion Filter Sonrası Beklenen Sekans Sayıları

### Sliding Window Tahmini (Filtreleme Öncesi)

```
Ortalama video süresi  : ~5 saniye @ 10 FPS = ~50 frame
Pencere               : 30 frame, stride=15
Pencere/video         : (50 - 30) / 15 + 1 ≈ 2.3 ≈ ortalama 2-3 sekans

Train (1400 video)    : ~3,000 sekans (Violence) + ~3,000 sekans (NonViolence)
Val   ( 300 video)    : ~  600 sekans (Violence) + ~  600 sekans (NonViolence)
Test  ( 300 video)    : ~  600 sekans (Violence) + ~  600 sekans (NonViolence)
```

### Motion Filter Etkisi (θ = 0.05, Violence Only)

```
Violence pencereleri filtreleme öncesi : ~3,000 (train)
Beklenen eleme oranı                  : %20–40
Violence pencereleri filtreleme sonrası: ~1,800–2,400 (train)
```

### Sınıf Dengesi Yeniden Kurulması

Motion filter Violence sekanslarını azalttığı için NonViolence'dan undersampling yapılır:

```python
n_violence    = len(filtered_violence_sequences)   # filtreleme sonrası
n_nonviolence = len(all_nonviolence_sequences)

if n_nonviolence > n_violence:
    nonviolence_sequences = random.sample(
        all_nonviolence_sequences,
        k=n_violence,
        random_state=42
    )

# Final train seti: n_violence + n_violence (dengeli)
```

### Tahmini Final Sekans Sayıları

```
                Violence    NonViolence    Toplam
Train           ~2,000       ~2,000        ~4,000
Val             ~  450       ~  450        ~  900
Test            ~  450       ~  450        ~  900
─────────────────────────────────────────────────
Genel Toplam                              ~5,800
```

> Bu tahminler gerçek preprocessing sonrası güncellenir. Gerçek değerleri bu tabloya yazın.

---

## 7. Veri Doğrulama Checklist'i

Preprocessing tamamlandıktan sonra, eğitime başlamadan önce aşağıdaki kontroller yapılır:

### Adım 1 — Ham Veri Kontrolü

```
□ data/raw/Violence/      → tam olarak 1000 dosya var mı?
□ data/raw/NonViolence/   → tam olarak 1000 dosya var mı?
□ preprocessing.log       → kaç video atlandı? (%5'ten fazlaysa araştır
□ Atlanan videolar        → nedenleri tutarlı mı? (bozuk dosya, çok kısa vb.)
```

### Adım 2 — .npy Dosya Kontrolü

```
□ data/features/train/violence/      → ~700 .npy dosyası
□ data/features/train/nonviolence/   → ~700 .npy dosyası
□ Rastgele 5 .npy aç → shape (N, 69) mi?
□ dtype float32 mi?
□ İçinde NaN veya Inf var mı?
   → np.isnan(arr).any() ve np.isinf(arr).any() ile kontrol et
□ Değer aralığı makul mü?
   → Hip-centered normalize koordinatlar genellikle [-5, 5] arasında
   → Merkez mesafesi [0, 1] arasında
```

### Adım 3 — Split Kontrolü

```
□ train.csv → 1400 satır, %50 Violence / %50 NonViolence
□ val.csv   → 300 satır,  %50 Violence / %50 NonViolence
□ test.csv  → 300 satır,  %50 Violence / %50 NonViolence
□ Üç CSV arasında ortak dosya yok (sızıntı kontrolü):
   train_set ∩ val_set  = ∅
   train_set ∩ test_set = ∅
   val_set   ∩ test_set = ∅
```

### Adım 4 — Sekans Kontrolü

```
□ Motion filter çalıştı → Violence sekans sayısı azaldı mı?
□ Undersampling yapıldı → train'de Violence ≈ NonViolence sekans sayısı
□ Tüm sekansların shape'i (30, 69) mi?
□ Sıfır vektörü olan sekanslar var mı? → kaçı? (%10'dan fazlaysa sorun var
```

### Adım 5 — Hızlı Görsel Doğrulama

```python
import numpy as np
import matplotlib.pyplot as plt

# Rastgele bir Violence sekansı yükle
seq = np.load("data/features/train/violence/train_violence_V_042.npy")
print(f"Shape: {seq.shape}")           # (N, 69) olmalı
print(f"Min: {seq.min():.3f}")         # -5 ile 5 arası beklenir
print(f"Max: {seq.max():.3f}")
print(f"NaN: {np.isnan(seq).any()}")   # False olmalı

# Motion skorunu görselleştir
motion = np.linalg.norm(np.diff(seq, axis=0), axis=1)
plt.plot(motion)
plt.axhline(y=0.05, color='r', linestyle='--', label='θ=0.05')
plt.title("Motion Score Over Time")
plt.show()
# Kavga anlarında motion skoru θ'nın üzerinde olmalı
```

---

## 8. Bilinen Veri Sorunları

**Sorun 1 — Video-Level Etiket Gürültüsü (Kısmen Çözüldü):**  
Violence videolarının sakin başlangıç bölümleri Violence=1 etiketi taşır. Motion filter (θ=0.05) bu sekansları büyük ölçüde eler. Tam çözüm frame-level etiketleme gerektirir — dataset'te mevcut değil, kapsam dışı.

**Sorun 2 — NonViolence'da Spor İçeriği:**  
Boks, güreş, martial arts gibi sporlar NonViolence sınıfında bulunabilir. Bu videolardaki yüksek hareket modeli kafa karıştırabilir. Motion filter bu videoları elemez — NonViolence'a uygulanmaz. Bu durum test setinde false negative olarak görünebilir.

**Sorun 3 — Değişken Video Kalitesi:**  
240p gibi çok düşük çözünürlüklü videolarda YOLOv8n-Pose keypoint confidence'ları düşer, daha fazla sıfır vektörü oluşur. Preprocessing log'unda confidence düşük olan video sayısını takip et.

**Sorun 4 — Çok Kısa Videolar:**  
30 frame'den kısa videolar sıfır padding ile doldurulur. Bu sekanslar bilgi içermez ama eğitime girer. Preprocessing log'unda kaç videonun padding aldığını kaydet; oranı yüksekse bu videoları eğitim setinden çıkarmayı değerlendir.

---

## 9. Karar Günlüğü

| # | Karar | Seçilen | Reddedilen | Gerekçe |
|---|---|---|---|---|
| 1 | Dataset | Real Life Violence Situations | UCF-Crime, MCFD | 1000/1000 dengeli, gerçek dünya, yeterli Violence verisi |
| 2 | Split oranı | 70/15/15 | 80/20 | Test seti izolasyonu, val ile hiperparametre optimizasyonu |
| 3 | Split yöntemi | Stratified | Rastgele | Sınıf dengesi garantisi |
| 4 | random_state | 42 | — | Reproducibility — değiştirme |
| 5 | Ham veri politikası | Salt okunur | — | Kaynak veri bütünlüğü |
| 6 | Bozuk video politikası | Atla + logla | Durdur | Pipeline sürekliliği |
| 7 | Undersampling | NonViolence'dan | Oversampling | Motion filter sonrası basit denge kurma |
| 8 | Etiket gürültüsü çözümü | Motion filter (θ=0.05) | Manuel etiket | Otomatik, pipeline verisi üzerinde çalışır |

---

*Son güncelleme: Veri doğrulama checklist'i ve bilinen sorunlar eklendi. Bir sonraki belge: `README.md`*