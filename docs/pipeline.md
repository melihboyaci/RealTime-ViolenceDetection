# PIPELINE.md
# Real-Time Violence Detection via Skeletal Pose Analysis

**Proje Adı:** Real-Time Violence Detection via Skeletal Pose Analysis  
**Problem Tipi:** Binary Video Classification — Violence / NonViolence  
**Teknik Yığın:** OpenCV · YOLOv8n-Pose · PyTorch · GRU  
**Versiyon:** 1.0.0

---

## İçindekiler

1. [Genel Mimari](#1-genel-mimari)
2. [Veri Organizasyonu ve Split Stratejisi](#2-veri-organizasyonu-ve-split-stratejisi)
3. [Video Okuma ve FPS Standardizasyonu](#3-video-okuma-ve-fps-standardizasyonu)
4. [Görüntü Ön İşleme](#4-görüntü-ön-işleme)
5. [Pose Estimation — YOLOv8n-Pose](#5-pose-estimation--yolov8n-pose)
6. [Multi-Person Stratejisi (Seçenek B+)](#6-multi-person-stratejisi-seçenek-b)
7. [Koordinat Normalizasyonu](#7-koordinat-normalizasyonu)
8. [Özellik Vektörü ve .npy Kayıt](#8-özellik-vektörü-ve-npy-kayıt)
9. [Sekans Oluşturma — Sliding Window](#9-sekans-oluşturma--sliding-window)
9B. [Motion-Based Sequence Filtering](#9b-motion-based-sequence-filtering)
10. [GRU Model Mimarisi](#10-gru-model-mimarisi)
11. [Eğitim Stratejisi](#11-eğitim-stratejisi)
12. [Değerlendirme Metrikleri ve Ablation](#12-değerlendirme-metrikleri-ve-ablation)
13. [Gerçek Zamanlı Inference Pipeline](#13-gerçek-zamanlı-inference-pipeline)
14. [Bilinen Kısıtlamalar](#14-bilinen-kısıtlamalar)
15. [Karar Günlüğü](#15-karar-günlüğü)

---

## 1. Genel Mimari

Bu pipeline, ham video akışını ikili sınıflandırma kararına dönüştüren uçtan uca bir sistemdir.
İki ana fazdan oluşur: **Offline Preprocessing** (eğitim öncesi, bir kez çalışır) ve **Online Inference** (gerçek zamanlı, her frame için çalışır).

```
════════════════════════════════════════════════════════════════
                     OFFLİNE PREPROCESSING
════════════════════════════════════════════════════════════════

Ham Video (.mp4)
      │
      ▼
[Aşama 3] FPS Standardizasyonu (→ 10 FPS)
      │
      ▼
[Aşama 4] Görüntü Ön İşleme
          CLAHE (koşullu) → Gaussian Blur 3×3 → Resize 640×640 → BGR→RGB
      │
      ▼
[Aşama 5] YOLOv8n-Pose
          17 COCO Keypoint çıkarımı · Confidence filtering
      │
      ▼
[Aşama 6] Multi-Person B+ Stratejisi
          X eksenine göre sıralama · Top-2 kişi seçimi · Merkez mesafesi
      │
      ▼
[Aşama 7] Koordinat Normalizasyonu
          Hip centering · Shoulder-hip scaling
      │
      ▼
[Aşama 8] Özellik Vektörü → .npy kayıt
          Shape: (n_frames, 69)
      │
      ▼
[Aşama 9] Sliding Window Sekans Oluşturma
          Shape: (n_sequences, 30, 69)
      │
      ▼
[Aşama 9B] Motion-Based Sequence Filtering  ← YENİ
           Violence pencereleri: düşük hareketliler elenir
           NonViolence pencereleri: filtreye tabi değil
           Çıktı: temizlenmiş sekans seti

════════════════════════════════════════════════════════════════
                         EĞİTİM
════════════════════════════════════════════════════════════════

[Aşama 10–11] GRU Eğitimi
          Input (batch, 30, 69) → GRU → Sigmoid → BCELoss

════════════════════════════════════════════════════════════════
                     ONLİNE INFERENCE
════════════════════════════════════════════════════════════════

Kamera / Video → Aşama 4 → Aşama 5 → Aşama 6 → Aşama 7
      │
      ▼
FIFO Buffer (30 frame)
      │
      ▼
GRU Modeli → Olasılık Skoru
      │
      ▼
Threshold Kararı → Görselleştirme
```

---

## 2. Veri Organizasyonu ve Split Stratejisi

### Dataset

| Özellik | Değer |
|---|---|
| Kaynak | Real Life Violence Situations Dataset (Kaggle) |
| Violence | 1000 video — gerçek sokak kavgaları, çeşitli ortamlar |
| NonViolence | 1000 video — spor, yemek, yürüyüş, sohbet |
| Toplam | 2000 video |
| Çekim Koşulları | YouTube kayıtları — değişken FPS, çözünürlük, ışık |
| Etiketleme | Video seviyesinde — frame seviyesinde etiket yok |

### Split Stratejisi

```
Toplam: 2000 video
├── Train:      1400 video (%70) — 700 Violence, 700 NonViolence
├── Validation:  300 video (%15) — 150 Violence, 150 NonViolence
└── Test:        300 video (%15) — 150 Violence, 150 NonViolence
```

**Stratified split zorunludur.** Rastgele bölme sınıf dengesini bozabilir.  
`sklearn.model_selection.train_test_split(stratify=labels)` kullanılır.

**Test seti eğitim boyunca dokunulmaz.** Hiperparametre kararları yalnızca validation seti üzerinden alınır. Test seti yalnızca final raporlama için açılır.

---

## 3. Video Okuma ve FPS Standardizasyonu

### Hedef

Dataset'teki videolar farklı FPS değerlerinde (genellikle 24–60 FPS). Modele giren zaman serilerinin tutarlı temporal çözünürlükte olması gerekir.

### Yöntem

Her video için:
1. `cv2.VideoCapture` ile aç
2. `cap.get(cv2.CAP_PROP_FPS)` ile native FPS oku
3. `frame_interval = round(native_fps / target_fps)` hesapla (`target_fps = 10`)
4. Her `frame_interval`'inci kareyi al, araları atla

```
Native 30 FPS → her 3. kare → efektif 10 FPS
Native 24 FPS → her 2-3. kare → efektif ~10 FPS
Native 60 FPS → her 6. kare → efektif 10 FPS
```

### Neden 10 FPS?

İnsan yumruğu veya kavga hareketi ~200–400ms sürer.  
10 FPS'te bu hareket 2–4 karede temsil edilir — yeterli temporal bilgi.  
30 FPS'te aynı hareket 6–12 kare olur, bitişik kareler çok benzer, bilgi tekrarı artar.  
10 FPS ayrıca işlem yükünü 3'te 1'e düşürür.

---

## 4. Görüntü Ön İşleme

### Dataset Karakteristiği

YouTube videoları şu gürültü türlerini içerir:
- **Compression artifact:** H.264/H.265 blok yapısı, özellikle hareket anlarında
- **Motion blur:** Hızlı hareketlerde (kavga sahneleri) belirgin
- **Değişken aydınlatma:** Gece sahneleri, iç mekan/dış mekan karışık
- **Salt & Pepper yok**, **Gaussian noise düşük**

### Preprocessing Zinciri

Her frame için sırasıyla:

**Adım 1 — Parlaklık Kontrolü (Koşullu CLAHE)**
```
frame → cv2.cvtColor(BGR→GRAY) → mean parlaklık hesapla

Eğer mean < 50:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    her kanal için CLAHE uygula (LAB color space üzerinden)
```
CLAHE yalnızca karanlık karelere uygulanır. Aydınlık karelerde gereksiz kontrast artışı önlenir.

**Adım 2 — Hafif Gaussian Blur**
```
cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5)
```
Compression artifact yumuşatması için minimal blur. Kernel 3×3 — YOLOv8'in keypoint tespitini bozmayacak kadar hafif.  
Bilateral kullanılmıyor — real-time inference'da çok yavaş, eğitim/inference tutarsızlığı yaratır.

**Adım 3 — Resize**
```
cv2.resize(frame, (640, 640))
```
YOLOv8'in varsayılan input boyutu. Aspect ratio bozulması kabul edilir — pose estimation için yeterli.

**Adım 4 — Renk Uzayı Dönüşümü**
```
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
OpenCV BGR okur, YOLOv8 (PyTorch) RGB bekler. Bu adım atlanırsa renk kanalları ters gider, tahminler anlamsızlaşır.

### Kesin Sıralama

```
BGR Frame → CLAHE (koşullu) → GaussianBlur → Resize → BGR→RGB → YOLOv8
```

> **Kritik:** Resize, blur'dan sonra gelir. Blur sonrası resize, alias artifact riskini azaltır.

---

## 5. Pose Estimation — YOLOv8n-Pose

### Model Seçimi

`yolov8n-pose.pt` — nano variant. Neden nano?

| Variant | Parametre | FPS (CPU) | Keypoint Doğruluğu |
|---|---|---|---|
| yolov8n-pose | 3.3M | ~15–20 | Yeterli |
| yolov8s-pose | 11.6M | ~8–12 | İyi |
| yolov8m-pose | 26.4M | ~4–6 | Çok iyi |

Real-time inference hedefi var. Nano, CPU'da bile kabul edilebilir FPS verir.  
Eğer GPU varsa `yolov8s-pose` tercih edilebilir.

### COCO 17 Keypoint Anatomisi

```
Index  Keypoint
  0    Burun
  1    Sol Göz        2    Sağ Göz
  3    Sol Kulak      4    Sağ Kulak
  5    Sol Omuz       6    Sağ Omuz
  7    Sol Dirsek     8    Sağ Dirsek
  9    Sol Bilek     10    Sağ Bilek
 11    Sol Kalça     12    Sağ Kalça
 13    Sol Diz       14    Sağ Diz
 15    Sol Ayak      16    Sağ Ayak
```

### Confidence Filtering

YOLOv8-Pose her keypoint için `(x, y, confidence)` döndürür.

```
Eğer keypoint confidence < 0.5:
    keypoint koordinatları → (0.0, 0.0) olarak işaretle
```

(0, 0) koordinatı normalizasyon sonrasında modele "bu keypoint güvenilmez" sinyali verir.  
Threshold 0.5 — literatürde standart değer. Değiştirme gerekirse MODEL.md'ye not düş.

---

## 6. Multi-Person Stratejisi (Seçenek B+)

### Neden Tek Kişi Yeterli Değil

Kavga tanımı gereği en az iki kişi içerir. Tek kişinin hareketlerine bakarak boks antrenmanlıyla gerçek kavgayı ayırt etmek mümkün değildir — etkileşim bilgisi zorunludur.

### Neden Seçenek C (Joint-Specific Mesafe) Reddedildi

Bilek-kafa gibi spesifik eklem mesafeleri kamera açısı bozulduğunda veya kişi arkasını döndüğünde YOLOv8'in o eklemi 0 olarak döndürmesine yol açar. Sıfır üzerinden hesaplanan mesafe anlamsız veri üretir. Seçenek C'nin semantik fikri (yakınlık = şiddet) korunur ama daha sağlam bir noktadan ölçülür.

### B+ Uygulama Adımları

**Adım 1 — Kişi Seçimi**
```
YOLO'nun tespit ettiği tüm kişiler → bounding box alanına göre büyükten küçüğe sırala
→ En büyük 2 kişiyi al
→ Eğer 1 kişi varsa: İkinci kişi dizisi sıfır vektörü ile doldurulur (zero-padding)
→ Eğer 0 kişi varsa: Frame tamamen sıfır — buffer'a sıfır vektörü eklenir
```

**Adım 2 — X Eksenine Göre Sıralama**
```
Seçilen 2 kişiyi bounding box X merkezine göre soldan sağa sırala:
→ Dizi her zaman: [Soldaki Kişi | Sağdaki Kişi]
```
Bu sıralama identity switch problemini büyük ölçüde çözer.

> **Bilinen Kısıt:** Kavganın zirve anında iki kişi ekranda çakışabilir. Bu anda X merkezleri birbirine çok yaklaşır ve sıralama kararsızlaşabilir. Bu durum kaçınılmaz ama istisnadır. Bkz. [Bölüm 14](#14-bilinen-kısıtlamalar).

**Adım 3 — Interaction Feature: Normalize Merkez Mesafesi**
```python
center1 = bbox1_center  # (x1, y1)
center2 = bbox2_center  # (x2, y2)

raw_distance = math.dist(center1, center2)
normalized_distance = raw_distance / frame_width   # → [0, 1] arası
```

Ham piksel mesafesi çözünürlüğe bağımlıdır — frame genişliğine bölmek onu ölçekten bağımsız hale getirir. Bu normalizasyon, iskelet koordinat normalizasyonuyla tutarlılık sağlar.

Eğer yalnızca 1 kişi varsa: `normalized_distance = 1.0` (maksimum uzaklık, etkileşim yok)

**Adım 4 — Final Özellik Vektörü**
```
[Kişi_1 normalized skeleton (34)] +
[Kişi_2 normalized skeleton (34)] +
[normalized_center_distance (1)]
= 69 boyutlu vektör / frame
```

---

## 7. Koordinat Normalizasyonu

Her kişi için bağımsız olarak uygulanır.

### Adım 1 — Hip Centering (Origin Kaydırma)

```python
hip_mid_x = (keypoints[11][0] + keypoints[12][0]) / 2
hip_mid_y = (keypoints[11][1] + keypoints[12][1]) / 2

for i in range(17):
    keypoints[i][0] -= hip_mid_x
    keypoints[i][1] -= hip_mid_y
```

Sonuç: Kişi ekranın neresinde olursa olsun, kalça orijin noktasıdır.

### Adım 2 — Ölçek Normalizasyonu

```python
shoulder_mid_y = (keypoints[5][1] + keypoints[6][1]) / 2
torso_height = abs(hip_mid_y - shoulder_mid_y)

if torso_height > 1e-6:   # sıfıra bölme koruması
    for i in range(17):
        keypoints[i][0] /= torso_height
        keypoints[i][1] /= torso_height
else:
    # Torso yüksekliği hesaplanamıyor — bu frame'i sıfır vektörü ile doldur
    keypoints = np.zeros((17, 2))
```

Sonuç: Kişi kameraya ne kadar yakın ya da uzak olursa olsun, iskelet boyutu aynı ölçeğe gelir.

### Neden Bu Sıra?

Hip centering önce gelir. Scaling sonra gelir.  
Ters yapılırsa scaling, orijinal piksel koordinatlarını ölçekler ve kaydırma anlamlı olmaz.

### Confidence-0 Keypoint'lerin Davranışı

Confidence < 0.5 olarak işaretlenmiş keypoint'ler (0, 0) olarak girildi.  
Hip centering sonrası bu noktalar `(-hip_mid_x, -hip_mid_y)` olur — orijin değil.  
Bu semantik bir sorun yaratır.

**Çözüm:** Confidence masking — normalizasyondan önce düşük-confidence keypoint'leri `(hip_mid_x, hip_mid_y)` olarak set et. Bu noktalar normalizasyon sonrası tam (0, 0) olur ve modele "bu bilgi yok" sinyali verir.

---

## 8. Özellik Vektörü ve .npy Kayıt

### Frame-Level Çıktı

Her video frame'i için 69 boyutlu vektör:

```
[kp1_x, kp1_y, kp2_x, kp2_y, ..., kp17_x, kp17_y,   ← Kişi 1 (34 değer)
 kp1_x, kp1_y, kp2_x, kp2_y, ..., kp17_x, kp17_y,   ← Kişi 2 (34 değer, yoksa 0)
 normalized_center_distance]                           ← Interaction (1 değer)
```

### Video-Level Kayıt

```
Her video → (n_frames, 69) shape numpy array → .npy dosyası

Dosya adı: {split}_{label}_{video_id}.npy
Örnek: train_violence_0042.npy

features/
├── train/
│   ├── violence/
│   └── nonviolence/
├── val/
└── test/
```

Bu aşamadan sonra ham video dosyalarına bir daha ihtiyaç yoktur.

---

## 9. Sekans Oluşturma — Sliding Window

### Parametreler

| Parametre | Değer | Gerekçe |
|---|---|---|
| Pencere boyutu | 30 frame | 3 saniye @ 10 FPS — kavga anını kapsayan minimum süre |
| Stride (train/val) | 15 frame | %50 overlap — veri artırımı, sekans sayısını 2× artırır |
| Stride (inference) | 1 frame | Her frame'de güncellenen karar — minimum gecikme |

### Label Stratejisi

Video-level etiket, videonun tüm sekanslarına atanır — **ancak Violence videoları için Motion Filtering uygulanır** (bkz. Aşama 9B).

```
violence_video_042.npy → n sekans üret → motion filter uygula → kalan sekanslar Violence=1
nonviolence_video_017.npy → n sekans → filtresiz → tümü NonViolence=0
```

### Kısa Video Kenar Durumu

Eğer bir video 30 frame'den kısaysa:
```
Video sonuna sıfır vektörleri ekle (right-padding) → tam 30 frame yap
Sadece bu video'dan tek bir sekans üretilir
```

### Veri Boyutu Tahmini

```
1000 video × ortalama 200 frame/video = 200,000 toplam frame
Stride=15 → ~13,000 sekans (filtreleme öncesi, train seti için ~9,100)
Motion filter sonrası Violence sekansları ~%20-30 azalır → sınıf dengesi bozulur
→ NonViolence'dan da eşit sayıda sekans alınır (random undersampling)
Her sekans: (30, 69) × float32 = 8,280 byte ≈ 8 KB
Toplam: ~10,000 × 8 KB ≈ 80 MB — yönetilebilir
```

---

## 9B. Motion-Based Sequence Filtering

### Problem

Video-level etiketleme ciddi bir gürültü kaynağıdır. Violence olarak etiketlenmiş bir videoda kavga yalnızca son birkaç saniyede başlayabilir. Önceki sakin bölümler GRU'ya "sabit duran iki kişi = Violence" yanlış örüntüsünü öğretir. Bu, gerçek dünya demosunda false positive üretir.

```
Violence videosu (20 sn @ 10 FPS = 200 frame):
├── Frame 0–150:   iki adam konuşuyor  → iskelet: sabit, düşük hız
└── Frame 150–200: kavga              → iskelet: hızlı, büyük açılar

Tüm sekanslar Violence=1 alırsa:
  GRU öğrenir: "sabit iskelet = Violence" ← YANLIŞ
  Demo: yan yana duran iki kişi = Violence alarmı ← FALSE POSITIVE
```

### Çözüm: Motion Skoru ile Otomatik Filtreleme

Zaten hesaplanmış normalize iskelet vektörleri üzerinde çalışır. Ekstra model, araç veya manuel iş gerektirmez.

**Motion Skoru Hesabı:**

Bir penceredeki ardışık frame çiftleri arasındaki ortalama L2 mesafesi:

```
motion_score(pencere) = mean( ||frame[t] - frame[t-1]||₂ )
                        t = 1, 2, ..., 29
```

`frame[t]` → 69 boyutlu normalize vektör  
Her ardışık çift arasındaki Öklid mesafesi → 29 değer → ortalaması = motion skoru

**Filtreleme Kararı:**

```
Violence videosu pencereleri için:

  motion_score < θ ?
  ├─ Evet (θ = 0.05) → ATIL ❌
  │   Bu pencere sakin bir bölgeyi temsil eder.
  │   Violence=1 etiketi bu sekans için güvenilmez.
  └─ Hayır → KABUL ✓
      Bu pencerede anlamlı hareket var.
      Violence=1 etiketi geçerli kabul edilir.

NonViolence videoları → filtreye tabi değil
  Sakin durmak NonViolence'ın doğru temsilidir, elenmez.
```

**θ (Motion Threshold) Kalibrasyonu:**

Başlangıç değeri `θ = 0.05` (normalize koordinat uzayında).

```
θ çok küçük → az sekans elenir → gürültü yüksek kalır
θ çok büyük → çok sekans elenir → Violence örnekleri azalır, model yetersiz kalır
```

Doğru θ değeri ablation ile belirlenir (bkz. Bölüm 12, Ablation 4).

**Sınıf Dengesi Yeniden Kurulması:**

Motion filter sonrası Violence sekans sayısı azalır. Sınıf dengesini korumak için:

```
filtered_violence_count = N
nonviolence_count       = M  (M > N ise)

→ NonViolence'dan rastgele N sekans seç (undersampling)
→ Eğitim seti dengeli: N Violence + N NonViolence
```

### Neden Bu Yaklaşım Sağlamdır

Koordinat normalizasyonu sayesinde motion skoru çözünürlükten bağımsızdır. 1080p ve 480p videolarda aynı θ değeri tutarlı çalışır. Normalizasyon yapılmamış ham koordinatlarla bu yaklaşım kullanılamaz — bu iki aşamanın birbirini zorunlu kıldığının göstergesidir.

### Sunumda Nasıl Gösterilir

Bu adım için iki görsel yeterlidir:

```
Görsel 1: Violence videosundan örnek sekansların motion skoru dağılımı
          → Histogram: düşük skorlu (sakin) ve yüksek skorlu (aktif) bölgeler görülür
          → θ çizgisi ile hangi sekansların elendiği işaretlenir

Görsel 2: Filtreleme öncesi / sonrası confusion matrix karşılaştırması
          → False Positive sayısının düştüğü sayısal olarak gösterilir
```

---

## 10. GRU Model Mimarisi

### Neden GRU, LSTM Değil?

GRU, LSTM'e göre daha az parametre kullanır (kapı sayısı: 2 vs 3).  
Bu dataset boyutunda GRU genellikle LSTM ile karşılaştırılabilir performans verir.  
Eğitim hızı daha yüksektir.

### Mimari

```
Input Layer:     (batch_size, 30, 69)
                          │
GRU Layer 1:     128 unit · return_sequences=True · bidirectional=False
                          │
Dropout:         rate=0.3
                          │
GRU Layer 2:     64 unit · return_sequences=False
                          │
Dropout:         rate=0.3
                          │
Dense Layer:     32 unit · aktivasyon=ReLU
                          │
Output Layer:    1 unit · aktivasyon=Sigmoid  →  P(Violence) ∈ [0, 1]
```

### Parametre Sayısı (Tahmini)

```
GRU Layer 1:  3 × (69 + 128) × 128 ≈ 75,776 parametre
GRU Layer 2:  3 × (128 + 64) × 64  ≈ 36,864 parametre
Dense:        32 × 64 + 32          ≈  2,080 parametre
Output:       1 × 32 + 1            ≈     33 parametre
Toplam:       ~115,000 parametre
```

Hafif model — GPU gerektirmez, CPU inference mümkün.

### Loss ve Optimizer

```python
loss      = nn.BCELoss()
optimizer = torch.optim.Adam(lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                patience=5, factor=0.5, monitor='val_loss')
```

---

## 11. Eğitim Stratejisi

### Parametreler

| Parametre | Değer |
|---|---|
| Max Epoch | 100 |
| Batch Size | 32 |
| Early Stopping Patience | 10 epoch |
| Early Stopping Monitor | val_loss |
| Model Kayıt Kriteri | En düşük val_loss |

### Eğitim Döngüsü

```
Her epoch:
  1. Train batches → forward pass → BCELoss → backward → optimizer step
  2. Validation batches → forward pass (no_grad) → val_loss hesapla
  3. ReduceLROnPlateau → gerekirse lr düşür
  4. Early stopping kontrolü
  5. Eğer val_loss en düşükse → best_model.pt kaydet
```

### Overfitting Belirtileri

```
train_loss sürekli düşüyor AMA val_loss duruyorsa veya artıyorsa → overfitting

Çözüm sırası:
1. Early stopping zaten devrede
2. Dropout rate'i 0.3 → 0.4-0.5 artır
3. weight_decay 1e-4 → 1e-3 artır
4. Batch size artır (32 → 64)
```

---

## 12. Değerlendirme Metrikleri ve Ablation

### Temel Metrikler

Test seti üzerinde hesaplanır, yalnızca eğitim bittikten sonra açılır.

```
1. Confusion Matrix          — TP, FP, FN, TN görsel dağılım
2. Precision (her sınıf)     — Tespit ettiklerimin kaçı doğru?
3. Recall (her sınıf)        — Gerçek olayların kaçını yakaladık?
4. F1 Score (her sınıf)      — Precision ve Recall harmonik ortalaması
5. AUC-ROC                   — Threshold bağımsız model kalitesi
```

### Ablation 1 — Threshold Analizi

Threshold 0.7 başlangıç değeridir, sabitlenemez.

```
Threshold   Precision   Recall   F1   Yorum
  0.50        ...         ...      ...   Recall yüksek, FP fazla
  0.60        ...         ...      ...
  0.70        ...         ...      ...   ← Başlangıç noktamız
  0.80        ...         ...      ...
  0.90        ...         ...      ...   Precision yüksek, FN fazla
```

Gerçek değerler eğitimden sonra doldurulur. Sunumda bu tablo güçlü bir slayttır.  
Kullanım senaryosuna göre optimal threshold seçilir:
- Güvenlik sistemi → Recall önemli (kaçırma kabul edilemez) → düşük threshold
- Düşük FP istenen ortam → Precision önemli → yüksek threshold

### Ablation 2 — Interaction Feature Katkısı

```
Model A: 68 özellik (34+34, mesafe yok) → eğit → F1 = ?
Model B: 69 özellik (34+34+1, mesafe var) → eğit → F1 = ?
```

İki modelin karşılaştırması, interaction feature'ının sayısal katkısını kanıtlar.  
Bu ablation sunumda metodolojik güvenilirlik göstergesidir.

### Ablation 3 — Normalizasyon Katkısı

```
Model A: Raw koordinatlar (normalizasyon yok) → F1 = ?
Model B: Hip-centered + scaled koordinatlar → F1 = ?
```

Domain shift etkisini sayısal olarak gösterir.

### Ablation 4 — Motion Filter Threshold (θ) Analizi

Motion filtering'in hem etkisini hem de doğru θ değerini sayısal olarak belirler.

```
θ Değeri   Elinen Violence Sekans %   Val F1   Val FP Sayısı   Yorum
  0.00      %0  (filtresiz)             ...      ...             Baseline
  0.03      %~10                        ...      ...
  0.05      %~25           ← Başlangıç noktamız
  0.08      %~40                        ...      ...
  0.12      %~60                        ...      ...             Çok agresif?
```

Gerçek değerler preprocessing sonrası doldurulur.  
En yüksek Val F1 veren θ final model için seçilir.  
Sunumda Ablation 4 ve Ablation 1 (threshold) birlikte gösterilirse "sistem kalibrasyonu" slaytı oluşur.

---

## 13. Gerçek Zamanlı Inference Pipeline

### Akış

```
Kaynak: Video dosyası VEYA webcam (cv2.VideoCapture)
         │
         ▼
┌─────────────────────────────────┐
│        FRAME DÖNGÜSÜ           │
│                                 │
│  1. Frame oku                   │
│  2. FPS kontrolü (her N. kare) │
│  3. CLAHE (koşullu)            │
│  4. GaussianBlur 3×3           │
│  5. Resize 640×640             │
│  6. BGR → RGB                  │
│  7. YOLOv8n-pose çalıştır      │
│  8. Multi-person B+ stratejisi │
│  9. Koordinat normalizasyonu   │
│ 10. 69-boyutlu vektör oluştur  │
│ 11. FIFO Buffer'a ekle         │
│                                 │
│  Buffer doldu mu? (30 frame)   │
│  ├─ Hayır → görselleştir,      │
│  │          "Analiz ediliyor"  │
│  └─ Evet  → GRU'ya ver        │
│                                 │
│  GRU → olasılık skoru P        │
│                                 │
│  P > 0.7?                      │
│  ├─ Evet → Kırmızı iskelet     │
│  │         "ŞİDDET TESPİT"    │
│  │         Bounding Box kırmızı│
│  └─ Hayır → Yeşil iskelet      │
│             "NORMAL"           │
│                                 │
│  Frame'i ekranda göster        │
│  (OpenCV ile çizim yapılmış)   │
└─────────────────────────────────┘
```

### FIFO Buffer

```python
from collections import deque
buffer = deque(maxlen=30)

# Her frame'de:
buffer.append(feature_vector_69)

# Tahmin:
if len(buffer) == 30:
    sequence = np.array(buffer)           # (30, 69)
    tensor   = torch.FloatTensor(sequence).unsqueeze(0)  # (1, 30, 69)
    prob     = model(tensor).item()
```

`deque(maxlen=30)` otomatik FIFO yönetimi sağlar. 31. eleman eklenince ilk eleman düşer.

### Görselleştirme Katmanı

```
Her frame üzerine çizilecekler:

1. İskelet çizgileri
   → P > 0.7: RGB(220, 50, 50)   — Kırmızı
   → P ≤ 0.7: RGB(50, 200, 100)  — Yeşil

2. Keypoint noktaları (her eklem için daire)

3. Olasılık skoru (sol üst köşe)
   → "Violence: 0.87"

4. Karar etiketi (sağ üst köşe)
   → "⚠ VIOLENCE DETECTED" veya "✓ NORMAL"

5. Kişi bounding box'ları (sadece seçilen 2 kişi)
```

---

## 14. Bilinen Kısıtlamalar

Bu kısıtlamalar çözülmemiş bug değildir. Sistemin bilinçli olarak kabul ettiği sınırlar ve trade-off'lardır.

**Kısıt 1 — X-Sorting Kararsızlığı:**  
Kavganın doruk anında iki kişinin bounding box'ları örtüşebilir. Bu anda X merkezleri birbirine çok yaklaşır, sıralama frame to frame değişebilir. Gerçek çözüm `model.track()` ile ByteTrack entegrasyonudur. Bu proje kapsamında X-sorting kabul edilebilir bir yaklaşımdır.

**Kısıt 2 — Video-Level Etiket Gürültüsü (Kısmen Çözüldü):**  
Dataset'te frame-level etiket mevcut değil. Violence videolarının sakin başlangıç bölümleri yanlış etiket taşıyabilir. Bu sorun Aşama 9B'deki motion-based filtering ile büyük ölçüde otomatik olarak giderilir. Ancak filtering bir tahmindir — gerçek kavga başlangıç zamanını kesin olarak belirleyemez. Düşük hareketle başlayan (örn. ani tek yumruk) bazı Violence sekansları da filtreye takılabilir. Bu trade-off bilinçli olarak kabul edilmiştir.

**Kısıt 3 — Sıfır Kişi Frame'leri:**  
YOLOv8 hiç kişi bulamazsa (boş sahne, çok karanlık) buffer'a sıfır vektörü eklenir. 30 frame'in büyük çoğunluğu sıfırsa GRU anlamsız tahmin üretebilir. Bu durum gerçek dünyada nadir ama mümkündür.

**Kısıt 4 — Tek Kaynaklı Dataset:**  
Tüm veriler YouTube'dan gelir. Gerçek güvenlik kamerası dağıtımında (düşük çözünürlük, tepeden çekim, balık gözü lens) performans düşebilir. Ek data augmentation veya fine-tuning gerekebilir.

---

## 15. Karar Günlüğü

Proje boyunca alınan mimari kararlar ve gerekçeleri.

| # | Karar | Seçilen | Reddedilen | Gerekçe |
|---|---|---|---|---|
| 1 | Target FPS | 10 | 30, 5 | Yeterli temporal çözünürlük + işlem verimliliği |
| 2 | Preprocessing agresifliği | CLAHE + Gaussian-3×3 | Bilateral | Real-time/eğitim tutarlılığı, hız |
| 3 | Pose modeli | YOLOv8n-pose | YOLOv8s-pose | Real-time FPS gereksinimi |
| 4 | Multi-person stratejisi | B+ (2 kişi + mesafe) | A (tek kişi), C (joint mesafe) | Etkileşim bilgisi + sağlamlık dengesi |
| 5 | Sıralama yöntemi | X-axis sorting | Tracking | Proje kapsamı, ByteTrack overkill |
| 6 | Interaction feature | Normalize bbox mesafesi | Joint-specific mesafe | Keypoint-0 sağlamlığı |
| 7 | Sekans penceresi | 30 frame | 15, 60 | 3 sn kavga anını kapsar, bellek dengeli |
| 8 | Model ağırlığı | GRU | LSTM | Daha az parametre, benzer performans |
| 9 | Eğitim split | 70/15/15 stratified | 80/20 | Test seti izolasyonu, sınıf dengesi |
| 10 | Başlangıç threshold | 0.7 | 0.5 | Ablation ile optimize edilecek |
| 11 | Video-level gürültü çözümü | Motion-based filtering (θ=0.05) | Manuel etiketleme, MIL | Otomatik, ekstra araç gerektirmez, pipeline verisi üzerinde çalışır |

---

*Son güncelleme: Motion-based sequence filtering eklendi (Aşama 9B). Tüm mimari kararlar kilitlendi.*