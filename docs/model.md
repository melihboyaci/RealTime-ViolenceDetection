# MODEL.md
# GRU Mimarisi, Hiperparametreler ve Eğitim Kılavuzu

**Proje:** Real-Time Violence Detection via Skeletal Pose Analysis  
**Model Tipi:** Gated Recurrent Unit (GRU) — Binary Sequence Classifier  
**Framework:** PyTorch  
**Versiyon:** 1.0.0

---

## İçindekiler

1. [Neden GRU?](#1-neden-gru)
2. [Mimari Tasarım Kararları](#2-mimari-tasarım-kararları)
3. [Katman Detayları ve Gerekçeleri](#3-katman-detayları-ve-gerekçeleri)
4. [Loss Fonksiyonu ve Optimizer](#4-loss-fonksiyonu-ve-optimizer)
5. [Hiperparametre Tablosu](#5-hiperparametre-tablosu)
6. [Eğitim Döngüsü](#6-eğitim-döngüsü)
7. [Eğitim Davranışı — Ne Görmeliyiz?](#7-eğitim-davranışı--ne-görmeliyiz)
8. [Sorun Giderme Kılavuzu](#8-sorun-giderme-kılavuzu)
9. [Model Kayıt ve Yükleme](#9-model-kayıt-ve-yükleme)
10. [Model Kısıtlamaları](#10-model-kısıtlamaları)
11. [Model-Side Ablation Noktaları](#11-model-side-ablation-noktaları)
12. [Karar Günlüğü](#12-karar-günlüğü)

---

## 1. Neden GRU?

### Problem Tanımı

Modelimizin girdisi tek bir frame değil, **30 frame'lik bir zaman serisidir.**  
`(30, 69)` → 3 saniyelik iskelet hareketi dizisi.

Kavgayı tanımlamak için tek bir kare yeterli değildir:
- Frame 15: adam kolunu kaldırıyor → boks antrenmanı mı? kavga mı?
- Frame 14–16: ani hız + kısa mesafe + öne eğilme → kavga örüntüsü

**Bu temporal bağımlılığı öğrenmek için tasarlanmış mimariler:**

| Mimari | Temporal Hafıza | Parametre | Eğitim Hızı |
|---|---|---|---|
| Vanilla RNN | Var (zayıf) | Az | Hızlı |
| LSTM | Var (güçlü) | Fazla | Yavaş |
| **GRU** | **Var (güçlü)** | **Orta** | **Hızlı** |
| Transformer | Var (global) | Çok fazla | Çok yavaş |
| 3D CNN | Var (lokal) | Fazla | Yavaş |

### GRU vs LSTM

GRU, LSTM'in sadeleştirilmiş versiyonudur.

```
LSTM kapıları: Input Gate · Forget Gate · Output Gate · Cell State  (4 operasyon)
GRU  kapıları: Update Gate · Reset Gate                             (2 operasyon)
```

GRU parametresi ≈ LSTM parametresinin %75'i.  
Bu dataset boyutunda (~10,000 sekans) GRU genellikle LSTM ile eşdeğer veya daha iyi performans verir.  
Daha az parametre → daha az overfitting riski → daha hızlı eğitim.

**Vanilla RNN neden değil?**  
Vanishing gradient problemi — 30 frame'lik sekanslarda ilk frame'lerin bilgisi son frame'e taşınmaz.  
GRU'nun update gate mekanizması bu problemi çözer.

**Transformer neden değil?**  
10,000 sekans veriyle Transformer eğitmek ciddi overfitting riski taşır.  
Transformer mimarisi büyük veri setlerinde (100k+ örnek) avantajlıdır.  
Bu proje kapsamında overkill.

---

## 2. Mimari Tasarım Kararları

### Girdi Boyutu

```
(batch_size, sequence_length, feature_size)
      B           30               69
```

`69` özelliğin dökümü:
```
Kişi 1 skeleton:  17 keypoint × 2 koordinat = 34
Kişi 2 skeleton:  17 keypoint × 2 koordinat = 34
Interaction:      normalize bbox mesafesi   =  1
Toplam:                                       69
```

### Mimari Şeması

```
                    ┌─────────────────────────────────┐
INPUT               │  (batch, 30, 69)                │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
GRU LAYER 1         │  GRU(input=69, hidden=128)      │
                    │  return_sequences = True         │
                    │  Çıktı: (batch, 30, 128)         │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
DROPOUT 1           │  Dropout(p=0.3)                 │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
GRU LAYER 2         │  GRU(input=128, hidden=64)      │
                    │  return_sequences = False        │
                    │  Çıktı: (batch, 64)              │
                    │  [yalnızca son adımın çıktısı]   │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
DROPOUT 2           │  Dropout(p=0.3)                 │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
DENSE               │  Linear(64 → 32) + ReLU         │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
OUTPUT              │  Linear(32 → 1) + Sigmoid        │
                    │  P(Violence) ∈ [0.0, 1.0]        │
                    └─────────────────────────────────┘
```

---

## 3. Katman Detayları ve Gerekçeleri

### GRU Layer 1 — 128 Unit, return_sequences=True

**Neden 128 unit?**  
69 boyutlu giriş için 128 unit — yaklaşık 2× büyütme.  
Bu, GRU'nun giriş uzayını temsil etmesi için yeterli kapasite sağlar.  
64 ile denenebilir ama temporal örüntüler için bilgi sıkışabilir.  
256 overkill — bu veri boyutu için overfitting riski artar.

**Neden return_sequences=True?**  
Bu katman bir sonraki GRU katmanına veri verecek.  
İkinci GRU katmanının her adımda bilgiye ihtiyacı var.  
Eğer `False` olsaydı yalnızca son adım çıkarı geçerdi — bilgi kaybı.

### GRU Layer 2 — 64 Unit, return_sequences=False

**Neden 64 unit?**  
128'den 64'e daraltma — bilgi sıkıştırma (bottleneck).  
GRU'nun öğrendiği temporal örüntüyü özet temsile indirger.  
Aynı zamanda parametre sayısını azaltır.

**Neden return_sequences=False?**  
Bu son GRU katmanı. Sınıflandırma için tek bir vektöre ihtiyacımız var.  
`False` → sadece 30. adımın gizli durumu çıkar: `(batch, 64)`.  
Bu vektör, 3 saniyelik tüm hareket dizisinin özetini taşır.

### Dropout — p=0.3

Her iki GRU katmanından sonra uygulanır.

**Neden 0.3?**  
0.2 — genellikle yeterince düzenleyici değil.  
0.5 — agresif, model az öğrenebilir.  
0.3 — literatürde zaman serisi modelleri için standart başlangıç değeri.

Eğitimde `model.train()` → dropout aktif.  
Inference'da `model.eval()` → dropout devre dışı.  
Bu PyTorch'ta otomatik yönetilir — manuel kapatmaya gerek yok.

### Dense Layer — Linear(64→32) + ReLU

GRU çıktısını (64 boyut) sınıflandırma kararına hazırlar.  
ReLU negatif aktivasyonları sıfırlar — özellik seçimi etkisi yaratır.  
32 unit — yeterli kapasite, gereksiz büyüklük değil.

### Output Layer — Linear(32→1) + Sigmoid

**Neden Sigmoid?**  
Binary classification için standart.  
Çıktıyı `[0, 1]` aralığına sıkıştırır.  
`P > threshold` → Violence kararı.

**Neden Softmax değil?**  
Softmax çok-sınıflı problemler içindir (3+ sınıf).  
2 sınıf → Sigmoid yeterli ve daha verimli.

---

## 4. Loss Fonksiyonu ve Optimizer

### Binary Cross Entropy Loss

```python
criterion = nn.BCELoss()
```

**Formül:**
```
BCELoss = -[ y·log(p) + (1-y)·log(1-p) ]

y = gerçek etiket (0 veya 1)
p = model tahmini (Sigmoid çıktısı, 0-1 arası)
```

**Neden BCE?**  
Binary classification için matematiksel olarak doğru loss fonksiyonu.  
Model çok emin yanlış tahmin yaparsa (örn. Violence=0.99 ama gerçek=0) loss çok yüksek — sert ceza.  
Model az emin tahmin yaparsa (örn. Violence=0.6 ama gerçek=0) loss düşük — hafif ceza.  
Bu davranış modeli doğru örneklere yüksek confidence üretmeye zorlar.

### Adam Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

**Neden Adam?**  
SGD'ye göre adaptif öğrenme oranı kullanır — her parametre için ayrı lr.  
GRU gibi recurrent mimarilerde gradyan büyüklükleri değişkendir, Adam bunu dengeler.  
`lr=1e-3` — Adam için standart başlangıç değeri.

**weight_decay=1e-4 (L2 Regularization):**  
Parametre büyüklüklerini küçük tutar.  
Overfitting'e karşı ek savunma katmanı.  
Dropout ile birlikte çalışır, ikisi birbirini destekler.

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5,
    min_lr=1e-6
)
```

**Nasıl çalışır?**  
5 epoch boyunca val_loss iyileşmezse → lr'yi yarıya indir.  
`lr: 1e-3 → 5e-4 → 2.5e-4 → ...` şeklinde azalır.  
`min_lr=1e-6` — bu değerin altına inmez.

**Neden gerekli?**  
Yüksek lr ile başlamak hızlı öğrenme sağlar.  
Loss platoya ulaştığında lr düşürülmesi ince ayar yapmaya izin verir.  
Sabit düşük lr'ye göre genellikle daha iyi final performansı.

---

## 5. Hiperparametre Tablosu

| Hiperparametre | Değer | Aralık | Değiştirme Koşulu |
|---|---|---|---|
| GRU Layer 1 units | 128 | 64–256 | Val F1 < 0.75 ve model küçük görünüyorsa artır |
| GRU Layer 2 units | 64 | 32–128 | Layer 1 ile birlikte ölçeklendir |
| Dropout rate | 0.3 | 0.2–0.5 | Overfitting varsa artır |
| Batch size | 32 | 16–128 | GPU belleği yetersizse azalt |
| Learning rate | 1e-3 | 1e-4–1e-2 | Değiştirme — scheduler halleder |
| weight_decay | 1e-4 | 1e-5–1e-3 | Overfitting devam ederse artır |
| Scheduler patience | 5 | 3–10 | Eğitim çok uzunsa azalt |
| Scheduler factor | 0.5 | 0.3–0.7 | — |
| Max epoch | 100 | — | Early stopping zaten keser |
| Early stop patience | 10 | 7–15 | Eğitim çok erken bitiyorsa artır |
| Sequence length | 30 | — | PIPELINE.md kararı, burada değiştirme |
| Feature size | 69 | — | PIPELINE.md kararı, burada değiştirme |

> **Kural:** Hiperparametreleri tek tek değiştir. Aynı anda birden fazla değişken değiştirirsen hangi değişkenin etkili olduğunu bilemezsin.

---

## 6. Eğitim Döngüsü

### Sözde Kod

```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):

    # ── TRAIN ──────────────────────────────────────────
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)          # (batch, 1)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # ── VALIDATION ─────────────────────────────────────
    model.eval()
    val_losses = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_losses.append(loss.item())

    train_loss = mean(train_losses)
    val_loss   = mean(val_losses)

    # ── SCHEDULER ──────────────────────────────────────
    scheduler.step(val_loss)

    # ── MODEL KAYIT ────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    # ── EARLY STOPPING ─────────────────────────────────
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Gradient Clipping — Neden Şart?

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

GRU'larda zaman zaman **exploding gradient** problemi yaşanır.  
Gradyan normu 1.0'ı aşarsa ölçeklenir — parametre güncellemeleri patlamaz.  
Bu satır olmadan eğitim loss'u aniden `NaN`'a gidebilir.  
Optimizer step'inden **önce** çağrılması zorunludur.

---

## 7. Eğitim Davranışı — Ne Görmeliyiz?

### Sağlıklı Eğitim

```
Epoch  Train Loss  Val Loss  Val Acc  Yorum
  1     0.68        0.65      0.61    Başlangıç — rastgele tahmine yakın
  5     0.52        0.50      0.74    Model örüntü öğreniyor
 15     0.35        0.37      0.83    İyi ilerleme
 30     0.22        0.26      0.89    Olgunlaşıyor
 45     0.18        0.24      0.91    Scheduler devreye girmiş olabilir
 60     0.15        0.23      0.92    Yakınsama
```

**İşaretler:**
- Train loss her epoch düşüyor ✓
- Val loss train loss'u yakından takip ediyor ✓  
- Val acc 0.85+ ulaşıyor ✓
- Loss'lar `NaN` değil ✓

### Loss Eğrisi Yorumlama

```
         Loss
          │
   0.70 ──┤ ╲  ← Hızlı düşüş (model temel örüntüyü öğreniyor)
          │  ╲
   0.40 ──┤   ╲╲
          │     ╲╲  ← Yavaşlayan düşüş (ince ayar)
   0.25 ──┤      ╲─────
          │           ──── ← Yakınsama (plato)
   0.20 ──┤
          └────────────────── Epoch
               10   30   50
```

Train ve val eğrilerinin birbirine yakın seyretmesi hedef.  
Aralarındaki fark büyürse overfitting başlıyor demektir.

---

## 8. Sorun Giderme Kılavuzu

### Sorun 1 — Loss NaN Oluyor

```
Belirtiler: Epoch 3-5'te loss aniden NaN
Neden:      Exploding gradient

Kontrol:
  1. Gradient clipping var mı? → clip_grad_norm_ eklendi mi?
  2. Learning rate çok yüksek mi? → 1e-3'ten 1e-4'e düşür
  3. Veri normalizasyonu yapıldı mı? → .npy dosyalarındaki değerler makul aralıkta mı?
```

### Sorun 2 — Model Hep 0.5 Tahmin Ediyor

```
Belirtiler: Tüm tahminler ~0.5, acc %50 civarı
Neden:      Model hiç öğrenmiyor

Kontrol:
  1. Etiketler doğru yüklendi mi? → y_batch sıfır mı, bir mi içeriyor?
  2. Sınıf dengesi bozuk mu? → train setinde Violence/NonViolence oranı?
  3. Özellik vektörleri sıfır mı? → .npy dosyaları gerçek veri içeriyor mu?
  4. lr çok küçük mü? → 1e-5 gibi değerlerde model öğrenemez
```

### Sorun 3 — Overfitting

```
Belirtiler: train_loss düşüyor, val_loss düşmüyor veya artıyor

Çözüm sırası (tek tek uygula):
  1. Early stopping zaten var — 10 epoch bekle, kapatmıyor mu?
  2. Dropout: 0.3 → 0.4
  3. weight_decay: 1e-4 → 5e-4
  4. GRU unit sayısı: 128/64 → 64/32
  5. Batch size: 32 → 64 (daha gürültülü gradient, regularization etkisi)
```

### Sorun 4 — Çok Yüksek False Positive (Normal → Violence)

```
Belirtiler: Precision düşük, sakin sahneler Violence tetikliyor

Neden olabilir:
  A. Motion filtering θ değeri çok küçük → sakin Violence sekansları eğitime girdi
  B. Threshold 0.7 çok düşük → 0.8'e çıkar

Çözüm:
  A: θ değerini 0.05 → 0.08 artır, modeli yeniden eğit
  B: Threshold ablation tablosuna bak, Precision/Recall trade-off'unu değerlendir
```

### Sorun 5 — Çok Yüksek False Negative (Violence Kaçırılıyor)

```
Belirtiler: Recall düşük, gerçek kavgalar tespit edilemiyor

Neden olabilir:
  A. Motion filtering θ çok büyük → gerçek Violence sekansları elendi
  B. Threshold 0.7 çok yüksek → model emin olmadan karar veremiyor

Çözüm:
  A: θ değerini 0.05 → 0.03'e düşür, modeli yeniden eğit
  B: Threshold 0.7 → 0.6'ya düşür
```

---

## 9. Model Kayıt ve Yükleme

### Kayıt (Eğitim Sırasında)

```python
# Yalnızca ağırlıkları kaydet (önerilen)
torch.save(model.state_dict(), 'best_model.pt')

# Tüm modeli kaydet (daha büyük dosya, taşınabilirlik sorunu olabilir)
torch.save(model, 'best_model_full.pt')
```

`state_dict()` tercih edilir — sadece öğrenilmiş ağırlıkları içerir, mimari bağımlılığı yoktur.

### Yükleme (Inference İçin)

```python
# Modeli tanımla (PIPELINE.md'deki aynı mimari)
model = ViolenceGRU(input_size=69, hidden1=128, hidden2=64)

# Ağırlıkları yükle
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))

# Inference moduna al (Dropout kapanır)
model.eval()
```

`map_location='cpu'` → GPU'da eğitilen model CPU'da çalıştırılabilir.  
Demo makinesinde GPU olmasa bile inference çalışır.

### Checkpoint Yapısı

```
models/
├── best_model.pt          → En iyi val_loss'a sahip model (final kullanım)
├── checkpoint_epoch30.pt  → Ara kayıt (isteğe bağlı)
└── training_log.csv       → Epoch, train_loss, val_loss, val_acc (her epoch)
```

`training_log.csv` eğitim eğrisi grafiği için kullanılır — sunumda gösterilir.

---

## 10. Model Kısıtlamaları

Bu kısıtlamalar PIPELINE.md'deki sistem-düzey kısıtlamalardan ayrıdır. Bunlar doğrudan **model mimarisinin** davranışsal sınırlarıdır.

**Kısıt 1 — Pose-Only Girdi:**  
Model yalnızca iskelet koordinatlarını görür. YOLOv8n-Pose'un küçük, örtüşmüş veya karanlık sahnede kaçırdığı kişilerden gelen bilgi tamamen kaybolur. Bu durumda model eksik veriyle karar vermek zorunda kalır.

**Kısıt 2 — Sabit Pencere Uzunluğu (30 frame):**  
~3 saniyeden kısa kavgalar (ani tek yumruk) pencere içinde seyreltilir, tespit zorlaşır. ~3 saniyeden uzun kavgalar art arda gelen pencere kararlarının tutarlılığına bağlıdır — model pencereler arası hafıza taşımaz.

**Kısıt 3 — İki Kişi Üst Sınırı:**  
Sahnede ikiden fazla aktif katılımcı varsa yalnızca en büyük iki bounding box alınır. Gruptaki diğer kişilerin hareketi modele hiç girmez.

**Kısıt 4 — Pencereler Arası Hafıza Yok:**  
Eğitim sırasında model her pencereyi bağımsız görür. Online inference'da FIFO buffer dolduğunda GRU'nun iç durumu sıfırlanmaz — bu tutarlı görünse de model, önceki pencerelerdeki olaylardan bilinçli olarak öğrenmez.

**Kısıt 5 — X-Axis Sort Hassasiyeti:**  
Kavganın doruk anında iki kişi çakışabilir. Bu anda X koordinatlarına göre sıralama kararsızlaşır ve GRU, kişi kimliklerinin anlık yer değiştirdiğini sanabilir. Model bunu "ani hareket" olarak yorumlayabilir.

---

## 11. Model-Side Ablation Noktaları

Bu ablation'lar doğrudan model mimarisine aittir. `evaluation.md`'deki pipeline-level ablation'lardan (threshold, normalizasyon, motion filter, interaction feature) bağımsız tutulur.

| # | Ablation | Değişen | Sabit Kalan | Amaç |
|---|---|---|---|---|
| M1 | Hidden size sweep | GRU hidden: 64 / **128** / 256 | Diğer her şey | Kapasite vs. real-time budget dengesi |
| M2 | Depth sweep | num_layers: 1 / **2** / 3 | Diğer her şey | Ek katmanın marginal katkısı |
| M3 | GRU vs LSTM | Backbone: **GRU** / LSTM | Diğer her şey | Kilitli GRU kararını sayısal olarak doğrula |
| M4 | Aggregation | last-step / mean / attention | Diğer her şey | 30-frame özetleme stratejisinin etkisi |
| M5 | Dropout sweep | p: 0.0 / 0.2 / **0.3** / 0.5 | Diğer her şey | Regularization hassasiyeti |

**Kalın** değerler kilitli baseline değerlerdir.

> Herhangi bir ablation kilitli baseline'ı geçerse → `decision_log.md` güncellenir ve yeni değer baseline olur. Ablation sırası önemlidir: önce M1 ve M2'yi çalıştır (kapasite), sonra M3 (backbone kararını doğrula), sonra M4-M5.

---

## 12. Karar Günlüğü

| # | Karar | Seçilen | Reddedilen | Gerekçe |
|---|---|---|---|---|
| 1 | Temporal model | GRU | LSTM, Transformer, 3D CNN | Parametre verimliliği, veri boyutuna uygunluk |
| 2 | GRU derinliği | 2 katman | 1, 3 katman | 1 katman yetersiz temsil, 3 katman overfitting riski |
| 3 | Layer 1 unit | 128 | 64, 256 | 2× büyütme — yeterli kapasite, aşırı değil |
| 4 | Layer 2 unit | 64 | 32, 128 | Bottleneck için 128'den yarıya indirme |
| 5 | Dropout rate | 0.3 | 0.2, 0.5 | Zaman serisi literatüründe standart başlangıç |
| 6 | Loss fonksiyonu | BCELoss | CrossEntropyLoss | Binary problem — CE ikili sınıf için indirgenmiş BCE'dir |
| 7 | Optimizer | Adam | SGD, RMSProp | Adaptif lr, recurrent mimarilerde kararlı |
| 8 | Gradient clipping | max_norm=1.0 | Yok | GRU exploding gradient koruması |
| 9 | Aktivasyon (dense) | ReLU | Tanh, LeakyReLU | Sade, etkili, hesaplama verimli |
| 10 | Çıktı aktivasyon | Sigmoid | Softmax | Binary problem için matematiksel doğru seçim |

---

*Son güncelleme: Model kısıtlamaları (Bölüm 10) ve model-side ablation noktaları (Bölüm 11) repodaki model.md'den entegre edildi. Tüm TBD'ler kapatıldı. Bir sonraki belge: `DATA.md`*