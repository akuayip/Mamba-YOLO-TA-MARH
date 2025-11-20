# Panduan Training from Scratch Mamba-YOLO untuk Object Detection

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Perbedaan Training from Scratch vs Fine-tuning](#perbedaan-training-from-scratch-vs-fine-tuning)
- [Persiapan Sebelum Training](#persiapan-sebelum-training)
- [Langkah-Langkah Training from Scratch](#langkah-langkah-training-from-scratch)
- [Monitoring Training Process](#monitoring-training-process)
- [Evaluasi Hasil Training](#evaluasi-hasil-training)
- [Optimasi dan Tips](#optimasi-dan-tips)
- [Troubleshooting](#troubleshooting)

---

## Pendahuluan

**Training from Scratch** adalah proses melatih model object detection Mamba-YOLO **dari nol** (tanpa menggunakan pre-trained weights). Model akan mempelajari semua fitur dan pattern dari awal hanya berdasarkan dataset yang Anda sediakan.

### 🤔 Kapan Harus Training from Scratch?

**Gunakan Training from Scratch jika:**

- ✅ Dataset Anda **sangat besar** (lebih dari 10,000 gambar)
- ✅ Domain aplikasi **sangat berbeda** dari COCO dataset (misal: medical imaging, satellite imagery, microscopy)
- ✅ Anda punya **resource komputasi yang kuat** (multiple GPUs)
- ✅ Ingin penelitian akademis tentang arsitektur model

**JANGAN gunakan Training from Scratch jika:**

- ❌ Dataset kecil (kurang dari 1,000 gambar) → Gunakan Fine-tuning
- ❌ Resource terbatas (1 GPU atau CPU only)
- ❌ Aplikasi umum (deteksi mobil, orang, hewan, dll) → Gunakan Fine-tuning

---

## Perbedaan Training from Scratch vs Fine-tuning

| Aspek               | Training from Scratch                | Fine-tuning                         |
| ------------------- | ------------------------------------ | ----------------------------------- |
| **Starting Point**  | Weights acak (random initialization) | Pre-trained weights dari COCO       |
| **Dataset Minimal** | 10,000+ gambar                       | 100-1,000 gambar                    |
| **Waktu Training**  | Sangat lama (days-weeks)             | Lebih cepat (hours-days)            |
| **Resource**        | Multi-GPU, memori besar              | 1 GPU cukup                         |
| **Epochs**          | 300-500 epochs                       | 50-100 epochs                       |
| **Akurasi Awal**    | Sangat rendah (random)               | Sudah lumayan dari awal             |
| **Use Case**        | Domain sangat berbeda                | Aplikasi umum dengan custom classes |

### Ilustrasi Sederhana:

**Training from Scratch:**

```
Bayi baru lahir → belajar dari NOL → jadi ahli
(Random weights) → (Training lama) → (Model bagus)
```

**Fine-tuning:**

```
Anak SD pintar → spesialisasi → jadi ahli
(Pre-trained) → (Training singkat) → (Model bagus)
```

---

## Persiapan Sebelum Training

### 1. Requirement Sistem

**Minimum (Tidak Direkomendasikan):**

- GPU: 1x RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: 100GB SSD
- Dataset: 1,000+ gambar
- Waktu: 1-2 minggu

**Recommended:**

- GPU: 2-4x RTX 4090 atau A100 (24GB+ VRAM)
- RAM: 64GB+
- Storage: 500GB NVMe SSD
- Dataset: 10,000+ gambar
- Waktu: 3-7 hari

**Optimal:**

- GPU: 8x A100 (80GB VRAM)
- RAM: 256GB+
- Storage: 1TB NVMe SSD
- Dataset: 100,000+ gambar
- Waktu: 1-3 hari

---

### 2. Persiapan Dataset

Dataset untuk training from scratch harus **JAUH LEBIH BESAR** daripada fine-tuning.

#### 2.1 Struktur Folder Dataset

```
C:\DATA\my_large_dataset\
├── images\
│   ├── train\              ← 80% dari total (minimal 8,000 gambar)
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ... (ribuan gambar)
│   ├── val\                ← 15% dari total (minimal 1,500 gambar)
│   │   ├── img_val_0001.jpg
│   │   └── ... (ribuan gambar)
│   └── test\               ← 5% dari total (minimal 500 gambar) - OPSIONAL
│       ├── img_test_0001.jpg
│       └── ... (ratusan gambar)
├── labels\
│   ├── train\
│   │   ├── img_0001.txt
│   │   ├── img_0002.txt
│   │   └── ...
│   ├── val\
│   │   ├── img_val_0001.txt
│   │   └── ...
│   └── test\
│       ├── img_test_0001.txt
│       └── ...
└── dataset.yaml
```

#### 2.2 Format Label (YOLO Format)

Setiap file `.txt` berisi bounding box dalam format:

```
class_id x_center y_center width height
```

**Contoh: img_0001.txt**

```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
0 0.8 0.7 0.25 0.35
```

Nilai koordinat adalah **normalized** (0.0 - 1.0):

- `x_center`: Posisi tengah horizontal (0=kiri, 1=kanan)
- `y_center`: Posisi tengah vertikal (0=atas, 1=bawah)
- `width`: Lebar box (proporsi dari lebar gambar)
- `height`: Tinggi box (proporsi dari tinggi gambar)

#### 2.3 File dataset.yaml

```yaml
# File: dataset.yaml

# Path ke root dataset (bisa absolute atau relative)
path: C:\DATA\my_large_dataset

# Path relatif ke folder images
train: images\train
val: images\val
test: images\test # opsional

# Number of classes
nc: 3

# Class names
names:
  0: person
  1: car
  2: bicycle
```

#### 2.4 Tips Dataset Quality

**Kualitas > Kuantitas, tapi Training from Scratch butuh KEDUANYA!**

✅ **HARUS ADA:**

1. **Variasi Angle**: Foto dari berbagai sudut pandang
2. **Variasi Jarak**: Dekat, sedang, jauh
3. **Variasi Lighting**: Terang, gelap, backlight, indoor, outdoor
4. **Variasi Background**: Berbagai latar belakang
5. **Variasi Object Size**: Kecil, sedang, besar
6. **Variasi Occlusion**: Objek tertutupi sebagian
7. **Berbagai Kondisi Cuaca**: Cerah, mendung, hujan (kalau outdoor)
8. **Berbagai Waktu**: Siang, sore, malam

**Rasio Dataset Split:**

- Training: 80% (8,000+ gambar)
- Validation: 15% (1,500+ gambar)
- Test: 5% (500+ gambar) - opsional

---

## Langkah-Langkah Training from Scratch

### **LANGKAH 1: Install Environment** 💻

Sama seperti fine-tuning, pastikan environment sudah siap:

```powershell
# Masuk ke folder Mamba-YOLO
cd "C:\DATA\KULIYEAH\Tugas Akhir\Mamba-YOLO"

# Aktifkan environment
conda activate mambayolo

# Verifikasi instalasi
python -c "from ultralytics import YOLO; print('Ready for Training!')"
```

---

### **LANGKAH 2: Pilih Arsitektur Model** 📐

Untuk training from scratch, pilih arsitektur berdasarkan resource:

| Model            | Parameter | VRAM Needed | Batch Size | Training Time\* | Akurasi Final |
| ---------------- | --------- | ----------- | ---------- | --------------- | ------------- |
| **Mamba-YOLO-T** | 5.8M      | 8GB         | 64-128     | 3-5 hari        | ⭐⭐⭐        |
| **Mamba-YOLO-B** | 19.1M     | 16GB        | 32-64      | 5-10 hari       | ⭐⭐⭐⭐      |
| **Mamba-YOLO-L** | 57.6M     | 32GB+       | 16-32      | 10-20 hari      | ⭐⭐⭐⭐⭐    |

\*Dengan 10,000 gambar dan 1 GPU

**Rekomendasi:**

- **Pemula/Budget**: Mamba-YOLO-T
- **Balanced**: Mamba-YOLO-B (paling populer)
- **Research/High-end**: Mamba-YOLO-L

---

### **LANGKAH 3: Buat Script Training** 🎓

#### 3.1 Script Training Dasar

Buat file `train_from_scratch.py`:

```python
# File: train_from_scratch.py
from ultralytics import YOLO
import torch

# Cek GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 1. Load model architecture (HANYA CONFIG, TANPA WEIGHTS!)
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml')

# PENTING: Jangan load weights pre-trained!
# SALAH: model = YOLO('pretrained.pt') → ini adalah fine-tuning
# BENAR: model = YOLO('config.yaml') → ini training from scratch

# 2. Start training from scratch
results = model.train(
    # Dataset Configuration
    data='C:/DATA/my_large_dataset/dataset.yaml',

    # Training Duration
    epochs=300,             # Training from scratch butuh BANYAK epochs (300-500)

    # Image Settings
    imgsz=640,              # Image size (640 standard, bisa 1280 untuk detail lebih)

    # Batch Settings
    batch=32,               # Batch size - sesuaikan dengan VRAM
                           # RTX 3060 (12GB): batch=16
                           # RTX 4090 (24GB): batch=32-64
                           # A100 (80GB): batch=128

    # Device Settings
    device='0',            # Single GPU: '0', Multi-GPU: '0,1,2,3'
    workers=8,             # Data loading workers (sesuaikan dengan CPU cores)

    # Optimizer Settings
    optimizer='SGD',       # SGD (recommended) atau 'Adam', 'AdamW'
    lr0=0.01,              # Learning rate awal (PENTING untuk from scratch!)
    lrf=0.01,              # Learning rate final (0.01 = turun ke 1% dari lr0)
    momentum=0.937,        # Momentum untuk SGD
    weight_decay=0.0005,   # Weight decay untuk regularization

    # Augmentation (PENTING untuk from scratch!)
    hsv_h=0.015,           # Hue augmentation
    hsv_s=0.7,             # Saturation augmentation
    hsv_v=0.4,             # Value augmentation
    degrees=0.0,           # Rotation augmentation (±degrees)
    translate=0.1,         # Translation augmentation
    scale=0.5,             # Scale augmentation (±50%)
    shear=0.0,             # Shear augmentation
    perspective=0.0,       # Perspective augmentation
    flipud=0.0,            # Vertical flip probability
    fliplr=0.5,            # Horizontal flip probability (50%)
    mosaic=1.0,            # Mosaic augmentation probability
    mixup=0.0,             # Mixup augmentation probability
    copy_paste=0.0,        # Copy-paste augmentation probability

    # Training Settings
    patience=50,           # Early stopping patience (50 epochs tanpa improvement)
    save=True,             # Save checkpoints
    save_period=10,        # Save checkpoint setiap 10 epochs
    cache=False,           # Cache images ke RAM (True jika RAM besar)
    pretrained=False,      # PENTING: False untuk training from scratch!

    # Output Settings
    project='scratch_training',
    name='mambayolo_from_scratch_v1',
    exist_ok=True,

    # Mixed Precision Training (untuk speed up)
    amp=True,              # Automatic Mixed Precision (FP16)

    # Verbose
    verbose=True,
    plots=True,            # Generate plots
)

print("=" * 60)
print("✅ TRAINING SELESAI!")
print("=" * 60)
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
print(f"Last model saved at: {results.save_dir}/weights/last.pt")
print(f"View results at: {results.save_dir}/results.png")
```

#### 3.2 Script Training Multi-GPU (Advanced)

Untuk training lebih cepat dengan multiple GPUs:

```python
# File: train_from_scratch_multigpu.py
from ultralytics import YOLO
import torch

print(f"Available GPUs: {torch.cuda.device_count()}")

# Load model
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml')

# Training dengan Multiple GPUs
results = model.train(
    data='C:/DATA/my_large_dataset/dataset.yaml',
    epochs=300,
    imgsz=640,
    batch=128,              # Batch lebih besar karena multi-GPU
    device='0,1,2,3',       # Gunakan 4 GPUs
    workers=32,             # Lebih banyak workers

    # DDP (Distributed Data Parallel) Settings
    # Otomatis aktif ketika device > 1

    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    patience=50,
    save=True,
    save_period=10,
    pretrained=False,       # PENTING!

    project='scratch_training',
    name='mambayolo_multigpu_v1',
    exist_ok=True,
    amp=True,
    verbose=True,
)

print("Training complete!")
```

---

### **LANGKAH 4: Jalankan Training** 🚀

#### 4.1 Single GPU Training

```powershell
# Jalankan training
python train_from_scratch.py
```

#### 4.2 Multi-GPU Training

```powershell
# Multi-GPU dengan DDP (Distributed Data Parallel)
python train_from_scratch_multigpu.py
```

#### 4.3 Training dengan Resume (jika terputus)

```python
# File: resume_training.py
from ultralytics import YOLO

# Load dari checkpoint terakhir
model = YOLO('scratch_training/mambayolo_from_scratch_v1/weights/last.pt')

# Resume training
results = model.train(resume=True)
```

```powershell
python resume_training.py
```

---

## Monitoring Training Process

### 1. Real-time Monitoring

Training from scratch akan menampilkan output seperti ini:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/300      12.5G      1.234      2.345      1.567        150        640

      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       1500       3000      0.123      0.089      0.067      0.034
     person       1500       1200      0.145      0.102      0.078      0.042
        car       1500       1000      0.134      0.095      0.071      0.038
    bicycle       1500        800      0.089      0.067      0.052      0.022
```

**Penjelasan Metrics:**

| Metric          | Arti                                | Target             |
| --------------- | ----------------------------------- | ------------------ |
| `box_loss`      | Error lokasi bounding box           | Harus **turun** ⬇️ |
| `cls_loss`      | Error klasifikasi class             | Harus **turun** ⬇️ |
| `dfl_loss`      | Distribution focal loss             | Harus **turun** ⬇️ |
| `P` (Precision) | Berapa banyak deteksi yang benar    | Harus **naik** ⬆️  |
| `R` (Recall)    | Berapa banyak objek yang terdeteksi | Harus **naik** ⬆️  |
| `mAP50`         | Akurasi pada IoU 50%                | Harus **naik** ⬆️  |
| `mAP50-95`      | Akurasi rata-rata IoU 50-95%        | Harus **naik** ⬆️  |

### 2. Progress yang Normal vs Abnormal

#### ✅ Training NORMAL (from Scratch):

**Epoch 1-50:**

- Loss tinggi (2.0 - 5.0)
- mAP sangat rendah (< 0.1)
- Precision & Recall rendah (< 0.2)
- **INI NORMAL!** Model masih "bayi" yang belajar dari nol

**Epoch 50-150:**

- Loss mulai turun signifikan (1.0 - 2.0)
- mAP mulai naik (0.2 - 0.4)
- Precision & Recall membaik (0.3 - 0.5)
- **Progress terlihat jelas**

**Epoch 150-300:**

- Loss stabil rendah (< 1.0)
- mAP terus naik (0.5 - 0.7+)
- Precision & Recall bagus (0.6 - 0.8+)
- **Model sudah matang**

#### ❌ Training BERMASALAH:

**Tanda-tanda:**

- Loss tetap tinggi setelah 100 epochs
- Loss naik-turun ekstrem (tidak smooth)
- mAP tidak naik sama sekali setelah 100 epochs
- Loss menjadi NaN (Not a Number)
- GPU memory error terus-menerus

---

## Evaluasi Hasil Training

### 1. File Output Training

Setelah training selesai, folder output berisi:

```
scratch_training\
└── mambayolo_from_scratch_v1\
    ├── weights\
    │   ├── best.pt              ← Model dengan mAP terbaik
    │   ├── last.pt              ← Model dari epoch terakhir
    │   ├── epoch10.pt           ← Checkpoint setiap 10 epoch
    │   ├── epoch20.pt
    │   └── ...
    ├── results.png              ← Grafik training metrics
    ├── results.csv              ← Data metrics dalam CSV
    ├── confusion_matrix.png     ← Confusion matrix
    ├── confusion_matrix_normalized.png
    ├── F1_curve.png             ← F1 score curve
    ├── P_curve.png              ← Precision curve
    ├── R_curve.png              ← Recall curve
    ├── PR_curve.png             ← Precision-Recall curve
    ├── labels.jpg               ← Distribusi label dataset
    ├── labels_correlogram.jpg   ← Korelasi antar label
    ├── train_batch0.jpg         ← Sample augmented training images
    ├── train_batch1.jpg
    ├── train_batch2.jpg
    ├── val_batch0_labels.jpg    ← Ground truth validation
    ├── val_batch0_pred.jpg      ← Predictions pada validation
    └── args.yaml                ← Semua hyperparameter yang digunakan
```

### 2. Analisis Grafik results.png

Buka file `results.png` dan perhatikan:

#### A. Loss Curves (Harus Turun ⬇️)

**Box Loss, Class Loss, DFL Loss:**

- Grafik harus **trend menurun** dari kiri ke kanan
- Boleh ada fluktuasi kecil, tapi tren umum turun
- Di training from scratch, penurunan lebih **bertahap** (tidak langsung turun drastis)

**❌ Bad Training:**

```
Loss
  ^
  |    *
  |  *   *
  |*       *
  |          *
  +------------> Epoch
  (Naik turun gak karuan)
```

**✅ Good Training:**

```
Loss
  ^
  |*
  | \
  |  \__
  |     \___
  |         \___
  +------------> Epoch
  (Turun smooth)
```

#### B. Metrics Curves (Harus Naik ⬆️)

**Precision, Recall, mAP50, mAP50-95:**

- Grafik harus **trend naik** dari kiri ke kanan
- Di awal training from scratch, pertumbuhan **lambat**
- Setelah epoch 50-100, baru mulai naik signifikan

**Target Akhir (Epoch 300+):**

- mAP50: > 0.6 (60%) = OK, > 0.7 (70%) = Good, > 0.8 (80%) = Excellent
- mAP50-95: > 0.4 (40%) = OK, > 0.5 (50%) = Good, > 0.6 (60%) = Excellent

### 3. Confusion Matrix

Buka `confusion_matrix.png`:

**Ideal Confusion Matrix:**

```
          Predicted
          person  car  bicycle  background
Actual
person    [800]   10     5         15         ← Bagus! 800/830 correct
car        12   [750]   8         30         ← Bagus! 750/800 correct
bicycle    8     15   [650]       27         ← Bagus! 650/700 correct
background 20    25    10       [545]        ← Bagus!
```

**Interpretasi:**

- Diagonal (angka besar dalam kurung) = Prediksi BENAR ✅
- Off-diagonal (angka kecil) = Prediksi SALAH ❌
- Ideal: Diagonal jauh lebih besar dari off-diagonal

---

## Optimasi dan Tips

### 1. Hyperparameter Tuning

Setelah training pertama, optimalkan dengan menyesuaikan hyperparameter:

#### Learning Rate

**Terlalu tinggi** (lr0 > 0.1):

```python
# Symptoms: Loss naik-turun ekstrem, tidak konvergen
lr0=0.01,  # Turunkan
```

**Terlalu rendah** (lr0 < 0.001):

```python
# Symptoms: Training sangat lambat, loss turun sangat pelan
lr0=0.01,  # Naikkan
```

**Optimal untuk from scratch:**

```python
lr0=0.01,      # Starting learning rate
lrf=0.01,      # Final learning rate (1% of lr0)
```

#### Batch Size

**GPU Memory vs Batch Size:**

```python
# RTX 3060 (12GB):
batch=16, imgsz=640

# RTX 4090 (24GB):
batch=32, imgsz=640
# atau
batch=16, imgsz=1280  # untuk detail lebih tinggi

# A100 (80GB):
batch=128, imgsz=640
```

**Trade-off:**

- Batch besar = Training lebih cepat, tapi butuh VRAM besar
- Batch kecil = VRAM kecil cukup, tapi training lebih lama

#### Augmentation Strength

Untuk training from scratch, augmentation **SANGAT PENTING**:

**Weak Augmentation** (data sangat banyak > 50k):

```python
hsv_h=0.01,
hsv_s=0.5,
hsv_v=0.3,
fliplr=0.5,
mosaic=0.5,
```

**Medium Augmentation** (data 10k-50k) **← RECOMMENDED**:

```python
hsv_h=0.015,
hsv_s=0.7,
hsv_v=0.4,
fliplr=0.5,
mosaic=1.0,
```

**Strong Augmentation** (data < 10k):

```python
hsv_h=0.02,
hsv_s=0.9,
hsv_v=0.5,
degrees=10.0,        # Rotasi ±10 derajat
translate=0.2,       # Geser 20%
scale=0.9,           # Scale ±90%
fliplr=0.5,
mosaic=1.0,
mixup=0.15,          # Mixup 15%
copy_paste=0.1,      # Copy-paste 10%
```

### 2. Learning Rate Schedule

Gunakan cosine annealing atau step decay:

```python
# Cosine Annealing (Recommended)
lr0=0.01,
lrf=0.01,  # Smooth decay dari 0.01 → 0.0001

# Warmup untuk stabilisasi awal
warmup_epochs=3,      # 3 epoch warmup
warmup_momentum=0.8,
warmup_bias_lr=0.1,
```

### 3. Progressive Training Strategy

**Stage 1: Lower Resolution (Faster)**

```python
# Epochs 0-150: Train dengan resolusi rendah
epochs=150,
imgsz=416,      # Resolusi lebih rendah
batch=64,       # Batch bisa lebih besar
```

**Stage 2: Higher Resolution (Fine-tune)**

```python
# Epochs 150-300: Fine-tune dengan resolusi tinggi
model = YOLO('scratch_training/.../weights/last.pt')
model.train(
    epochs=150,  # 150 epoch lagi
    imgsz=640,   # Resolusi normal
    batch=32,
)
```

### 4. Data Augmentation Examples

Lihat augmented images untuk memastikan tidak terlalu ekstrem:

```python
# Cek train_batch0.jpg, train_batch1.jpg
# Pastikan:
# ✅ Objects masih recognizable
# ✅ Labels masih match dengan objects
# ❌ Objects terlalu distorted
# ❌ Colors tidak natural
```

---

## Troubleshooting

### ❓ Problem 1: CUDA Out of Memory

**Symptoms:**

```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions:**

```python
# Solusi 1: Kurangi batch size
batch=16,  # atau batch=8

# Solusi 2: Kurangi image size
imgsz=416,  # atau imgsz=320

# Solusi 3: Disable cache
cache=False,

# Solusi 4: Reduce workers
workers=4,

# Solusi 5: Gunakan model lebih kecil
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
```

---

### ❓ Problem 2: Loss Tidak Turun

**Symptoms:**

```
Epoch 100: loss = 2.5 (masih tinggi)
Epoch 150: loss = 2.3 (turun sangat lambat)
```

**Possible Causes & Solutions:**

**A. Learning Rate Terlalu Rendah**

```python
# Naikkan learning rate
lr0=0.01,  # dari 0.001 ke 0.01
```

**B. Dataset Terlalu Kecil**

```
Minimal 10,000 gambar untuk training from scratch
Solusi: Tambah data atau gunakan fine-tuning
```

**C. Label Error**

```
Cek labels - pastikan bounding box benar
Gunakan tools visualisasi untuk verify labels
```

**D. Augmentation Terlalu Kuat**

```python
# Kurangi augmentation
mosaic=0.5,  # dari 1.0
mixup=0.0,   # disable mixup
```

---

### ❓ Problem 3: Loss Menjadi NaN

**Symptoms:**

```
Epoch 50: loss = nan
```

**Solutions:**

**A. Learning Rate Terlalu Tinggi**

```python
lr0=0.001,  # Turunkan drastis
```

**B. Gradient Explosion**

```python
# Tambahkan gradient clipping
# (sudah default di Ultralytics, tapi bisa adjust)
max_grad_norm=10.0,
```

**C. Bad Data**

```
Cek apakah ada:
- Gambar corrupted
- Label dengan nilai invalid (< 0 atau > 1)
- Bounding box dengan width/height = 0
```

---

### ❓ Problem 4: Overfitting

**Symptoms:**

```
train_loss: 0.5 (rendah)
val_loss: 2.0 (tinggi)
→ Gap besar = Overfitting!
```

**Solutions:**

**A. Tambah Regularization**

```python
weight_decay=0.001,  # Naikkan dari 0.0005
dropout=0.1,         # Tambah dropout
```

**B. Tambah Augmentation**

```python
hsv_h=0.02,
hsv_s=0.9,
hsv_v=0.5,
mixup=0.15,
```

**C. Early Stopping**

```python
patience=30,  # Stop jika val_loss tidak improve 30 epochs
```

**D. Tambah Data**

```
Collect more training data
atau gunakan external dataset
```

---

### ❓ Problem 5: Training Sangat Lambat

**Symptoms:**

```
1 epoch = 2 jam
Estimated total: 600 jam (25 hari!)
```

**Solutions:**

**A. Gunakan Multiple GPUs**

```python
device='0,1,2,3',  # 4 GPUs → 4x faster
batch=128,
```

**B. Reduce Image Size**

```python
imgsz=416,  # dari 640
# Trade-off: Speed ↑, Accuracy ↓
```

**C. Mixed Precision Training**

```python
amp=True,  # FP16 → 2x faster
```

**D. Optimize Data Loading**

```python
workers=8,       # Sesuaikan dengan CPU cores
persistent_workers=True,
pin_memory=True,
```

**E. Cache Dataset ke RAM**

```python
cache='ram',  # Jika RAM > 64GB
# atau
cache=True,  # Cache ke disk
```

---

### ❓ Problem 6: mAP Tidak Naik Setelah Epoch 200

**Symptoms:**

```
Epoch 200: mAP = 0.45
Epoch 250: mAP = 0.46
Epoch 300: mAP = 0.46
→ Stuck!
```

**Solutions:**

**A. Learning Rate Warmup Restart**

```python
# Manual restart dengan LR tinggi
model = YOLO('weights/last.pt')
model.train(
    epochs=50,  # 50 epoch lagi
    lr0=0.005,  # LR lebih tinggi untuk "kick"
    resume=False,  # Fresh optimizer state
)
```

**B. Change Optimizer**

```python
optimizer='Adam',  # Coba Adam daripada SGD
lr0=0.001,         # Adam butuh LR lebih kecil
```

**C. Unfreeze Backbone**

```python
# Jika backbone di-freeze, unfreeze
freeze=0,  # Unfreeze all layers
```

---

## Kesimpulan dan Best Practices

### ✅ Checklist Training from Scratch

**Sebelum Training:**

- [ ] Dataset minimal 10,000 gambar
- [ ] Labels sudah di-verify (tidak ada error)
- [ ] Dataset split: 80% train, 15% val, 5% test
- [ ] GPU minimal 12GB VRAM
- [ ] Disk space cukup (100GB+)

**Saat Training:**

- [ ] Monitor loss (harus turun smooth)
- [ ] Monitor mAP (harus naik bertahap)
- [ ] Save checkpoint setiap 10-20 epochs
- [ ] Backup weights ke cloud/external drive

**Setelah Training:**

- [ ] Evaluasi confusion matrix
- [ ] Test pada data real-world
- [ ] Compare dengan baseline model
- [ ] Document hyperparameters & results

### 🎯 Expected Timeline

| Dataset Size | Model        | Hardware | Expected Time |
| ------------ | ------------ | -------- | ------------- |
| 10k images   | Mamba-YOLO-T | RTX 3060 | 5-7 hari      |
| 10k images   | Mamba-YOLO-B | RTX 4090 | 3-5 hari      |
| 50k images   | Mamba-YOLO-B | 4x A100  | 3-5 hari      |
| 100k images  | Mamba-YOLO-L | 8x A100  | 5-7 hari      |

### 📊 Expected Final Performance

**Good Training from Scratch:**

- mAP@50: **0.60 - 0.75** (60-75%)
- mAP@50-95: **0.40 - 0.55** (40-55%)
- Precision: **0.65 - 0.80**
- Recall: **0.60 - 0.75**

**Perbandingan dengan Fine-tuning:**

- From Scratch: Bisa mencapai akurasi optimal untuk domain spesifik
- Fine-tuning: Lebih cepat, tapi mungkin tidak optimal untuk domain sangat berbeda

---

## Referensi Lanjutan

**Training Monitoring Tools:**

- TensorBoard: `tensorboard --logdir=scratch_training`
- Weights & Biases: Integration dengan Ultralytics
- MLflow: Tracking experiments

**Advanced Techniques:**

- EMA (Exponential Moving Average): Automatic in Ultralytics
- Label Smoothing: Untuk reduce overfitting
- Knowledge Distillation: Transfer dari model besar ke kecil

**Dataset Tools:**

- Roboflow: Dataset management & augmentation
- CVAT: Annotation tool
- LabelImg: Simple bbox annotation

---

**Selamat Training! 🚀**

Jika ada pertanyaan atau butuh bantuan troubleshooting, jangan ragu untuk bertanya!
