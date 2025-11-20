# Panduan Penggunaan Mamba-YOLO untuk Object Detection

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Instalasi](#instalasi)
- [Model yang Tersedia](#model-yang-tersedia)
- [Tutorial Fine-tuning untuk Pemula (Deteksi Kepala)](#tutorial-fine-tuning-untuk-pemula-deteksi-kepala)
- [Cara Penggunaan](#cara-penggunaan)
  - [1. Training Model](#1-training-model)
  - [2. Inference/Prediksi](#2-inferenceprediksi)
  - [3. Validasi Model](#3-validasi-model)
  - [4. Export Model](#4-export-model)
- [Contoh Kode Python](#contoh-kode-python)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Pendahuluan

Mamba-YOLO adalah model object detection berbasis State Space Model (SSM) yang dikembangkan dari arsitektur YOLO. Model ini tersedia dalam 3 varian:
- **Mamba-YOLO-T** (Tiny): 5.8M parameter - untuk aplikasi ringan
- **Mamba-YOLO-B** (Base/Medium): 19.1M parameter - keseimbangan antara akurasi dan kecepatan
- **Mamba-YOLO-L** (Large): 57.6M parameter - untuk akurasi maksimal

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
cd Mamba-YOLO
```

### 2. Buat Virtual Environment
```bash
conda create -n mambayolo -y python=3.11
conda activate mambayolo
```

### 3. Install Dependencies
```bash
# Install PyTorch
pip3 install torch===2.3.0 torchvision torchaudio

# Install library pendukung
pip install seaborn thop timm einops

# Install selective scan (komponen SSM)
cd selective_scan
pip install .
cd ..

# Install Ultralytics dalam mode development
pip install -v -e .
```

---

## Model yang Tersedia

Model pre-trained Mamba-YOLO tersedia dalam 3 konfigurasi:

| Model | Config File | Parameter | FLOPs | mAP@50-95 |
|-------|-------------|-----------|-------|-----------|
| Mamba-YOLO-T | `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml` | 5.8M | 13.2G | 44.5% |
| Mamba-YOLO-B | `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml` | 19.1M | 45.4G | 49.1% |
| Mamba-YOLO-L | `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml` | 57.6M | 156.2G | 52.1% |

---

## Tutorial Fine-tuning untuk Pemula (Deteksi Kepala)

### 🤔 Apa itu Fine-tuning?
Bayangkan kamu sudah punya robot pintar yang sudah bisa mengenali banyak benda (mobil, anjing, kucing, dll). Sekarang kamu mau ajarin robot itu untuk jadi **ahli khusus** mengenali kepala manusia. 

Fine-tuning = mengajari robot yang sudah pintar untuk jadi lebih pintar lagi di tugas spesifik (deteksi kepala).

---

### 🎯 LANGKAH-LANGKAH LENGKAP

#### **LANGKAH 1: Siapkan Foto-foto Kepala** 📸

Pertama, kamu perlu kumpulin foto-foto yang ada kepalanya.

##### 1.1 Struktur Folder
Buat folder seperti ini di komputer:
```
C:\DATA\head_dataset\
├── images\
│   ├── train\          ← Foto untuk belajar (80% dari total foto)
│   │   ├── foto1.jpg
│   │   ├── foto2.jpg
│   │   └── foto3.jpg
│   └── val\            ← Foto untuk ujian (20% dari total foto)
│       ├── foto100.jpg
│       └── foto101.jpg
├── labels\
│   ├── train\          ← Label foto training (kotak di mana kepalanya)
│   │   ├── foto1.txt
│   │   ├── foto2.txt
│   │   └── foto3.txt
│   └── val\            ← Label foto validation
│       ├── foto100.txt
│       └── foto101.txt
└── head_dataset.yaml   ← File konfigurasi
```

##### 1.2 Cara Membuat Label
Setiap foto harus punya label (file .txt) yang bilang "kepalanya ada di sini":

**Contoh: foto1.txt**
```
0 0.5 0.3 0.15 0.25
```
Artinya:
- `0` = class kepala (karena kita cuma deteksi kepala, classnya 0)
- `0.5` = kepala ada di tengah-tengah lebar foto (50%)
- `0.3` = kepala ada di atas (30% dari tinggi foto)
- `0.15` = lebar kotak kepala (15% dari lebar foto)
- `0.25` = tinggi kotak kepala (25% dari tinggi foto)

**Tool untuk membuat label:**
- Gunakan **LabelImg** atau **Roboflow** (gratis & mudah)
- Tinggal klik-drag gambar kotak di kepalanya
- Otomatis jadi file .txt

##### 1.3 Buat File head_dataset.yaml
Buat file `head_dataset.yaml` di folder `C:\DATA\head_dataset\`:

```yaml
# File: head_dataset.yaml
path: C:\DATA\head_dataset
train: images\train
val: images\val

# Nama class (kita cuma punya 1 class: head)
names:
  0: head
```

---

#### **LANGKAH 2: Install Software yang Dibutuhkan** 💻

Buka **PowerShell** atau **Command Prompt**, lalu ketik satu per satu:

```powershell
# 1. Masuk ke folder Mamba-YOLO
cd "C:\DATA\KULIYEAH\Tugas Akhir\Mamba-YOLO"

# 2. Aktifkan environment (kalau sudah buat)
conda activate mambayolo

# 3. Cek apakah sudah install dengan benar
python -c "from ultralytics import YOLO; print('Berhasil!')"
```

Kalau keluar "Berhasil!" berarti sudah siap! ✅

---

#### **LANGKAH 3: Pilih Model Pre-trained** 📦

Model pre-trained = robot pintar yang sudah belajar dari ribuan foto.

Kamu punya 3 pilihan robot:

| Robot | Ukuran | Kecepatan | Akurasi | Pilih Kalau... |
|-------|--------|-----------|---------|----------------|
| **Mamba-YOLO-T** | Kecil (5.8M) | ⚡ Sangat Cepat | ⭐⭐⭐ Lumayan | Laptop lemah / butuh real-time |
| **Mamba-YOLO-B** | Sedang (19.1M) | ⚡⚡ Cepat | ⭐⭐⭐⭐ Bagus | **REKOMENDASI** untuk pemula |
| **Mamba-YOLO-L** | Besar (57.6M) | ⚡ Agak Lambat | ⭐⭐⭐⭐⭐ Sangat Bagus | Komputer kuat & butuh akurasi tinggi |

**Saran: Pakai Mamba-YOLO-B** (sedang/medium)

---

#### **LANGKAH 4: Fine-tuning Model (Melatih Ulang)** 🎓

Ini seperti ngajarin robot yang sudah pintar untuk fokus ke kepala saja.

##### 4.1 Cara Termudah: Buat Script Python

Buat file Python baru namanya `train_head_detection.py`:

```python
# File: train_head_detection.py
from ultralytics import YOLO

# 1. Load model pre-trained Mamba-YOLO
# Cara 1: Load dari config (training from scratch dengan arsitektur Mamba-YOLO)
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml')

# Cara 2: Load weights pre-trained kalau ada file .pt
# model = YOLO('path/to/pretrained_weights.pt')

# 2. Mulai fine-tuning
results = model.train(
    data='C:/DATA/head_dataset/head_dataset.yaml',  # Dataset kamu
    epochs=50,              # Latihan 50 kali putaran (bisa 30-100)
    imgsz=640,              # Ukuran foto jadi 640x640 pixel
    batch=8,                # Proses 8 foto sekaligus (turunin jadi 4 kalau error)
    device='0',             # Pakai GPU 0 (ganti 'cpu' kalau tidak ada GPU)
    patience=10,            # Berhenti otomatis kalau 10x tidak ada peningkatan
    save=True,              # Simpan model hasil training
    project='hasil_training',     # Folder hasil
    name='head_detection_v1',     # Nama experiment
    exist_ok=True,          # Ganti file lama kalau ada
    pretrained=True,        # Gunakan pre-trained weights (PENTING!)
    optimizer='SGD',        # Cara belajarnya
    verbose=True,           # Tampilkan progress
    lr0=0.01,               # Learning rate awal
    weight_decay=0.0005,    # Weight decay
)

print("✅ Training selesai!")
print(f"Model tersimpan di: {results.save_dir}")
```

##### 4.2 Jalankan Training

Di PowerShell, ketik:
```powershell
python train_head_detection.py
```

**Apa yang terjadi:**
- Komputer akan proses foto-foto kamu
- Muncul angka-angka (loss, precision, recall) → makin lama makin bagus
- Setelah selesai, model tersimpan otomatis

**Berapa lama?**
- Tergantung jumlah foto & komputer
- 100 foto + GPU = sekitar 10-30 menit
- 1000 foto + GPU = sekitar 1-3 jam
- Tanpa GPU = bisa 10x lebih lama

---

#### **LANGKAH 5: Hasil Training** 📊

Setelah selesai, cek folder:
```
hasil_training\
└── head_detection_v1\
    ├── weights\
    │   ├── best.pt         ← Model TERBAIK (pakai ini!)
    │   └── last.pt         ← Model terakhir
    ├── results.png         ← Grafik hasil training
    ├── confusion_matrix.png
    └── val_batch0_pred.jpg ← Contoh hasil deteksi
```

**File penting:**
- `best.pt` = Model terbaik yang kamu akan pakai untuk deteksi kepala
- `results.png` = Grafik untuk lihat apakah training bagus atau tidak

---

#### **LANGKAH 6: Tes Model Kamu (Inference)** 🧪

Sekarang saatnya tes robot pintarmu!

##### 6.1 Tes pada 1 Foto

Buat file `test_head_detection.py`:

```python
# File: test_head_detection.py
from ultralytics import YOLO
import cv2

# 1. Load model hasil training
model = YOLO('hasil_training/head_detection_v1/weights/best.pt')

# 2. Tes pada foto baru
foto_test = 'C:/path/to/foto_kepala.jpg'  # Ganti dengan foto kamu
results = model.predict(
    source=foto_test,
    conf=0.25,          # Deteksi kalau yakin minimal 25%
    save=True,          # Simpan hasil
    show=True           # Tampilkan langsung
)

# 3. Print hasil
for result in results:
    boxes = result.boxes
    print(f"Ditemukan {len(boxes)} kepala!")
    
    for box in boxes:
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  - Kepala dengan keyakinan {confidence*100:.1f}%")
        print(f"    Posisi: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

Jalankan:
```powershell
python test_head_detection.py
```

##### 6.2 Tes pada Video atau Webcam

```python
# File: test_video.py
from ultralytics import YOLO
import cv2

model = YOLO('hasil_training/head_detection_v1/weights/best.pt')

# Untuk video
cap = cv2.VideoCapture('video_orang.mp4')

# Untuk webcam, ganti dengan:
# cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi kepala di setiap frame
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # Tampilkan hasil
    annotated_frame = results[0].plot()
    cv2.imshow('Deteksi Kepala', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

#### **LANGKAH 7: Cara Tahu Training Berhasil atau Tidak** ✅❌

Buka file `hasil_training/head_detection_v1/results.png`

**Lihat grafik:**

1. **train/box_loss** dan **val/box_loss** → Harus turun seperti tangga ke bawah ⬇️
   - Kalau naik-turun gak karuan = training gagal ❌

2. **metrics/mAP50** → Harus naik seperti tangga ke atas ⬆️
   - Kalau di atas 0.7 (70%) = BAGUS! ✅
   - Kalau di bawah 0.5 (50%) = perlu lebih banyak foto 📸

3. **Cek confusion_matrix.png**
   - Kotak besar di diagonal = BAGUS ✅
   - Banyak kotak di luar diagonal = model bingung ❌

---

### 💡 TIPS PENTING

#### ✅ DO (Yang Harus Dilakukan):
1. **Foto harus jelas** - Jangan blur atau gelap
2. **Foto harus beragam** - Berbagai angle, jarak, cahaya
3. **Minimal 100-300 foto** - Makin banyak makin bagus
4. **Label harus benar** - Kotak pas di kepalanya
5. **80% training, 20% validation** - Jangan lupa split

#### ❌ DON'T (Jangan Dilakukan):
1. Jangan pakai foto yang sama di training dan validation
2. Jangan label asal-asalan (kotak di luar kepala)
3. Jangan epochs terlalu besar kalau foto sedikit (overfitting)

---

### 🔧 TROUBLESHOOTING

**❓ Error: CUDA out of memory**
```python
# Solusi: Turunkan batch size
batch=4,  # atau batch=2
```

**❓ Training sangat lambat**
```python
# Solusi: Kecilkan ukuran image atau pakai model lebih kecil
imgsz=416,  # atau 320
# atau ganti ke Mamba-YOLO-T
```

**❓ Akurasi rendah (mAP < 50%)**
- Tambah lebih banyak foto (minimal 300)
- Cek labelnya benar semua
- Naikkan epochs jadi 100
- Pastikan foto beragam

**❓ Model deteksi terlalu banyak (false positive)**
```python
# Naikkan confidence threshold
results = model.predict(source=foto, conf=0.5)  # dari 0.25 ke 0.5
```

---

### 📝 RINGKASAN SINGKAT

```
1. Kumpulin foto kepala (minimal 100)
2. Buat label pakai LabelImg
3. Buat file head_dataset.yaml
4. Buat train_head_detection.py
5. Jalankan: python train_head_detection.py
6. Tunggu selesai (1-3 jam)
7. Tes pakai best.pt
8. DONE! 🎉
```

---

## Cara Penggunaan

### 1. Training Model

#### Training dari Scratch
```bash
python mbyolo_train.py \
  --task train \
  --data ultralytics/cfg/datasets/coco.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
  --epochs 300 \
  --batch_size 16 \
  --device 0 \
  --amp \
  --project ./output_dir/mscoco \
  --name mambayolo_t
```

#### Training dengan Custom Dataset
```bash
python mbyolo_train.py \
  --task train \
  --data path/to/custom_dataset.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml \
  --epochs 100 \
  --batch_size 8 \
  --device 0,1 \
  --amp \
  --project ./output_dir/custom \
  --name my_model
```

**Catatan**: Dataset harus dalam format COCO dengan struktur:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

---

### 2. Inference/Prediksi

#### A. Menggunakan Command Line

**Prediksi pada gambar tunggal:**
```bash
python mbyolo_train.py \
  --task predict \
  --config path/to/model_weights.pt \
  --data path/to/image.jpg \
  --device 0
```

**Prediksi pada folder gambar:**
```bash
python mbyolo_train.py \
  --task predict \
  --config path/to/model_weights.pt \
  --data path/to/images/ \
  --device 0
```

**Prediksi pada video:**
```bash
python mbyolo_train.py \
  --task predict \
  --config path/to/model_weights.pt \
  --data path/to/video.mp4 \
  --device 0
```

**Prediksi dengan webcam:**
```bash
python mbyolo_train.py \
  --task predict \
  --config path/to/model_weights.pt \
  --data 0 \
  --device 0
```

#### B. Menggunakan Python API

```python
from ultralytics import YOLO

# Load model pre-trained
model = YOLO('path/to/weights.pt')

# Prediksi pada gambar
results = model('path/to/image.jpg')

# Prediksi dengan parameter custom
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,        # confidence threshold
    iou=0.45,         # NMS IoU threshold
    imgsz=640,        # ukuran input image
    device='0',       # GPU device
    save=True,        # simpan hasil
    save_txt=True     # simpan label dalam txt
)

# Akses hasil prediksi
for result in results:
    boxes = result.boxes  # object boxes
    for box in boxes:
        cls = int(box.cls[0])  # class id
        conf = float(box.conf[0])  # confidence
        xyxy = box.xyxy[0].tolist()  # bounding box koordinat
        print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")
```

#### C. Prediksi Batch dengan Loop

```python
from ultralytics import YOLO
import os
from pathlib import Path

# Load model
model = YOLO('path/to/weights.pt')

# Folder gambar
image_folder = 'path/to/images'
output_folder = 'path/to/results'
os.makedirs(output_folder, exist_ok=True)

# Proses semua gambar
for img_file in Path(image_folder).glob('*.jpg'):
    results = model.predict(
        source=str(img_file),
        conf=0.25,
        save=True,
        project=output_folder
    )
    print(f"Processed: {img_file.name}")
```

---

### 3. Validasi Model

Validasi model pada dataset untuk mengukur performa:

```bash
python mbyolo_train.py \
  --task val \
  --config path/to/model_weights.pt \
  --data ultralytics/cfg/datasets/coco.yaml \
  --batch_size 32 \
  --device 0
```

Atau menggunakan Python:

```python
from ultralytics import YOLO

# Load model
model = YOLO('path/to/weights.pt')

# Validasi
metrics = model.val(
    data='ultralytics/cfg/datasets/coco.yaml',
    batch=32,
    imgsz=640,
    device='0'
)

# Print metrics
print(f"mAP@50-95: {metrics.box.map}")
print(f"mAP@50: {metrics.box.map50}")
print(f"mAP@75: {metrics.box.map75}")
```

---

### 4. Export Model

Export model ke format lain untuk deployment:

```python
from ultralytics import YOLO

# Load model
model = YOLO('path/to/weights.pt')

# Export ke ONNX
model.export(format='onnx', imgsz=640)

# Export ke TensorRT
model.export(format='engine', imgsz=640, device=0)

# Export ke TorchScript
model.export(format='torchscript', imgsz=640)

# Export ke TFLite
model.export(format='tflite', imgsz=640)
```

Format export yang tersedia:
- `onnx` - ONNX format
- `engine` - TensorRT engine
- `torchscript` - TorchScript
- `tflite` - TensorFlow Lite
- `coreml` - CoreML (untuk iOS)
- `pb` - TensorFlow SavedModel

---

## Contoh Kode Python

### Contoh Lengkap: Inference dengan Visualisasi

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load model pre-trained Mamba-YOLO
model = YOLO('path/to/mambayolo_weights.pt')

# Load gambar
image_path = 'path/to/image.jpg'
image = cv2.imread(image_path)

# Prediksi
results = model.predict(
    source=image,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device='0'
)

# Visualisasi hasil
for result in results:
    # Gambar dengan annotation
    annotated_img = result.plot()
    
    # Tampilkan
    cv2.imshow('Mamba-YOLO Detection', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Simpan hasil
    cv2.imwrite('result.jpg', annotated_img)
    
    # Print detil deteksi
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        print(f"Object: {cls_name}")
        print(f"Confidence: {conf:.3f}")
        print(f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print("-" * 40)
```

### Contoh: Inference Real-time pada Video

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('path/to/weights.pt')

# Buka video atau webcam (0 untuk webcam)
cap = cv2.VideoCapture('path/to/video.mp4')  # atau 0 untuk webcam

# Setup video writer untuk menyimpan hasil
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # Visualisasi
    annotated_frame = results[0].plot()
    
    # Tampilkan
    cv2.imshow('Mamba-YOLO Video Detection', annotated_frame)
    
    # Simpan
    out.write(annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

### Contoh: Load Model dari Config YAML

```python
from ultralytics import YOLO

# Build model baru dari config (untuk training from scratch)
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')

# Atau load pre-trained weights
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
model = YOLO('path/to/pretrained_weights.pt')  # load weights

# Training
results = model.train(
    data='ultralytics/cfg/datasets/coco.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='0'
)
```

---

## Tips & Troubleshooting

### 1. **Memilih Model yang Tepat**
- **Mamba-YOLO-T**: Gunakan untuk aplikasi real-time pada device dengan resource terbatas
- **Mamba-YOLO-B**: Pilihan terbaik untuk keseimbangan akurasi dan kecepatan
- **Mamba-YOLO-L**: Gunakan jika prioritas utama adalah akurasi

### 2. **Optimasi Inference**
```python
# Gunakan half precision untuk inference lebih cepat (GPU support FP16)
model = YOLO('path/to/weights.pt')
results = model.predict(source='image.jpg', half=True, device='0')

# Batch processing untuk multiple images
results = model.predict(source=['img1.jpg', 'img2.jpg', 'img3.jpg'], batch=8)
```

### 3. **Menyesuaikan Confidence Threshold**
```python
# Turunkan conf jika terlalu sedikit deteksi
results = model.predict(source='image.jpg', conf=0.15)

# Naikkan conf jika terlalu banyak false positive
results = model.predict(source='image.jpg', conf=0.5)
```

### 4. **Multi-GPU Training**
```bash
# Training dengan multiple GPU
python mbyolo_train.py \
  --task train \
  --data dataset.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml \
  --device 0,1,2,3 \
  --batch_size 64
```

### 5. **Memory Error**
Jika mendapat error out of memory:
- Kurangi batch size: `--batch_size 8` atau lebih kecil
- Kurangi ukuran image: `--imgsz 320` atau `--imgsz 416`
- Gunakan model yang lebih kecil (Mamba-YOLO-T)

### 6. **Custom Dataset Format**
Dataset YAML harus berisi:
```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
names:
  0: person
  1: car
  2: dog
  # ... dst
```

### 7. **Resume Training**
```python
# Resume dari checkpoint terakhir
model = YOLO('path/to/last.pt')
model.train(resume=True)
```

### 8. **Menggunakan AMP (Automatic Mixed Precision)**
```bash
# Tambahkan --amp untuk training lebih cepat dengan memory lebih efisien
python mbyolo_train.py --task train --amp
```

---

## Parameter Penting

### Training Parameters
- `--epochs`: Jumlah epoch training (default: 300)
- `--batch_size`: Batch size (default: 512, sesuaikan dengan VRAM)
- `--imgsz`: Ukuran input image (default: 640)
- `--device`: GPU device (contoh: '0' atau '0,1,2,3')
- `--amp`: Enable Automatic Mixed Precision
- `--optimizer`: Optimizer (SGD, Adam, AdamW)
- `--workers`: Jumlah data loader workers

### Inference Parameters
- `conf`: Confidence threshold (0-1, default: 0.25)
- `iou`: IoU threshold untuk NMS (0-1, default: 0.45)
- `max_det`: Maximum detections per image (default: 300)
- `classes`: Filter hanya class tertentu
- `agnostic_nms`: Class-agnostic NMS
- `half`: Use FP16 half-precision inference

---

## Struktur Output

Setelah inference, hasil akan disimpan dalam struktur:
```
output_dir/
└── project_name/
    └── name/
        ├── labels/          # File .txt dengan format YOLO
        ├── crops/           # Cropped objects (opsional)
        └── image_name.jpg   # Gambar dengan annotation
```

Format file label (.txt):
```
class_id confidence x_center y_center width height
0 0.95 0.5 0.5 0.3 0.4
1 0.87 0.2 0.3 0.15 0.2
```

---

## Referensi

- Paper: [Mamba YOLO: SSMs-Based YOLO For Object Detection](https://arxiv.org/abs/2406.05835)
- Repository: [https://github.com/HZAI-ZJNU/Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- Ultralytics Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)

---

## Citation

Jika menggunakan Mamba-YOLO dalam penelitian, silakan cite:

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
    title={Mamba YOLO: SSMs-Based YOLO For Object Detection}, 
    author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
    year={2024},
    eprint={2406.05835},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2406.05835}, 
}
```

---

**Dibuat untuk memudahkan penggunaan Mamba-YOLO dalam object detection tasks.**
