# Pengantar Arsitektur Mamba-YOLO: Memahami Alur Kerja dari Backbone hingga Head

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Arsitektur Keseluruhan](#arsitektur-keseluruhan)
- [Komponen Utama](#komponen-utama)
  - [1. Backbone: Object Detection Mamba (ODMamba)](#1-backbone-object-detection-mamba-odmamba)
  - [2. Neck: PAFPN (Path Aggregation Feature Pyramid Network)](#2-neck-pafpn-path-aggregation-feature-pyramid-network)
  - [3. Head: Decoupled Head](#3-head-decoupled-head)
- [Alur Data Lengkap](#alur-data-lengkap)
- [Perbandingan dengan YOLO Klasik](#perbandingan-dengan-yolo-klasik)
- [Kelebihan Arsitektur Mamba-YOLO](#kelebihan-arsitektur-mamba-yolo)
- [Kesimpulan](#kesimpulan)

---

## Pendahuluan

**Mamba-YOLO** adalah arsitektur object detection yang menggabungkan **State Space Models (SSM)** dengan struktur YOLO tradisional. Berbeda dengan YOLO klasik yang menggunakan CNN atau Transformer, Mamba-YOLO menggunakan **Selective Scan Mechanism** untuk memproses informasi spasial secara lebih efisien.

### 🎯 Tujuan Dokumen

Dokumen ini menjelaskan **bagaimana data gambar diproses** dari awal (input) hingga menghasilkan deteksi objek (output), melalui 3 komponen utama:

1. **Backbone (ODMamba)** - Ekstraksi fitur
2. **Neck (PAFPN)** - Agregasi multi-scale features
3. **Head (Decoupled Head)** - Prediksi bounding box & class

---

## Arsitektur Keseluruhan

### Diagram Alur Sederhana

```
Input Image (640x640x3)
        ↓
┌───────────────────────────────────────┐
│  BACKBONE (ODMamba)                   │
│  - SimpleStem                         │
│  - VSSBlock (Mamba Blocks)            │
│  - VisionClueMerge (Downsampling)     │
├───────────────────────────────────────┤
│  Output: Multi-scale Features         │
│  • P3: 80x80x256   (1/8 resolution)   │
│  • P4: 40x40x512   (1/16 resolution)  │
│  • P5: 20x20x1024  (1/32 resolution)  │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  NECK (PAFPN)                         │
│  - Top-down pathway (Upsample)        │
│  - Bottom-up pathway (Downsample)     │
│  - Feature fusion with XSSBlock       │
├───────────────────────────────────────┤
│  Output: Enhanced Multi-scale Features│
│  • P3: 80x80x256                      │
│  • P4: 40x40x512                      │
│  • P5: 20x20x1024                     │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  HEAD (Decoupled Head)                │
│  - Classification Branch (cv3)        │
│  - Regression Branch (cv2)            │
├───────────────────────────────────────┤
│  Output: Predictions                  │
│  • Bounding Boxes (x, y, w, h)        │
│  • Class Probabilities (80 classes)   │
└───────────────────────────────────────┘
        ↓
Final Detections (NMS)
```

---

## Komponen Utama

## 1. Backbone: Object Detection Mamba (ODMamba)

### 🎯 Fungsi Utama

Backbone bertugas **mengekstrak fitur hierarkis** dari gambar input dengan resolusi berbeda untuk menangkap objek dengan ukuran berbeda (kecil, sedang, besar).

### 📐 Struktur Backbone Mamba-YOLO-B

Berdasarkan file config `Mamba-YOLO-B.yaml`:

```yaml
backbone:
  - [-1, 1, SimpleStem, [128, 3]] # Layer 0: P2/4
  - [-1, 3, VSSBlock, [128]] # Layer 1: Mamba blocks
  - [-1, 1, VisionClueMerge, [256]] # Layer 2: P3/8 ← Output 1
  - [-1, 3, VSSBlock, [256]] # Layer 3: Mamba blocks
  - [-1, 1, VisionClueMerge, [512]] # Layer 4: P4/16 ← Output 2
  - [-1, 9, VSSBlock, [512]] # Layer 5: Mamba blocks
  - [-1, 1, VisionClueMerge, [1024]] # Layer 6: P5/32
  - [-1, 3, VSSBlock, [1024]] # Layer 7: Mamba blocks
  - [-1, 1, SPPF, [1024, 5]] # Layer 8: SPPF ← Output 3
```

### 🔍 Detail Setiap Komponen

---

#### A. SimpleStem - "Gerbang Masuk"

**Lokasi:** Layer 0  
**Input:** `(B, 3, 640, 640)` - Gambar RGB  
**Output:** `(B, 128, 160, 160)` - Feature map dengan 1/4 resolusi

**Kode Implementasi:**

```python
class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            # First Conv: 640→320, channels: 3→64
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            # Second Conv: 320→160, channels: 64→128
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
```

**Fungsi:**

- Mengurangi resolusi spasial 4x (640→160)
- Mengubah 3 channel RGB menjadi 128 channel fitur
- Mirip seperti "zoom out" untuk melihat gambar dari jauh

**Analogi Sederhana:**

```
Foto HD (640x640 pixel, warna RGB)
        ↓ [Kompresi + Ekstraksi pola dasar]
Sketsa kasar (160x160, 128 layer informasi)
```

---

#### B. VSSBlock - "Otak Pemroses Utama"

**Lokasi:** Layer 1, 3, 5, 7  
**Jumlah:** Berulang 3-9x per stage  
**Fungsi:** Memproses fitur dengan **Selective Scan Mechanism**

**Struktur VSSBlock:**

```python
class VSSBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        # 1. Projection: Channel adjustment
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 2. Local Structure Block (LSBlock)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)

        # 3. Selective Scan 2D (SS2D) - CORE MAMBA!
        self.op = SS2D(d_model=hidden_dim, d_state=16, ssm_ratio=2.0)

        # 4. MLP (Feed-Forward)
        self.mlp = RGBlock(hidden_dim, hidden_dim * 4)

    def forward(self, input):
        # Step 1: Adjust channels
        x = self.proj_conv(input)

        # Step 2: Extract local patterns
        x_local = self.lsblock(x)

        # Step 3: Apply Selective Scan (MAMBA MAGIC!)
        x_mamba = x + self.op(self.norm(x_local))

        # Step 4: Enhance features with MLP
        output = x_mamba + self.mlp(x_mamba)

        return output
```

**Apa yang Terjadi di VSSBlock?**

1. **LSBlock (Local Structure Block)**

   - Menangkap pola lokal dengan depth-wise convolution
   - Mirip seperti "melihat detail kecil" (edges, corners)

2. **SS2D (Selective Scan 2D)** - **INI YANG SPESIAL!**

   - Scan gambar dari **4 arah berbeda** (↑ ↓ ← →)
   - Pilih informasi penting secara selektif (selective)
   - Lebih efisien dari attention mechanism

3. **MLP (RGBlock)**
   - Proses non-linear untuk memperkaya representasi
   - Mirip "menambah konteks" pada fitur yang sudah ada

**Visualisasi Selective Scan:**

```
Original Image:
┌─────────────┐
│  🚗  👤  🏠  │
│             │
│  🌳  🐕  ⚽  │
└─────────────┘

Scan Direction 1 (→):  🚗 → 👤 → 🏠 → 🌳 → 🐕 → ⚽
Scan Direction 2 (←):  ⚽ → 🐕 → 🌳 → 🏠 → 👤 → 🚗
Scan Direction 3 (↓):  🚗 → 🌳 → 👤 → 🐕 → 🏠 → ⚽
Scan Direction 4 (↑):  ⚽ → 🏠 → 🐕 → 👤 → 🌳 → 🚗

→ Aggregate semua hasil scan
→ Dapatkan representasi global yang kaya!
```

**Keunggulan SS2D vs Attention:**
| Aspek | Attention (Transformer) | Selective Scan (Mamba) |
|-------|------------------------|------------------------|
| Complexity | O(N²) - Kuadratik | O(N) - Linear |
| Memory | Tinggi | Rendah |
| Speed | Lambat untuk resolusi tinggi | Cepat |
| Long-range | Bagus | Bagus |

---

#### C. VisionClueMerge - "Downsampling Pintar"

**Lokasi:** Layer 2, 4, 6  
**Fungsi:** Mengurangi resolusi 2x sambil menggandakan jumlah channel

**Kode Implementasi:**

```python
class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)
        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # Pixel unshuffle: 2x2 patch → 4 channels
        y = torch.cat([
            x[..., ::2, ::2],   # Top-left pixels
            x[..., 1::2, ::2],  # Top-right pixels
            x[..., ::2, 1::2],  # Bottom-left pixels
            x[..., 1::2, 1::2]  # Bottom-right pixels
        ], dim=1)  # Concatenate on channel dimension
        return self.pw_linear(y)
```

**Visualisasi:**

```
Input: (B, 128, 160, 160)

Patch 2x2:
┌──┬──┐
│ A│ B│  →  [A, B, C, D] di channel dimension
├──┼──┤
│ C│ D│
└──┴──┘

Output: (B, 256, 80, 80)
- Resolusi: 160 → 80 (2x lebih kecil)
- Channel: 128 → 256 (2x lebih banyak)
- Tidak ada informasi hilang! (Pixel unshuffle)
```

**Analogi:**

```
Foto beresolusi tinggi
        ↓ [Pixel unshuffle]
Foto resolusi rendah TAPI informasi tetap lengkap
(seperti zoom out tanpa kehilangan detail)
```

---

#### D. SPPF (Spatial Pyramid Pooling Fast)

**Lokasi:** Layer 8 (akhir backbone)  
**Fungsi:** Menangkap fitur multi-scale dengan pooling berbeda

**Cara Kerja:**

```python
Input: (B, 1024, 20, 20)
     ↓
┌────────────────────────────────────────┐
│ MaxPool 5x5 → Feature A (Global info)  │
│ MaxPool 5x5 → Feature B (Mid info)     │
│ MaxPool 5x5 → Feature C (Local info)   │
│ Original → Feature D                   │
└────────────────────────────────────────┘
     ↓ [Concatenate all]
Output: (B, 1024, 20, 20)
```

**Manfaat:**

- Tangkap objek dalam berbagai ukuran
- Perluas receptive field tanpa menambah complexity

---

### 📊 Ringkasan Output Backbone

Setelah melewati semua layer backbone, kita dapat **3 feature maps** dengan resolusi berbeda:

```python
# Output dari Backbone
P3 = Layer[3]   # (B, 256, 80, 80)   - 1/8 resolution  → Deteksi objek KECIL
P4 = Layer[5]   # (B, 512, 40, 40)   - 1/16 resolution → Deteksi objek SEDANG
P5 = Layer[8]   # (B, 1024, 20, 20)  - 1/32 resolution → Deteksi objek BESAR
```

**Analogi Sederhana:**

```
P5 (20x20)   : Melihat dari jauh  → Deteksi mobil, bangunan (objek besar)
P4 (40x40)   : Melihat dari sedang → Deteksi orang, sepeda (objek sedang)
P3 (80x80)   : Melihat dari dekat  → Deteksi tangan, wajah (objek kecil)
```

---

## 2. Neck: PAFPN (Path Aggregation Feature Pyramid Network)

### 🎯 Fungsi Utama

Neck bertugas **menggabungkan fitur dari berbagai skala** agar setiap level feature map punya informasi dari level lainnya. Ini penting karena:

- Objek besar butuh informasi detail (dari P3)
- Objek kecil butuh informasi konteks (dari P5)

### 📐 Struktur PAFPN

Berdasarkan file config:

```yaml
head:
  # TOP-DOWN PATHWAY (Coarse-to-Fine)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # Layer 9: Upsample P5
  - [[-1, 5], 1, Concat, [1]] # Layer 10: Cat P5 + P4
  - [-1, 3, XSSBlock, [512]] # Layer 11: Fuse → P4'

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # Layer 12: Upsample P4'
  - [[-1, 3], 1, Concat, [1]] # Layer 13: Cat P4' + P3
  - [-1, 3, XSSBlock, [256]] # Layer 14: Fuse → P3' ✓

  # BOTTOM-UP PATHWAY (Fine-to-Coarse)
  - [-1, 1, Conv, [256, 3, 2]] # Layer 15: Downsample P3'
  - [[-1, 11], 1, Concat, [1]] # Layer 16: Cat P3'_down + P4'
  - [-1, 3, XSSBlock, [512]] # Layer 17: Fuse → P4'' ✓

  - [-1, 1, Conv, [512, 3, 2]] # Layer 18: Downsample P4''
  - [[-1, 8], 1, Concat, [1]] # Layer 19: Cat P4''_down + P5
  - [-1, 3, XSSBlock, [1024]] # Layer 20: Fuse → P5'' ✓
```

### 🔄 Alur Kerja PAFPN

#### **Phase 1: Top-Down Pathway (Informasi Global → Lokal)**

**Tujuan:** Bawa informasi semantic tingkat tinggi (dari P5) ke level rendah (P3)

```
Step 1: P5 → P4'
┌─────────────────────────────────────────┐
│ P5 (20x20x1024) [Konteks global]       │
└─────────────────────────────────────────┘
        ↓ [Upsample 2x]
(40x40x1024)
        ↓ [Concat dengan P4 dari backbone]
(40x40x1536) = P5_up + P4_backbone
        ↓ [XSSBlock fusion]
┌─────────────────────────────────────────┐
│ P4' (40x40x512) [Gabungan info]        │
└─────────────────────────────────────────┘

Step 2: P4' → P3'
┌─────────────────────────────────────────┐
│ P4' (40x40x512)                         │
└─────────────────────────────────────────┘
        ↓ [Upsample 2x]
(80x80x512)
        ↓ [Concat dengan P3 dari backbone]
(80x80x768) = P4'_up + P3_backbone
        ↓ [XSSBlock fusion]
┌─────────────────────────────────────────┐
│ P3' (80x80x256) [Detail + Konteks] ✓   │
└─────────────────────────────────────────┘
```

**Hasil Phase 1:**

- P3' sekarang punya informasi dari P4 dan P5 (konteks global)
- Bagus untuk deteksi objek kecil yang butuh konteks

---

#### **Phase 2: Bottom-Up Pathway (Informasi Lokal → Global)**

**Tujuan:** Bawa informasi detail dari P3' kembali ke P4 dan P5

```
Step 1: P3' → P4''
┌─────────────────────────────────────────┐
│ P3' (80x80x256) [Detail tinggi]        │
└─────────────────────────────────────────┘
        ↓ [Downsample Conv 3x3 stride=2]
(40x40x256)
        ↓ [Concat dengan P4' dari top-down]
(40x40x768) = P3'_down + P4'
        ↓ [XSSBlock fusion]
┌─────────────────────────────────────────┐
│ P4'' (40x40x512) [Detail + Konteks] ✓  │
└─────────────────────────────────────────┘

Step 2: P4'' → P5''
┌─────────────────────────────────────────┐
│ P4'' (40x40x512)                        │
└─────────────────────────────────────────┘
        ↓ [Downsample Conv 3x3 stride=2]
(20x20x512)
        ↓ [Concat dengan P5 dari backbone]
(20x20x1536) = P4''_down + P5_backbone
        ↓ [XSSBlock fusion]
┌─────────────────────────────────────────┐
│ P5'' (20x20x1024) [Semua info] ✓       │
└─────────────────────────────────────────┘
```

**Hasil Phase 2:**

- P4'' dan P5'' sekarang punya informasi detail dari P3
- Bagus untuk deteksi objek besar yang butuh detail

---

### 🔗 XSSBlock - "Fusion Engine"

**Fungsi:** Menggabungkan fitur dari 2 level berbeda secara cerdas

**Perbedaan XSSBlock vs VSSBlock:**

| Aspek                | VSSBlock (Backbone) | XSSBlock (Neck)       |
| -------------------- | ------------------- | --------------------- |
| Input                | Single feature map  | Concatenated features |
| Repeats              | 3-9x per stage      | 3x per fusion         |
| Purpose              | Feature extraction  | Feature fusion        |
| Has input projection | Yes (proj_conv)     | Yes (in_proj)         |

**Struktur XSSBlock:**

```python
class XSSBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, n=1):
        super().__init__()
        # 1. Input projection (jika channel berbeda)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()

        # 2. Multiple SS2D layers (repeat n times)
        self.ss2d = nn.Sequential(*(SS2D(...) for _ in range(n)))

        # 3. Local structure + MLP
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp = RGBlock(hidden_dim, hidden_dim * 4)
```

**Kenapa Perlu Fusion dengan Mamba?**

- **Selective Scan** bisa fokus pada informasi relevan dari kedua level
- Lebih efisien dari simple concatenation + convolution
- Preserve long-range dependency antar level

---

### 📊 Ringkasan Output Neck

Setelah melewati PAFPN, kita dapat **3 enhanced feature maps**:

```python
# Output dari Neck (untuk detection head)
P3_final = Layer[14]   # (B, 256, 80, 80)   → Small objects
P4_final = Layer[17]   # (B, 512, 40, 40)   → Medium objects
P5_final = Layer[20]   # (B, 1024, 20, 20)  → Large objects
```

**Perbedaan dengan Output Backbone:**

```
Backbone Output:        Neck Output:
P3: Info lokal saja  →  P3': Info lokal + global
P4: Info sedang      →  P4'': Info detail + konteks
P5: Info global saja →  P5'': Info global + detail
```

---

## 3. Head: Decoupled Head

### 🎯 Fungsi Utama

Head bertugas **memprediksi bounding box dan class** untuk setiap objek di gambar.

**Kenapa "Decoupled"?**

- Branch untuk **bounding box** dan **classification** dipisah
- Berbeda dengan YOLO klasik yang jadi satu branch
- Hasil lebih akurat karena task-specific optimization

### 📐 Struktur Detect Head

Berdasarkan kode di `head.py`:

```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 80 classes (COCO)
        self.nl = len(ch)  # 3 detection layers (P3, P4, P5)
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # Total outputs

        # Branch 1: Bounding Box Regression
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),      # Conv 3x3
                Conv(c2, c2, 3),     # Conv 3x3
                nn.Conv2d(c2, 4 * self.reg_max, 1)  # Predict: 64 values
            ) for x in ch
        )

        # Branch 2: Classification
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),      # Conv 3x3
                Conv(c3, c3, 3),     # Conv 3x3
                nn.Conv2d(c3, self.nc, 1)  # Predict: 80 classes
            ) for x in ch
        )

        # DFL: Distribution Focal Loss
        self.dfl = DFL(self.reg_max)
```

### 🔍 Detail Setiap Komponen

---

#### A. Bounding Box Branch (cv2)

**Input:** Feature maps dari P3, P4, P5  
**Output:** `4 * reg_max = 64` nilai untuk setiap anchor

**Proses:**

```
P3 (80x80x256)
     ↓ [Conv 3x3 + Conv 3x3]
(80x80x c2)
     ↓ [Conv 1x1]
(80x80x64)  →  64 = 4 (sisi box) × 16 (distribusi per sisi)
```

**Kenapa 64 output, bukan 4?**

Mamba-YOLO menggunakan **Distribution Focal Loss (DFL)**:

- Tidak prediksi langsung `(x, y, w, h)`
- Prediksi **distribusi probabilitas** untuk setiap sisi box
- Lebih akurat karena model uncertainty

**Visualisasi DFL:**

```
Traditional Regression:
left_offset = 5.3 pixel  [satu nilai]

DFL Regression:
left_offset distribution:
[0.0, 0.0, 0.1, 0.2, 0.4, 0.2, 0.1, 0.0, ...]
         ↑   ↑   ↑___Peak di 5.3___↑   ↑
Confidence di setiap possible offset (0-15)
```

**Keuntungan DFL:**

- Capture uncertainty → lebih robust
- Continuous prediction dari discrete values
- Better gradient flow during training

---

#### B. Classification Branch (cv3)

**Input:** Feature maps dari P3, P4, P5  
**Output:** `nc = 80` nilai probabilitas untuk setiap class

**Proses:**

```
P3 (80x80x256)
     ↓ [Conv 3x3 + Conv 3x3]
(80x80x c3)
     ↓ [Conv 1x1]
(80x80x80)  →  80 classes (person, car, dog, ...)
     ↓ [Sigmoid]
Probability per class [0.0 - 1.0]
```

**Output Format:**

```python
# Untuk setiap cell di feature map
class_scores = [
    0.95,  # person (sangat yakin)
    0.03,  # bicycle
    0.82,  # car (cukup yakin)
    0.01,  # motorcycle
    ...    # 76 classes lainnya
]
```

---

#### C. DFL (Distribution Focal Loss)

**Fungsi:** Konversi distribusi probabilitas → bounding box coordinates

**Implementasi:**

```python
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # x: (B, 4*reg_max, H, W) = (B, 64, H, W)
        b, _, h, w = x.shape
        # Reshape: (B, 4, 16, H, W)
        x = x.view(b, 4, self.c1, h, w)
        # Softmax on distribution dimension
        x = x.softmax(2)
        # Weighted sum: expected value
        x = self.conv(x.view(b, -1, h, w)).view(b, 4, h, w)
        return x
```

**Cara Kerja:**

```
Input: [0.0, 0.1, 0.3, 0.4, 0.2, 0.0, ...]  (16 values)
         ↓ [Softmax - normalize]
       [0.05, 0.10, 0.25, 0.35, 0.20, 0.05, ...]
         ↓ [Weighted sum dengan index]
Expected value = 0×0.05 + 1×0.10 + 2×0.25 + 3×0.35 + 4×0.20 + ...
               = 2.8 pixel offset
```

---

#### D. Anchor-free Detection

Mamba-YOLO menggunakan **anchor-free approach**:

**Traditional YOLO (anchor-based):**

```
Pre-defined anchor boxes:
- Small: 30×40, 40×50, 50×60
- Medium: 80×100, 100×120, 120×140
- Large: 200×250, 250×300, 300×350

Model predicts offset from anchors
```

**Mamba-YOLO (anchor-free):**

```
No pre-defined anchors!
Model predicts direct offset from cell center

For each cell:
- Predict: distance to 4 box edges (top, bottom, left, right)
- Bounding box = cell_center ± predicted_distances
```

**Keuntungan Anchor-free:**

- ✅ Lebih simple (tidak perlu tune anchor size)
- ✅ Lebih general (adaptif ke semua object size)
- ✅ Lebih efisien (tidak butuh anchor matching)

---

### 🔄 Forward Pass - Training vs Inference

#### Training Mode:

```python
def forward(self, x):
    # x = [P3, P4, P5]
    for i in range(self.nl):
        # Concat box prediction + class prediction
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # Output: (B, 64+80, H, W) = (B, 144, H, W)

    return x  # Return raw predictions untuk loss calculation
```

**Output Training:**

```python
[
    (B, 144, 80, 80),   # P3: 6400 predictions (small objects)
    (B, 144, 40, 40),   # P4: 1600 predictions (medium objects)
    (B, 144, 20, 20),   # P5: 400 predictions (large objects)
]
# Total: 8400 predictions per image
```

---

#### Inference Mode:

```python
def forward(self, x):
    # x = [P3, P4, P5]
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

    # Reshape predictions
    shape = x[0].shape  # (B, 144, H, W)
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    # x_cat: (B, 144, 8400)  [batch, outputs, total_anchors]

    # Split box and class predictions
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # box: (B, 64, 8400)
    # cls: (B, 80, 8400)

    # Generate anchors (cell centers)
    self.anchors, self.strides = make_anchors(x, self.stride, 0.5)

    # Decode bounding boxes
    dbox = self.decode_bboxes(self.dfl(box), self.anchors) * self.strides
    # dbox: (B, 4, 8400)  →  (x_center, y_center, width, height)

    # Apply sigmoid to class scores
    y = torch.cat((dbox, cls.sigmoid()), 1)
    # y: (B, 84, 8400)  →  [4 box coords + 80 class probs]

    return y
```

**Output Inference:**

```python
# Shape: (B, 84, 8400)
# 84 = 4 (box) + 80 (classes)
# 8400 = 6400 (P3) + 1600 (P4) + 400 (P5)

# Untuk setiap prediksi (8400 total):
prediction = [
    x_center,    # 0-640
    y_center,    # 0-640
    width,       # 0-640
    height,      # 0-640
    prob_class0, # 0-1 (person)
    prob_class1, # 0-1 (bicycle)
    ...
    prob_class79 # 0-1 (toothbrush)
]
```

---

#### E. Non-Maximum Suppression (NMS)

Setelah inference, ada **8400 predictions** per gambar. Banyak yang overlap!

**NMS Pipeline:**

```python
# 1. Filter by confidence threshold
predictions = predictions[max(class_probs) > 0.25]
# 8400 → ~100-500 predictions

# 2. Convert to corner format
boxes_xyxy = xywh2xyxy(boxes)  # (x_center, y_center, w, h) → (x1, y1, x2, y2)

# 3. NMS per class
for each class:
    # Sort by confidence
    sorted_boxes = sort_by_confidence(boxes)

    # Keep best, suppress overlap
    keep = []
    while sorted_boxes:
        best = sorted_boxes.pop(0)
        keep.append(best)

        # Remove boxes with IoU > threshold (0.45)
        sorted_boxes = [box for box in sorted_boxes
                        if IoU(best, box) < 0.45]

    final_boxes.append(keep)

# 4. Final output
# ~5-20 detections per image
```

**Visualisasi NMS:**

```
Before NMS (banyak overlap):
┌───────┐
│ 0.95  │
├───────┤
│ 0.92  │  → Semua detect "person" yang sama
├───────┤
│ 0.88  │
└───────┘

After NMS (hanya yang terbaik):
┌───────┐
│ 0.95  │  → Keep only best box!
└───────┘
```

---

### 📊 Ringkasan Output Head

**Training Output:**

```python
[
    (B, 144, 80, 80),   # P3 raw predictions
    (B, 144, 40, 40),   # P4 raw predictions
    (B, 144, 20, 20),   # P5 raw predictions
]
```

**Inference Output (before NMS):**

```python
(B, 84, 8400)  # [4 box + 80 classes] × 8400 anchors
```

**Final Output (after NMS):**

```python
# List of detections per image
[
    {
        'boxes': tensor([[x1, y1, x2, y2], ...]),  # (N, 4)
        'scores': tensor([0.95, 0.87, ...]),       # (N,)
        'classes': tensor([0, 2, ...])             # (N,)
    }
]
# N = jumlah deteksi final (~5-20)
```

---

## Alur Data Lengkap

Mari kita ikuti satu gambar dari input sampai output detection:

### 📸 Input: `image.jpg` (640×640×3)

```
┌──────────────────────────────────────────┐
│         Original RGB Image                │
│         [Person, Car, Dog in scene]       │
│         Shape: (1, 3, 640, 640)           │
└──────────────────────────────────────────┘
```

---

### 🏗️ Stage 1: BACKBONE (Feature Extraction)

```
Input: (1, 3, 640, 640)
    ↓
[SimpleStem] → (1, 128, 160, 160)  [P2: 1/4 resolution]
    ↓
[VSSBlock × 3] → Process features with Mamba
    ↓
[VisionClueMerge] → (1, 256, 80, 80)  [P3: 1/8 resolution] ← OUTPUT 1
    ↓
[VSSBlock × 3] → Enhanced features
    ↓
[VisionClueMerge] → (1, 512, 40, 40)  [P4: 1/16 resolution] ← OUTPUT 2
    ↓
[VSSBlock × 9] → Deep feature processing
    ↓
[VisionClueMerge] → (1, 1024, 20, 20)  [P5: 1/32 resolution]
    ↓
[VSSBlock × 3 + SPPF] → (1, 1024, 20, 20)  ← OUTPUT 3
```

**Apa yang Terjadi:**

- Gambar dikompresi dari `640×640` → `20×20` (32x lebih kecil)
- Channels meningkat dari `3` → `1024` (341x lebih banyak!)
- Setiap pixel di feature map merepresentasikan area `32×32` pixel di gambar asli
- **Selective Scan** menangkap pattern global + lokal

---

### 🔗 Stage 2: NECK (Feature Fusion)

#### Phase 1: Top-Down (Global → Local)

```
P5 (20×20×1024) [Deep semantic info]
    ↓ [Upsample]
(40×40×1024)
    ↓ [Concat P4]
(40×40×1536)
    ↓ [XSSBlock fusion]
P4' (40×40×512)  [Semantic + Spatial info]
    ↓ [Upsample]
(80×80×512)
    ↓ [Concat P3]
(80×80×768)
    ↓ [XSSBlock fusion]
P3' (80×80×256)  [Rich multi-scale features]
```

#### Phase 2: Bottom-Up (Local → Global)

```
P3' (80×80×256)
    ↓ [Downsample]
(40×40×256)
    ↓ [Concat P4']
(40×40×768)
    ↓ [XSSBlock fusion]
P4'' (40×40×512)  [Detail + Context]
    ↓ [Downsample]
(20×20×512)
    ↓ [Concat P5]
(20×20×1536)
    ↓ [XSSBlock fusion]
P5'' (20×20×1024)  [All information aggregated]
```

**Hasil:**

- P3'', P4'', P5'' sekarang punya informasi dari semua scale
- P3'' bisa detect objek kecil dengan konteks global
- P5'' bisa detect objek besar dengan detail spatial

---

### 🎯 Stage 3: HEAD (Prediction)

Untuk **setiap** level (P3'', P4'', P5''):

```
Feature Map
    ↓
┌────────────────────┬────────────────────┐
│   Box Branch       │   Class Branch     │
│   (cv2)            │   (cv3)            │
│                    │                    │
│ Conv 3×3           │ Conv 3×3           │
│ Conv 3×3           │ Conv 3×3           │
│ Conv 1×1           │ Conv 1×1           │
│    ↓               │    ↓               │
│ (H, W, 64)         │ (H, W, 80)         │
│ DFL distribution   │ Class scores       │
└────────────────────┴────────────────────┘
           ↓                    ↓
    [DFL decode]          [Sigmoid]
           ↓                    ↓
    Box coordinates      Class probabilities
    (x, y, w, h)         [0.0 - 1.0]
           ↓                    ↓
           └──────────┬──────────┘
                      ↓
              Final Predictions
          (84 values per anchor)
```

**Total Predictions:**

```
P3: 80×80 = 6,400 predictions  (small objects)
P4: 40×40 = 1,600 predictions  (medium objects)
P5: 20×20 =   400 predictions  (large objects)
────────────────────────────────
Total:      8,400 predictions per image
```

---

### 🧹 Stage 4: Post-Processing

```
8,400 Predictions
    ↓ [Confidence threshold > 0.25]
~500 Predictions
    ↓ [NMS - Remove overlap]
~10-20 Final Detections
    ↓
Output:
┌─────────────────────────────────────┐
│ Detection 1:                        │
│   Box: [x1=120, y1=80, x2=240, y2=320] │
│   Class: person                     │
│   Confidence: 0.95                  │
├─────────────────────────────────────┤
│ Detection 2:                        │
│   Box: [x1=300, y1=150, x2=500, y2=400] │
│   Class: car                        │
│   Confidence: 0.87                  │
├─────────────────────────────────────┤
│ Detection 3:                        │
│   Box: [x1=50, y1=400, x2=150, y2=550] │
│   Class: dog                        │
│   Confidence: 0.79                  │
└─────────────────────────────────────┘
```

---

### 📊 Information Flow Summary

```
INPUT (640×640×3)
     ↓ [Rich pixel information]
BACKBONE
     ↓ [Hierarchical features]
P3: Local details  (edges, textures)
P4: Mid-level info (parts, shapes)
P5: Global context (scenes, layout)
     ↓ [Multi-scale features]
NECK
     ↓ [Feature fusion]
P3'': Details + Context
P4'': Balanced information
P5'': Context + Details
     ↓ [Enhanced features]
HEAD
     ↓ [Task-specific prediction]
Boxes + Classes
     ↓ [Post-processing]
OUTPUT (Final detections)
```

---

## Perbandingan dengan YOLO Klasik

### YOLO v5/v8 (CNN-based)

```
┌────────────────────────────────────────┐
│ BACKBONE: CSPDarknet                   │
│ - Conv layers                          │
│ - Bottleneck blocks (ResNet-like)     │
│ - Complexity: O(N²) untuk spatial     │
├────────────────────────────────────────┤
│ NECK: PANet                            │
│ - Simple concatenation                 │
│ - CSP blocks for fusion                │
├────────────────────────────────────────┤
│ HEAD: Coupled Head (old) / Decoupled  │
│ - Box + Class prediction               │
└────────────────────────────────────────┘
```

### Mamba-YOLO (SSM-based)

```
┌────────────────────────────────────────┐
│ BACKBONE: ODMamba                      │
│ - VSSBlock (Selective Scan)            │
│ - Complexity: O(N) linear!             │
│ - Better long-range modeling           │
├────────────────────────────────────────┤
│ NECK: PAFPN with XSSBlock              │
│ - Mamba-based fusion                   │
│ - More efficient feature aggregation   │
├────────────────────────────────────────┤
│ HEAD: Decoupled Head (same)            │
│ - Box + Class prediction               │
└────────────────────────────────────────┘
```

### 📊 Benchmark Comparison

| Metric          | YOLO v8 | Mamba-YOLO | Improvement |
| --------------- | ------- | ---------- | ----------- |
| **Speed (FPS)** | 45      | 52         | +15% ⬆️     |
| **Parameters**  | 21.5M   | 19.1M      | -11% ⬇️     |
| **FLOPs**       | 52.4G   | 45.4G      | -13% ⬇️     |
| **mAP@50**      | 65.2%   | 66.5%      | +1.3% ⬆️    |
| **mAP@50-95**   | 48.5%   | 49.1%      | +0.6% ⬆️    |

**Kesimpulan:**

- ✅ **Lebih cepat** dengan parameter lebih sedikit
- ✅ **Lebih efisien** dalam computational cost
- ✅ **Lebih akurat** pada dataset COCO

---

## Kelebihan Arsitektur Mamba-YOLO

### 1. **Linear Complexity (O(N) vs O(N²))**

**Vision Transformer:**

```python
# Self-attention: Compare EVERY pixel with EVERY other pixel
For image 640×640 = 409,600 pixels:
  Attention Matrix = 409,600 × 409,600 = 167 billion comparisons!
  Memory: ~640 GB (FP32)
```

**Mamba-YOLO:**

```python
# Selective Scan: Sequential processing with gating
For image 640×640 = 409,600 pixels:
  Scan 4 directions × 409,600 = 1.6 million operations
  Memory: ~1.6 MB (FP32)
```

**Impact:**

- 100,000× less memory
- 10× faster processing
- Scale to higher resolution easily

---

### 2. **Better Long-Range Dependency**

**CNN:**

```
Receptive field terbatas
Butuh banyak layer untuk melihat area luas
┌─────┐
│ 3×3 │ Layer 1: Lihat 3×3 pixel
└─────┘
  ↓
┌─────┐
│ 7×7 │ Layer 2: Lihat 7×7 pixel
└─────┘
  ↓
┌──────┐
│11×11│ Layer 3: Lihat 11×11 pixel
└──────┘
```

**Mamba:**

```
Global receptive field dari layer pertama!
Selective scan bisa lihat SELURUH gambar
┌─────────────────┐
│ Full Image View │ Layer 1: Lihat 640×640 pixel!
└─────────────────┘
```

---

### 3. **Selective Information Processing**

**Traditional Approach:**

```
Process ALL information equally
┌──┬──┬──┬──┐
│🚗│  │🌳│  │  → All pixels processed sama
├──┼──┼──┼──┤
│  │👤│  │🏠│
└──┴──┴──┴──┘
```

**Mamba Approach:**

```
SELECTIVELY focus on important information
┌──┬──┬──┬──┐
│🚗│  │🌳│  │  → Focus: 🚗👤🏠 (objects)
├──┼──┼──┼──┤  → Ignore: background
│  │👤│  │🏠│
└──┴──┴──┴──┘
```

**Keuntungan:**

- Robust to clutter & noise
- Better feature representation
- Efficient computation

---

### 4. **Hardware-Friendly**

**Why Mamba is Fast:**

1. **Sequential Operations** (GPU friendly)

   ```
   Scan: pixel₁ → pixel₂ → pixel₃ → ...
   Can be parallelized across directions & channels
   ```

2. **Low Memory Footprint**

   ```
   No need to store attention matrix
   Only need hidden states (small)
   ```

3. **Custom CUDA Kernels**
   ```
   Selective Scan implemented in optimized CUDA
   Fused operations reduce memory bandwidth
   ```

---

## Kesimpulan

### 🎯 Rangkuman Arsitektur Mamba-YOLO

**Input → Output Pipeline:**

```
Image (640×640×3)
    ↓
┌─────────────────────────────────────────┐
│ BACKBONE: ODMamba                       │
│ • SimpleStem: Initial feature extraction│
│ • VSSBlock: Mamba-based processing      │
│ • VisionClueMerge: Efficient downsampling│
│ Output: P3, P4, P5 (multi-scale)       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ NECK: PAFPN                             │
│ • Top-down: Semantic info to all levels │
│ • Bottom-up: Spatial info to all levels │
│ • XSSBlock: Mamba-based fusion          │
│ Output: P3'', P4'', P5'' (enhanced)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ HEAD: Decoupled Head                    │
│ • Box branch: DFL-based bbox prediction │
│ • Class branch: Multi-class classification│
│ Output: 8,400 predictions per image     │
└─────────────────────────────────────────┘
    ↓
Post-processing (Confidence + NMS)
    ↓
Final Detections (~10-20 per image)
```

---

### 💡 Key Innovations

1. **Selective Scan Mechanism (SS2D)**

   - Linear complexity O(N)
   - Global receptive field
   - Selective information processing

2. **Mamba-based Feature Fusion (XSSBlock)**

   - Better multi-scale aggregation
   - Efficient computation
   - Preserve long-range dependencies

3. **Decoupled Head with DFL**
   - Task-specific optimization
   - Distribution-based bbox regression
   - Better accuracy

---

### 📈 Performance Summary

| Aspect          | Advantage                          |
| --------------- | ---------------------------------- |
| **Speed**       | +15% faster than YOLOv8            |
| **Efficiency**  | -13% FLOPs, -11% params            |
| **Accuracy**    | +1.3% mAP@50                       |
| **Scalability** | Linear complexity enables high-res |
| **Memory**      | Lower peak memory usage            |

---

### 🚀 When to Use Mamba-YOLO?

**✅ Best For:**

- Real-time applications (high FPS requirement)
- Edge devices (limited memory/compute)
- High-resolution images (benefits from O(N) complexity)
- Scenes with many objects (selective scan helps)
- Long-range dependency tasks

**❌ Maybe Not For:**

- Very small datasets (< 1000 images) - fine-tuning helps
- Extremely low-latency requirements (< 5ms) - use tiny models
- Simple detection tasks (YOLOv8-nano might be enough)

---

### 📚 Further Reading

**Papers:**

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-YOLO: SSMs-Based YOLO For Object Detection](https://arxiv.org/abs/2406.05835)

**Code:**

- [Official Mamba-YOLO Repository](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [Mamba Original Implementation](https://github.com/state-spaces/mamba)

---

**Happy Learning! 🎓**

Jika ada bagian yang masih kurang jelas atau butuh penjelasan lebih detail, jangan ragu untuk bertanya!
