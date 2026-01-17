# Mamba-YOLO Head Detection Setup Guide - Ubuntu Linux

Panduan lengkap untuk setup environment Mamba-YOLO di Ubuntu Linux untuk task **Head Detection**.

---

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA GPU (minimal GTX 1660 atau lebih tinggi)
- RAM minimal 16GB (recommended 32GB)
- Storage minimal 50GB free space

### Software Requirements
- Ubuntu 20.04/22.04/24.04 LTS 64-bit
- NVIDIA GPU Driver (versi terbaru)

---

## 🎮 Step 1: Install NVIDIA Driver

### 1.1 Check Current Driver
```bash
nvidia-smi
```

Jika muncul output GPU info, driver sudah terinstall. Skip ke Step 2.

### 1.2 Install NVIDIA Driver (jika belum ada)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install recommended driver
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Or install specific version
# sudo apt install nvidia-driver-545 -y

# Reboot
sudo reboot
```

### 1.3 Verify Driver Installation
```bash
nvidia-smi
```

---

## 🔧 Step 2: Install CUDA Toolkit 12.1

### 2.1 Download CUDA 12.1
```bash
# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download CUDA repository package (Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb

# Install repository
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb

# Add CUDA keyring
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update and install CUDA
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

**Note:** Untuk Ubuntu 20.04, ganti `ubuntu2204` dengan `ubuntu2004`.

### 2.2 Setup CUDA Environment Variables
```bash
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
source ~/.bashrc
```

### 2.3 Verify CUDA Installation
```bash
nvcc --version
```

Output yang diharapkan:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.1, V12.1.xxx
```

---

## 🐍 Step 3: Install Miniconda

### 3.1 Download dan Install Miniconda
```bash
# Download Miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh
```

Ikuti prompt instalasi:
- Press ENTER untuk review license
- Ketik `yes` untuk accept
- Press ENTER untuk default location
- Ketik `yes` untuk init conda

### 3.2 Reload Shell
```bash
source ~/.bashrc
```

### 3.3 Verifikasi Conda
```bash
conda --version
```

### 3.4 Update Conda (optional)
```bash
conda update -n base -c defaults conda -y
```

---

## 📥 Step 4: Clone/Setup Project

### 4.1 Install Git (jika belum ada)
```bash
sudo apt install git -y
```

### 4.2 Clone Repository atau Pindah ke Project Directory
```bash
# Jika clone dari GitHub:
# git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
# cd Mamba-YOLO

# Atau jika sudah ada project:
cd /path/to/Mamba-YOLO-TA-MARH
```

---

## 📦 Step 5: Setup Python Environment

### 5.1 Create Conda Environment
```bash
conda create -n mambayolo python=3.11 -y
conda activate mambayolo
```

### 5.2 Install Build Tools
```bash
sudo apt-get install -y build-essential cmake ninja-build
```

### 5.3 Install PyTorch 2.3.0 with CUDA 12.1
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### 5.4 Verify PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Output yang diharapkan:
```
PyTorch: 2.3.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3060
```

### 5.5 Install Base Dependencies
```bash
pip install seaborn thop timm einops packaging ninja wheel
pip install opencv-python pillow matplotlib pyyaml requests tqdm
pip install scipy pandas
```

---

## 🔧 Step 6: Build Selective Scan CUDA Extension

### 6.1 Navigate to selective_scan Directory
```bash
cd selective_scan
```

### 6.2 Build and Install
```bash
pip install -v . --no-build-isolation
```

**Note:** Proses ini memakan waktu 5-15 menit tergantung GPU dan CPU. Tunggu sampai selesai.

### 6.3 Verify Installation
```bash
python -c "import selective_scan_cuda_core; print('Selective scan installed successfully!')"
```

Jika berhasil, akan muncul:
```
Selective scan installed successfully!
```

### 6.4 Return to Project Root
```bash
cd ..
```

---

## 📥 Step 7: Install Ultralytics (Mamba-YOLO)

### 7.1 Install in Editable Mode
```bash
pip install -v -e .
```

### 7.2 Verify Installation
```bash
python -c "from ultralytics import YOLO; print('Mamba-YOLO installed successfully!')"
```

### 7.3 Check All Imports
```bash
python -c "
from ultralytics import YOLO
import torch
import selective_scan_cuda_core
print('✅ All components installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

---

## 🎯 Step 8: Prepare Head Detection Dataset

### 8.1 Create Dataset Directory Structure
```bash
mkdir -p datasets/head_detection/images/train
mkdir -p datasets/head_detection/images/val
mkdir -p datasets/head_detection/labels/train
mkdir -p datasets/head_detection/labels/val
```

### 8.2 Dataset Structure
```
Mamba-YOLO-TA-MARH/
├── datasets/
│   └── head_detection/
│       ├── images/
│       │   ├── train/          # Training images
│       │   │   ├── img001.jpg
│       │   │   ├── img002.jpg
│       │   │   └── ...
│       │   └── val/            # Validation images
│       │       ├── img001.jpg
│       │       └── ...
│       ├── labels/
│       │   ├── train/          # Training labels (YOLO format)
│       │   │   ├── img001.txt
│       │   │   ├── img002.txt
│       │   │   └── ...
│       │   └── val/            # Validation labels
│       │       ├── img001.txt
│       │       └── ...
│       └── head_detection.yaml # Dataset config
```

### 8.3 Create Dataset Configuration
```bash
cat > datasets/head_detection/head_detection.yaml << 'EOF'
# Head Detection Dataset Configuration

# Dataset path (absolute or relative to this file)
path: ../datasets/head_detection
train: images/train
val: images/val
test: # optional

# Classes
names:
  0: head

# Number of classes
nc: 1

# Image augmentation parameters (optional)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7    # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4    # image HSV-Value augmentation (fraction)
degrees: 10.0  # image rotation (+/- deg)
translate: 0.1 # image translation (+/- fraction)
scale: 0.5     # image scale (+/- gain)
shear: 0.0     # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0    # image flip up-down (probability)
fliplr: 0.5    # image flip left-right (probability)
mosaic: 1.0    # image mosaic (probability)
mixup: 0.0     # image mixup (probability)
EOF
```

### 8.4 Label Format (YOLO Format)
Setiap file `.txt` berisi satu baris per object:
```
class_id x_center y_center width height
```

Semua nilai normalized (0-1). Contoh `img001.txt`:
```
0 0.5 0.3 0.2 0.25
0 0.7 0.4 0.18 0.22
```

Penjelasan:
- `class_id`: 0 (head)
- `x_center`: 0.5 (center X di tengah image)
- `y_center`: 0.3 (center Y di 30% dari atas)
- `width`: 0.2 (width 20% dari image width)
- `height`: 0.25 (height 25% dari image height)

---

## 🏋️ Step 9: Training

### 9.1 Create Training Script
```bash
cat > train_head_detection.py << 'EOF'
#!/usr/bin/env python3
"""
Mamba-YOLO Head Detection Training Script
"""

from ultralytics import YOLO
import torch

def main():
    # Print system info
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)
    
    # Load Mamba-YOLO model
    # Options: Mamba-YOLO-T.yaml (small/fast), Mamba-YOLO-M.yaml (medium), Mamba-YOLO-L.yaml (large)
    model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
    
    # Optional: Load pretrained weights
    # model = YOLO('path/to/pretrained/mamba-yolo-t.pt')
    
    # Training configuration
    results = model.train(
        # Dataset
        data='datasets/head_detection/head_detection.yaml',
        
        # Training parameters
        epochs=100,              # Number of epochs
        batch=16,                # Batch size (adjust based on GPU memory)
        imgsz=640,              # Input image size
        
        # Optimization
        optimizer='AdamW',       # Optimizer: SGD, Adam, AdamW
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final learning rate (lr0 * lrf)
        momentum=0.937,         # SGD momentum/Adam beta1
        weight_decay=0.0005,    # Optimizer weight decay
        warmup_epochs=3.0,      # Warmup epochs
        warmup_momentum=0.8,    # Warmup initial momentum
        
        # Device
        device=0,               # GPU device (0, 1, 2, ...) or 'cpu'
        workers=8,              # Number of dataloader workers
        
        # Saving
        project='runs/train',   # Project directory
        name='mamba_yolo_head_detection',  # Experiment name
        exist_ok=False,         # Overwrite existing experiment
        save=True,              # Save checkpoints
        save_period=10,         # Save checkpoint every x epochs
        
        # Validation
        val=True,               # Validate during training
        patience=20,            # Early stopping patience (epochs)
        
        # Logging
        verbose=True,           # Verbose output
        plots=True,             # Save plots
        
        # Advanced
        cache=False,            # Cache images (True/'ram'/'disk')
        rect=False,             # Rectangular training
        cos_lr=False,           # Cosine LR scheduler
        close_mosaic=10,        # Disable mosaic augmentation for final epochs
        amp=True,               # Automatic Mixed Precision training
        fraction=1.0,           # Dataset fraction to use
        profile=False,          # Profile model
        
        # Multi-GPU (uncomment if using multiple GPUs)
        # device=[0, 1],        # Multiple GPUs
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best weights: runs/train/mamba_yolo_head_detection/weights/best.pt")
    print(f"Last weights: runs/train/mamba_yolo_head_detection/weights/last.pt")
    print(f"Results: runs/train/mamba_yolo_head_detection/")
    print("=" * 60)

if __name__ == '__main__':
    main()
EOF

chmod +x train_head_detection.py
```

### 9.2 Adjust Batch Size (jika GPU memory kecil)
```bash
# For 6GB GPU: batch=8
# For 8GB GPU: batch=16
# For 12GB+ GPU: batch=32
```

### 9.3 Start Training
```bash
python train_head_detection.py
```

### 9.4 Monitor Training
Training progress akan tersimpan di:
```
runs/train/mamba_yolo_head_detection/
├── weights/
│   ├── best.pt          # Best checkpoint
│   ├── last.pt          # Last checkpoint
│   └── epoch*.pt        # Periodic checkpoints
├── results.png          # Training curves
├── results.csv          # Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── PR_curve.png
└── args.yaml            # Training arguments
```

### 9.5 Monitor with TensorBoard (optional)
```bash
pip install tensorboard
tensorboard --logdir runs/train
```

Buka browser: `http://localhost:6006`

---

## 🔍 Step 10: Inference/Detection

### 10.1 Create Detection Script
```bash
cat > detect_head.py << 'EOF'
#!/usr/bin/env python3
"""
Mamba-YOLO Head Detection Inference Script
"""

from ultralytics import YOLO
import torch
import sys

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python detect_head.py <source>")
        print("  <source>: path to image/video/folder or 0 for webcam")
        print("Example: python detect_head.py path/to/image.jpg")
        sys.exit(1)
    
    source = sys.argv[1]
    
    # Load trained model
    model_path = 'runs/train/mamba_yolo_head_detection/weights/best.pt'
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {source}")
    
    # Run inference
    results = model.predict(
        source=source,          # Image/video path, folder, or 0 for webcam
        conf=0.25,              # Confidence threshold
        iou=0.45,               # NMS IOU threshold
        imgsz=640,              # Input size
        device=0,               # GPU device (0, 1, 2, ...) or 'cpu'
        max_det=300,            # Maximum detections per image
        
        # Visualization
        show=False,             # Show results (set True for display)
        save=True,              # Save results
        save_txt=True,          # Save results as .txt
        save_conf=True,         # Save confidence scores
        save_crop=False,        # Save cropped detections
        
        # Output
        project='runs/detect',  # Save directory
        name='head_detection_results',  # Experiment name
        exist_ok=True,          # Overwrite existing
        
        # Advanced
        visualize=False,        # Visualize features
        augment=False,          # Augmented inference
        agnostic_nms=False,     # Class-agnostic NMS
        classes=None,           # Filter by class (None = all)
        retina_masks=False,     # High-res masks
        line_width=2,           # Bounding box thickness
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"Image {i+1}: Detected {len(boxes)} heads")
        
        # Print individual detections
        for j, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"  Head {j+1}: confidence={conf:.2f}, bbox=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
    
    print("=" * 60)
    print(f"Results saved to: runs/detect/head_detection_results/")
    print("=" * 60)

if __name__ == '__main__':
    main()
EOF

chmod +x detect_head.py
```

### 10.2 Run Detection on Image
```bash
python detect_head.py path/to/test/image.jpg
```

### 10.3 Run Detection on Video
```bash
python detect_head.py path/to/test/video.mp4
```

### 10.4 Run Detection on Folder
```bash
python detect_head.py path/to/test/images/
```

### 10.5 Run Detection on Webcam
```bash
python detect_head.py 0
```

### 10.6 View Results
Results akan tersimpan di:
```
runs/detect/head_detection_results/
├── image1.jpg           # Annotated images
├── image1.txt           # Detection results (YOLO format)
└── ...
```

---

## 📊 Step 11: Evaluation

### 11.1 Create Evaluation Script
```bash
cat > evaluate_model.py << 'EOF'
#!/usr/bin/env python3
"""
Mamba-YOLO Head Detection Evaluation Script
"""

from ultralytics import YOLO
import torch

def main():
    # Load trained model
    model_path = 'runs/train/mamba_yolo_head_detection/weights/best.pt'
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Evaluate on validation set
    print("Running evaluation on validation set...")
    metrics = model.val(
        data='datasets/head_detection/head_detection.yaml',
        batch=16,
        imgsz=640,
        device=0,
        workers=8,
        plots=True,
        save_json=True,
        project='runs/val',
        name='head_detection_eval',
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"F1-Score:  {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6):.4f}")
    print("=" * 60)
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(f"Class 'head':")
    print(f"  AP50:     {metrics.box.ap50[0]:.4f}")
    print(f"  AP50-95:  {metrics.box.ap[0]:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()
EOF

chmod +x evaluate_model.py
```

### 11.2 Run Evaluation
```bash
python evaluate_model.py
```

### 11.3 View Evaluation Results
```
runs/val/head_detection_eval/
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── PR_curve.png
└── results.csv
```

---

## 📈 Step 12: Export Model

### 12.1 Export to Different Formats
```bash
cat > export_model.py << 'EOF'
#!/usr/bin/env python3
"""
Mamba-YOLO Model Export Script
"""

from ultralytics import YOLO

def main():
    # Load trained model
    model = YOLO('runs/train/mamba_yolo_head_detection/weights/best.pt')
    
    # Export to ONNX
    print("Exporting to ONNX...")
    model.export(format='onnx', dynamic=True, simplify=True)
    
    # Export to TensorRT (requires TensorRT installed)
    # print("Exporting to TensorRT...")
    # model.export(format='engine', device=0, half=True)
    
    # Export to CoreML (macOS only)
    # print("Exporting to CoreML...")
    # model.export(format='coreml')
    
    # Export to TFLite
    # print("Exporting to TFLite...")
    # model.export(format='tflite')
    
    print("\nExport completed!")
    print("ONNX model: runs/train/mamba_yolo_head_detection/weights/best.onnx")

if __name__ == '__main__':
    main()
EOF

chmod +x export_model.py
```

### 12.2 Run Export
```bash
python export_model.py
```

---

## 🎨 Step 13: Visualize Results

### 13.1 Install Visualization Tools
```bash
pip install matplotlib seaborn pandas
```

### 13.2 Plot Training Curves
```bash
cat > plot_results.py << 'EOF'
#!/usr/bin/env python3
"""
Plot training results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load results
results = pd.read_csv('runs/train/mamba_yolo_head_detection/results.csv')
results.columns = results.columns.str.strip()

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mamba-YOLO Head Detection Training Results', fontsize=16)

# Plot metrics
metrics = [
    ('train/box_loss', 'Train Box Loss'),
    ('val/box_loss', 'Val Box Loss'),
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP@0.5'),
    ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
]

for ax, (col, title) in zip(axes.flatten(), metrics):
    if col in results.columns:
        ax.plot(results['epoch'], results[col], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('runs/train/mamba_yolo_head_detection/training_curves.png', dpi=300)
print("Training curves saved to: runs/train/mamba_yolo_head_detection/training_curves.png")

EOF

chmod +x plot_results.py
python plot_results.py
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA out of memory
**Solution:** Reduce batch size
```python
batch=8  # or even batch=4
```

### Issue 2: Selective scan build failed
**Solution:** Check dependencies
```bash
sudo apt-get install -y build-essential cmake ninja-build
pip install ninja packaging wheel
cd selective_scan
pip install -v . --no-build-isolation --force-reinstall
```

### Issue 3: CUDA not detected
**Solution:** Check NVIDIA driver and CUDA
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: Slow training
**Solutions:**
- Increase workers: `workers=16`
- Use AMP: `amp=True`
- Cache images: `cache='ram'` or `cache='disk'`
- Use multiple GPUs: `device=[0, 1]`

### Issue 5: Low mAP
**Solutions:**
- Train longer: `epochs=200`
- Use pretrained weights
- Increase image size: `imgsz=1280`
- Adjust augmentation parameters
- Check dataset labels
- Balance dataset (equal train/val split)

### Issue 6: Dataset not found
**Solution:** Check paths in YAML
```bash
# Use absolute path
path: /absolute/path/to/datasets/head_detection

# Or relative to project root
path: datasets/head_detection
```

---

## 📚 Advanced Tips

### 1. Transfer Learning with Pretrained Weights
```python
# Download COCO pretrained weights (if available)
model = YOLO('mamba-yolo-t-coco.pt')

# Fine-tune on head detection
results = model.train(
    data='datasets/head_detection/head_detection.yaml',
    epochs=50,
    freeze=10,  # Freeze first 10 layers
)
```

### 2. Multi-GPU Training
```python
results = model.train(
    data='datasets/head_detection/head_detection.yaml',
    device=[0, 1, 2, 3],  # Use 4 GPUs
    batch=64,  # Total batch size (16 per GPU)
)
```

### 3. Resume Training
```bash
python train_head_detection.py --resume runs/train/mamba_yolo_head_detection/weights/last.pt
```

### 4. Hyperparameter Tuning
```bash
# Auto-tune hyperparameters
model.tune(
    data='datasets/head_detection/head_detection.yaml',
    epochs=30,
    iterations=300,
    optimizer='AdamW',
    plots=True,
    save=True,
)
```

### 5. Optimize for Small Heads
```python
# Use larger input size
imgsz=1280

# Lower confidence threshold
conf=0.15

# Lower NMS IOU for crowded scenes
iou=0.3
```

### 6. Data Augmentation for Head Detection
```yaml
# In head_detection.yaml
mosaic: 1.0      # Enable mosaic augmentation
mixup: 0.1       # Small mixup
copy_paste: 0.1  # Copy-paste augmentation
degrees: 10      # Rotation
scale: 0.5       # Scaling
fliplr: 0.5      # Horizontal flip
```

---

## 🔄 Common Workflows

### Workflow 1: Quick Test on New Images
```bash
# Single image
python detect_head.py test_image.jpg

# Batch of images
python detect_head.py test_images/

# Results in runs/detect/head_detection_results/
```

### Workflow 2: Retrain with More Data
```bash
# Add new images to datasets/head_detection/images/train/
# Add new labels to datasets/head_detection/labels/train/

# Retrain from checkpoint
python train_head_detection.py
```

### Workflow 3: Deploy on Edge Device
```bash
# Export to ONNX
python export_model.py

# Export to TensorRT (for NVIDIA Jetson)
model.export(format='engine', device=0, half=True)

# Export to TFLite (for mobile)
model.export(format='tflite', imgsz=320)
```

---

## 📊 Model Comparison

| Model | Size | Params | FLOPs | mAP50 (COCO) | Speed (V100) |
|-------|------|--------|-------|--------------|--------------|
| Mamba-YOLO-T | 640 | 5.8M | 12.3G | 61.2 | 2.1ms |
| Mamba-YOLO-M | 640 | 19.1M | 45.2G | 66.5 | 3.8ms |
| Mamba-YOLO-L | 640 | 57.6M | 124.8G | 69.8 | 6.5ms |

**Recommendation for Head Detection:**
- **Real-time applications (>30 FPS):** Mamba-YOLO-T
- **Balance accuracy/speed:** Mamba-YOLO-M  
- **Maximum accuracy:** Mamba-YOLO-L

---

## 🚀 Performance Optimization

### 1. FP16 Inference (faster)
```python
model = YOLO('best.pt')
model.predict(source='image.jpg', half=True)  # Use FP16
```

### 2. Batch Inference
```python
# Process multiple images at once
results = model.predict(source='images/', batch=16)
```

### 3. TensorRT Optimization
```bash
# Export to TensorRT (requires TensorRT installed)
model.export(format='engine', device=0, half=True, workspace=4)

# Use TensorRT model
model = YOLO('best.engine')
```

### 4. Profile Model
```bash
python -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.profile()
"
```

---

## 📖 Useful Commands

```bash
# Activate environment
conda activate mambayolo

# Deactivate environment
conda deactivate

# Check GPU usage
nvidia-smi

# Monitor GPU continuously
watch -n 1 nvidia-smi

# Check disk usage
df -h

# Check memory usage
free -h

# Kill training process
pkill -f train_head_detection.py

# Find model checkpoints
find runs/ -name "*.pt"

# Tensorboard
tensorboard --logdir runs/train

# Clean up old runs
rm -rf runs/train/mamba_yolo_head_detection*
```

---

## ✅ Setup Checklist

- [ ] Ubuntu installed
- [ ] NVIDIA driver working (`nvidia-smi`)
- [ ] CUDA 12.1 installed (`nvcc --version`)
- [ ] Miniconda installed
- [ ] Python 3.11 environment created
- [ ] PyTorch 2.3.0 + CUDA installed
- [ ] PyTorch detects GPU (`torch.cuda.is_available()`)
- [ ] Build tools installed (gcc, cmake, ninja)
- [ ] Selective scan compiled successfully
- [ ] Ultralytics installed
- [ ] Dataset prepared (images + labels in YOLO format)
- [ ] Dataset YAML configured
- [ ] Training script created
- [ ] First training run successful
- [ ] Model evaluation completed
- [ ] Inference script working

---

## 🆘 Getting Help

### Check System Status
```bash
./check_system.sh
```

Create check script:
```bash
cat > check_system.sh << 'EOF'
#!/bin/bash
echo "=== System Check ==="
echo ""
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
echo ""
echo "CUDA Version:"
nvcc --version | grep release
echo ""
echo "Python:"
python --version
echo ""
echo "PyTorch:"
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""
echo "Selective Scan:"
python -c "import selective_scan_cuda_core; print('OK')" 2>&1
echo ""
echo "Disk Usage:"
df -h | grep -E '^/dev'
echo ""
echo "Memory:"
free -h
EOF

chmod +x check_system.sh
./check_system.sh
```

---

## 📄 License

Mamba-YOLO is licensed under AGPL-3.0 License.

---

## 🎯 Quick Start Summary

```bash
# 1. Setup environment
conda create -n mambayolo python=3.11 -y
conda activate mambayolo

# 2. Install PyTorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install seaborn thop timm einops packaging ninja wheel opencv-python pillow matplotlib pyyaml requests tqdm

# 4. Build selective_scan
cd selective_scan && pip install -v . --no-build-isolation && cd ..

# 5. Install Mamba-YOLO
pip install -v -e .

# 6. Train
python train_head_detection.py

# 7. Detect
python detect_head.py path/to/image.jpg

# 8. Evaluate
python evaluate_model.py
```

---

**Good luck with your Head Detection project! 🎯👤**
