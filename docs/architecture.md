# Kiến trúc Dự án Brain Tumor Detection

> Tài liệu mô tả kiến trúc kỹ thuật đầy đủ của hệ thống phân loại khối u não từ ảnh MRI,
> bao gồm mô hình CNN, pipeline dữ liệu, quy trình huấn luyện, module Grad-CAM và giao diện người dùng.

---

## 1. Tổng quan Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BrainTumorDetection                          │
│                                                                     │
│  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐ │
│  │  Data Layer  │────▶│ Training Layer  │────▶│  Inference Layer │ │
│  │              │     │                 │     │                  │ │
│  │ Dataset 1    │     │ Trainer (AMP)   │     │ Predictor        │ │
│  │ Dataset 2    │     │ Metrics         │     │ GradCAM          │ │
│  │ DataModule   │     │ LR Scheduler    │     │ Preprocessing    │ │
│  └──────────────┘     └─────────────────┘     └──────────────────┘ │
│           │                    │                        │           │
│           │           ┌────────▼──────┐                │           │
│           │           │  BaselineCNN  │                │           │
│           │           │  (5 blocks)   │                │           │
│           │           └────────┬──────┘                │           │
│           │                    │                        │           │
│           │           ┌────────▼──────────────────────▼─────────┐ │
│           │           │              GUI Layer                    │ │
│           │           │  MainWindow  │  Controller  │  Widgets   │ │
│           │           └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mô hình CNN (Model Layer)

**File:** [`models/cnn.py`](../models/cnn.py)

### `BaselineCNN`

Mô hình CNN xây dựng hoàn toàn từ đầu (from scratch) bằng PyTorch — **không sử dụng Pretrained Weights hay Transfer Learning**.

#### Kiến trúc chi tiết

```
Input: (Batch, 3, H, W)  — ảnh RGB
        │
        ▼
┌─────────────────────────────────────┐
│  Block 1: Conv2d(3→32) + BN + ReLU  │  → MaxPool2d(2×2)
│  Block 2: Conv2d(32→64) + BN + ReLU │  → MaxPool2d(2×2)
│  Block 3: Conv2d(64→128) + BN + ReLU│  → MaxPool2d(2×2)
│  Block 4: Conv2d(128→256)+ BN + ReLU│  → MaxPool2d(2×2)
│  Block 5: Conv2d(256→512)+ BN + ReLU│  → MaxPool2d(2×2)
└─────────────────────────────────────┘
        │
        ▼
 AdaptiveAvgPool2d(1×1)   →  Flatten
        │
        ▼
┌─────────────────────────────────────┐
│  Linear(512 → 256) + ReLU           │
│  Dropout(p=0.5)                     │
│  Linear(256 → num_classes=4)        │
└─────────────────────────────────────┘
        │
        ▼
Output: Logits (Batch, 4)
```

#### Tham số mô hình

| Tham số | Giá trị |
|---|---|
| Kernel size (tất cả Conv) | 3×3, padding=1 |
| Activation | ReLU (inplace) |
| Normalization | BatchNorm2d sau mỗi Conv |
| Downsampling | MaxPool2d(2, 2) sau mỗi Block |
| Pooling cuối | AdaptiveAvgPool2d(1,1) |
| Regularization | Dropout(0.5) |
| Output | 4 classes (Glioma, Meningioma, No Tumor, Pituitary) |
| Target Layer (Grad-CAM) | `self.features[-4]` — Conv Block 5 |

---

## 3. Pipeline Dữ liệu (Data Layer)

### 3.1. Dataset Classes

**File:** [`datasets/brain_mri_dataset.py`](../datasets/brain_mri_dataset.py)

| Class | Mô tả |
|---|---|
| `BrainMRIDataset` | Dataset có nhãn cho Dataset 1. Duyệt theo 4 thư mục class (`glioma`, `meningioma`, `notumor`, `pituitary`). |
| `BrainMRIUnlabeledDataset` | Dataset không nhãn cho Dataset 2 (Br35H). Đệ quy tìm ảnh trong `yes/` và `no/`. Trả về `(image, path)`. |

### 3.2. Data Augmentation

**File:** [`datasets/transforms.py`](../datasets/transforms.py)

| Phase | Transforms |
|---|---|
| **Training** | Resize → RandomHorizontalFlip → RandomRotation(±15°) → ColorJitter → ToTensor → Normalize |
| **Validation/Test** | Resize → ToTensor → Normalize |

> **Normalize:** `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet stats)

### 3.3. DataModule

**File:** [`datasets/data_module.py`](../datasets/data_module.py)

| Phương thức | Chức năng |
|---|---|
| `get_train_dataloader()` | Load Dataset 1 Training. Hỗ trợ **Stratified Sampling** (cắt theo `subset_ratio` từng class). |
| `get_val_dataloader()` | Load Dataset 1 Testing (toàn bộ 100%, không cắt). |
| `get_unlabeled_dataloader()` | Load Dataset 2 (Br35H), phục vụ cho giai đoạn Semi-Supervised Learning. |

### 3.4. Cấu trúc Dữ liệu

```
data/
├── dataset1/
│   ├── Training/
│   │   ├── glioma/         (≈1400 ảnh)
│   │   ├── meningioma/     (≈1400 ảnh)
│   │   ├── notumor/        (≈1400 ảnh)
│   │   └── pituitary/      (≈1400 ảnh)
│   └── Testing/            (cùng cấu trúc)
│
└── dataset2/               (Br35H)
    ├── yes/                (ảnh có khối u - dùng cho SSL)
    ├── no/                 (ảnh không có khối u - dùng cho SSL)
    └── Br35H-Mask-RCNN/
        ├── TRAIN/
        ├── VAL/
        ├── TEST/
        └── annotations_all.json   (Ground truth polygon masks)
```

---

## 4. Quản lý Cấu hình (Config Layer)

**File:** [`configs/config.py`](../configs/config.py)  
**File YAML:** [`experiments/baseline.yaml`](../experiments/baseline.yaml)

Mọi hyperparameter được điều khiển hoàn toàn qua file YAML — **không cần sửa code**.

### Cấu trúc Config (Dataclasses)

```
Config
├── experiment_name: str
├── project_name: str
├── seed: int
├── DataConfig
│   ├── dataset1_train_path
│   ├── dataset1_test_path
│   ├── dataset2_path
│   └── num_workers
├── ModelConfig
│   ├── name: "custom_cnn"
│   ├── num_classes: 4
│   └── pretrained: false
├── TrainingConfig
│   ├── learning_rate: 0.0001
│   ├── weight_decay: 0.0001
│   └── ssl_enabled: false
├── ModeConfig (development)
│   ├── enabled: true
│   ├── subset_ratio: 0.7
│   ├── image_size: 128
│   ├── batch_size: 8
│   └── epochs: 10
└── ModeConfig (research)
    ├── enabled: false
    ├── subset_ratio: 1.0
    ├── image_size: 224
    ├── batch_size: 32
    └── epochs: 30
```

### Hai chế độ chạy

| Chế độ | Mục đích | Dữ liệu | Image Size |
|---|---|---|---|
| **Development** | Debug nhanh | 70% mỗi class | 128×128 |
| **Research** | Kết quả luận văn | 100% | 224×224 |

---

## 5. Quy trình Huấn luyện (Training Layer)

**File:** [`training/trainer.py`](../training/trainer.py) — **File:** [`training/metrics.py`](../training/metrics.py)  
**Entry point:** [`main.py`](../main.py)

### Luồng huấn luyện

```
main.py
  │
  ├─ Load Config (YAML)
  ├─ Set Seed (reproducibility)
  ├─ Init DataModule → train_loader, val_loader
  ├─ Init BaselineCNN
  ├─ Init Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  ├─ Init Scheduler: CosineAnnealingLR(T_max=epochs)
  ├─ Init Trainer
  │
  └─ Training Loop (per epoch):
       ├─ trainer.train_epoch(train_loader, optimizer)   ← AMP + tqdm
       ├─ trainer.validate(val_loader)
       ├─ scheduler.step()
       ├─ [Nếu val_loss cải thiện] → Lưu best_model.pth
       └─ [Nếu không cải thiện ≥ patience] → Early Stopping
```

### Kỹ thuật huấn luyện

| Kỹ thuật | Chi tiết |
|---|---|
| **Loss Function** | `CrossEntropyLoss` |
| **Optimizer** | `AdamW` — lr=0.0001, weight_decay=0.0001 |
| **LR Scheduler** | `CosineAnnealingLR` |
| **AMP** | `torch.cuda.amp.GradScaler` — chỉ kích hoạt khi có CUDA |
| **Early Stopping** | Patience = 7 epochs mặc định |
| **Progress Bar** | `tqdm` hiển thị loss real-time mỗi batch |
| **Checkpoint** | Lưu `saved_model/best_model.pth` khi val_loss giảm |

### Metrics đánh giá

**File:** [`training/metrics.py`](../training/metrics.py)

Tính sau mỗi epoch, báo cáo cho cả Train và Val:

| Metric | Cách tính |
|---|---|
| Accuracy | `accuracy_score` |
| Precision | `precision_score(average='macro')` |
| Recall | `recall_score(average='macro')` |
| F1-Score | `f1_score(average='macro')` |
| Loss | `CrossEntropyLoss` trung bình trên batch |

---

## 6. Explainable AI — Grad-CAM (Inference Layer)

**File:** [`inference/predictor.py`](../inference/predictor.py)

### Nguyên lý Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) sử dụng gradient của lớp Conv cuối cùng để xác định vùng ảnh mà mô hình "tập trung" khi đưa ra quyết định.

```
Input image
    │
    ▼
BaselineCNN.forward()
    │
    ├──── Forward Hook → lưu Feature Maps (activations)
    ├──── Backward Hook → lưu Gradients
    │
    ▼
Predicted class logit → backward()
    │
    ▼
weights = GlobalAvgPool(gradients)     # (C, 1, 1)
grad_cam = ReLU(Σ weights × activations)  # weighted sum of feature maps
    │
    ▼
Normalize → Resize về kích thước ảnh gốc
    │
    ▼
Apply JET Colormap → Overlay lên ảnh gốc (alpha=0.4)
    │
    ▼
[Nếu không phải "No Tumor"] → Fit Ellipse quanh vùng activation > 0.5
```

### `GradCAM` Class

| Thuộc tính/Phương thức | Mô tả |
|---|---|
| `target_layer` | `model.features[-4]` (Conv2d của Block 5) |
| `save_activation()` | Forward hook — lưu feature map |
| `save_gradient()` | Backward hook (`register_full_backward_hook`) |
| `generate(x, class_idx)` | Trả về `(heatmap_2d, class_idx)` |
| `release()` | Xóa hooks sau khi dùng xong |

### `Predictor` Class

| Thuộc tính/Phương thức | Mô tả |
|---|---|
| `_load_model(path)` | Load checkpoint `.pth`, map to device |
| `predict(image_path)` | Trả về dict đầy đủ kết quả |
| **Output dict** | `class`, `class_idx`, `confidence`, `probabilities`, `inference_time_ms`, `gradcam_heatmap` |

---

## 7. Giao diện Người dùng (GUI Layer)

**Entry point:** [`gui_app.py`](../gui_app.py)

### Sơ đồ Component

```
gui_app.py
    │
    └─▶ MainWindow (main_window.py)
            │
            ├─▶ Controller (controller.py)   ── bridge đến Predictor
            │       └─▶ Predictor (inference/predictor.py)
            │
            ├─▶ Left Panel
            │       ├─ SectionFrame "Model"     → Load .pth
            │       ├─ SectionFrame "MRI Image" → Chọn & hiển thị ảnh
            │       └─ Buttons: [Run Prediction] | [Export PNG]
            │
            ├─▶ Right Panel
            │       ├─ PredictionResultWidget   → Tên class + confidence
            │       ├─ ConfidenceBar × 4        → Thanh % từng class
            │       └─ SectionFrame "Grad-CAM"  → Heatmap overlay
            │
            └─▶ InferenceThread (QThread)       → Chạy inference bất đồng bộ
```

### Luồng tương tác GUI

```
User chọn .pth  →  Controller.load_model()  →  Predictor(model_path)
User chọn ảnh   →  Controller.set_image()   →  Lưu image_path
User bấm Predict →  InferenceThread.start()  →  (chạy background)
                       │
                       └─▶ Predictor.predict(image_path)
                                │
                                ├─ Forward pass → Softmax probabilities
                                ├─ GradCAM.generate() → heatmap PIL image
                                └─ cv2: resize, colormap, ellipse overlay
                       │
                       └─▶ result_ready signal → MainWindow._on_result_ready()
                                                    ├─ update PredictionResultWidget
                                                    ├─ update ConfidenceBar × 4
                                                    └─ display gradcam_heatmap
```

### Widgets

**File:** [`gui/widgets.py`](../gui/widgets.py)

| Widget | Mô tả |
|---|---|
| `ConfidenceBar` | Thanh tiến trình màu sắc hiển thị % confidence mỗi class |
| `PredictionResultWidget` | Hiển thị tên class dự đoán + confidence + thời gian inference |
| `SectionFrame` | Panel có tiêu đề, dùng làm container cho các nhóm widget |

---

## 8. Thư viện Phụ thuộc

**File:** [`requirements.txt`](../requirements.txt)

| Thư viện | Mục đích |
|---|---|
| `torch >= 2.0.0` | Deep Learning framework |
| `torchvision >= 0.15.0` | Dataset utilities + transforms |
| `PyQt5 >= 5.15` | Desktop GUI |
| `opencv-python >= 4.8` | Heatmap coloring, contour fitting, ellipse drawing |
| `Pillow >= 10.0` | Image I/O |
| `scikit-learn >= 1.0` | Metrics (Accuracy, F1, ...) |
| `PyYAML >= 6.0` | Đọc file cấu hình YAML |
| `numpy >= 1.24` | Array operations |
| `tqdm >= 4.0` | Progress bar khi training |
| `pytest >= 8.4` | Unit testing |
| `matplotlib >= 3.8` | Visualization (đồ thị kết quả) |
| `seaborn >= 0.12` | Confusion matrix visualization |

---

## 9. Cấu trúc thư mục đầy đủ

```
BrainTumorDetection/
│
├── main.py                         # Entry point huấn luyện
├── gui_app.py                      # Entry point GUI Desktop
├── requirements.txt
│
├── configs/
│   └── config.py                   # Dataclass cấu hình
│
├── experiments/
│   └── baseline.yaml               # Điều chỉnh hyperparameter tại đây
│
├── models/
│   └── cnn.py                      # BaselineCNN (5-block from scratch)
│
├── datasets/
│   ├── brain_mri_dataset.py        # BrainMRIDataset + BrainMRIUnlabeledDataset
│   ├── data_module.py              # DataModule (Stratified Sampling + DataLoaders)
│   └── transforms.py              # Train/Val augmentation pipelines
│
├── training/
│   ├── trainer.py                  # Trainer (AMP, tqdm, checkpoint)
│   └── metrics.py                  # Accuracy, F1, Precision, Recall
│
├── inference/
│   ├── predictor.py                # GradCAM + Predictor (inference engine)
│   └── preprocessing.py           # Image loading và preprocess cho Predictor
│
├── gui/
│   ├── main_window.py              # PyQt5 layout + event handlers
│   ├── controller.py               # Bridge GUI ↔ Predictor
│   └── widgets.py                  # ConfidenceBar, PredictionResultWidget, SectionFrame
│
├── data/
│   ├── dataset1/                   # Dataset 4-class có nhãn
│   └── dataset2/                   # Br35H: yes/no + Polygon annotations
│
├── saved_model/
│   └── best_model.pth              # Checkpoint tốt nhất
│
├── docs/                           # Tài liệu dự án
├── evaluation/                     # Scripts đánh giá (Phase 5)
├── ssl/                            # Semi-Supervised Learning (Phase 4)
└── utils/
    ├── logger.py
    └── seed.py
```
