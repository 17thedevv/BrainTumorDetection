# Brain Tumor Detection

Phân loại khối u não từ ảnh MRI sử dụng Custom CNN xây dựng hoàn toàn từ đầu bằng PyTorch (không pretrained, không transfer learning).

---

## Yêu cầu nghiên cứu

- **Custom CNN** — xây dựng từ đầu bằng PyTorch (`nn.Conv2d`, `nn.BatchNorm2d`, …)
- **Không sử dụng** pretrained models / transfer learning
- **Dataset 1** (4-class supervised): Glioma · Meningioma · No Tumor · Pituitary
- **Dataset 2** (Br35H) — sử dụng ở Phase 3 (Semi-Supervised Learning)
- **Grad-CAM** — sẽ được triển khai ở Phase 3

---

## Cài đặt & Hướng dẫn Chạy dự án

**Bước 1: Clone repo và khởi tạo môi trường ảo (Khuyến nghị)**
```powershell
git clone https://github.com/17thedevv/BrainTumorDetection.git
cd BrainTumorDetection
python -m venv venv
.\venv\Scripts\activate
```

**Bước 2: Cài đặt thư viện (Dependencies)**
```powershell
pip install -r requirements.txt
```
> *Lưu ý: Nếu lúc chạy giao diện bạn gặp lỗi thiếu PyQt5, hãy chạy thêm lệnh: `pip install PyQt5`*

**Bước 3: Chạy Giao diện Ứng dụng (GUI)**
Để mở ứng dụng Desktop hỗ trợ load ảnh, dự đoán và sinh Grad-CAM Heatmap:
```powershell
python gui_app.py
```

**Bước 4: Chạy Huấn luyện Mô hình (Training)**
Dự án được điều khiển thông qua cấu hình trong file YAML. Để bắt đầu huấn luyện từ đầu:
```powershell
python main.py --config experiments/baseline.yaml
```

---

## Cấu trúc dữ liệu

```
data/
├── dataset1/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       └── (same structure)
└── dataset2/     # Br35H — dùng cho SSL (Phase 3)
```

---

## Chế độ Development & Research

Chế độ được điều khiển hoàn toàn qua file `experiments/baseline.yaml` — **không sửa code**.

### Development Mode (debug nhanh)

```yaml
development:
  enabled: true       # BẬT Development Mode
  subset_ratio: 0.2   # Dùng 20% dữ liệu mỗi class (280/1400 ảnh)
  image_size: 128     # Ảnh nhỏ hơn để tăng tốc
  batch_size: 8
  epochs: 5
```

Kết quả: `280 × 4 = 1120` ảnh train, validation/test **giữ nguyên 100%**.

### Research Mode (thí nghiệm chính thức)

```yaml
development:
  enabled: false      # TẮT Development Mode → tự động dùng Research

research:
  enabled: false
  subset_ratio: 1.0   # 100% Dataset 1 (1400 ảnh/class)
  image_size: 224
  batch_size: 32
  epochs: 30
```

> **Luận văn PHẢI dùng Research Mode** với toàn bộ dữ liệu và kiến trúc CNN không thay đổi.

---

## Training Workflow

```powershell
# Development Mode (nhanh, debug)
python main.py --config experiments/baseline.yaml

# Research Mode: sửa baseline.yaml: development.enabled = false
python main.py --config experiments/baseline.yaml
```

**Tính năng huấn luyện:**
- Automatic Mixed Precision (AMP) khi có CUDA
- tqdm progress bar với ETA mỗi batch
- Hiển thị thời gian huấn luyện mỗi epoch
- Cosine Annealing LR Scheduler
- Early Stopping (patience = 7 epochs mặc định)
- Chỉ lưu `saved_model/best_model.pth`
- Stratified sampling trong Development Mode

---

## SSL v2 — Huấn luyện Bán Giám sát Nâng cao (Semi-Supervised Learning)

Pipeline SSL v2 tích hợp **10 giải pháp** tăng accuracy so với v1 (88.88% → dự kiến 93–95%):

| Giải pháp | Mô tả |
|---|---|
| **[A1] Data Augmentation mạnh** | RandomCrop, VerticalFlip, Affine, GaussianBlur, **RandomErasing** (chống shortcut learning) |
| **[A2] Mixup Training** | Trộn 2 sample theo tỷ lệ ngẫu nhiên (α=0.2) — regularization mạnh |
| **[B1] SE-Block** | Squeeze-and-Excitation channel attention trong mỗi ResidualBlock |
| **[B2] Classifier Head** | 3-layer FC + BatchNorm1d + Dropout(0.5, 0.4) |
| **[B3] Label Smoothing** | `CrossEntropyLoss(label_smoothing=0.1)` — giảm overfitting |
| **[C1] LR Warmup** | LinearLR warmup 3 epochs → CosineAnnealingWarmRestarts(T₀=10) |
| **[C3] Gradient Clipping** | `clip_grad_norm_(max_norm=1.0)` — ổn định training |
| **[D1] Confidence-Weighted** | Pseudo-label được weight theo confidence score |

### Chạy SSL v2

```powershell
# Kích hoạt môi trường ảo
.\venv\Scripts\activate

# Chạy SSL v2 pipeline (15 warmup + 25 SSL = 40 epochs)
python main_ssl.py --config experiments/ssl_experiment_v2.yaml
```

### Chạy SSL v2 — bỏ qua Phase A (nếu đã có checkpoint warmup)

```powershell
python main_ssl.py --config experiments/ssl_experiment_v2.yaml --skip_warmup --checkpoint saved_model/ssl_best_model.pth
```

### Đánh giá model sau khi train

```powershell
python eval_best_model.py --config experiments/ssl_experiment_v2.yaml
```

### Cấu hình SSL v2

Tất cả cấu hình nằm trong `experiments/ssl_experiment_v2.yaml`:

```yaml
ssl:
  labeled_per_class: 450        # 450 × 4 = 1800 ảnh có nhãn
  warmup_epochs: 15             # Phase A: supervised warmup
  ssl_epochs: 25                # Phase B: pseudo-labeling
  pseudo_threshold_start: 0.90  # Ngưỡng confidence ban đầu
  pseudo_threshold_end: 0.97    # Ngưỡng confidence cuối (curriculum)
```

### So sánh v1 vs v2

| Tiêu chí | v1 (01/08/2026) | v2 (dự kiến) |
|---|---|---|
| Accuracy | 88.88% | 93–95% |
| Augmentations | 3 ops | 11 ops |
| Model | ResNet-18 (~11M) | ResNet-18 + SE (~11.66M) |
| Classifier | 2-layer FC | 3-layer FC + BN |
| Label Smoothing | ❌ | ✅ 0.1 |
| Mixup | ❌ | ✅ α=0.2 |
| Grad Clipping | ❌ | ✅ max_norm=1.0 |
| LR Warmup | ❌ | ✅ 3 epochs |
| Pseudo-label weighting | Equal | Confidence-weighted |

---

## Testing Workflow

```powershell
pytest
```

---

## GUI — Inference Desktop App

```powershell
python gui_app.py
```

**Chức năng:**
1. Load model `.pth` đã huấn luyện
2. Chọn ảnh MRI
3. Hiển thị ảnh MRI gốc
4. Dự đoán 1 trong 4 classes
5. Hiển thị confidence của từng class (thanh tiến trình màu)
6. Hiển thị thời gian inference (ms)
7. Export kết quả ra PNG
8. Grad-CAM overlay với viền khoanh vùng tumor

> GUI **không chứa business logic** — toàn bộ logic inference nằm ở `inference/predictor.py`.

---

## Cấu trúc dự án

```
BrainTumorDetection/
├── main.py                     # Entry point huấn luyện baseline
├── main_ssl.py                 # Entry point SSL pipeline (v1 & v2)
├── eval_best_model.py          # Đánh giá model + xuất biểu đồ
├── gui_app.py                  # Entry point GUI
├── experiments/
│   ├── baseline.yaml           # Config baseline supervised
│   ├── ssl_experiment.yaml     # Config SSL v1
│   └── ssl_experiment_v2.yaml  # Config SSL v2 (10 cải tiến)
├── configs/
│   └── config.py               # Dataclasses cấu hình
├── models/
│   └── cnn.py                  # BaselineCNN + ImprovedCNN (SE-Block)
├── datasets/
│   ├── brain_mri_dataset.py    # Dataset classes (Labeled, Unlabeled, Pseudo)
│   ├── data_module.py          # Stratified Sampling + DataLoaders
│   └── transforms.py           # Augmentation pipeline (v2: 11 ops)
├── training/
│   ├── trainer.py              # Baseline trainer (AMP, Label Smoothing)
│   ├── ssl_trainer.py          # SSL trainer (Mixup, GradClip, Confidence-Weighted)
│   └── metrics.py              # Accuracy, Precision, Recall, F1
├── inference/
│   ├── predictor.py            # Inference + Grad-CAM engine
│   └── preprocessing.py        # Image loading + transform
├── evaluation/
│   └── evaluate.py             # Standalone evaluation script
├── visualization/
│   ├── plot_metrics.py         # Training curves, confusion matrix
│   └── results/                # Saved plots
├── gui/
│   ├── main_window.py          # PyQt5 UI layout
│   ├── controller.py           # Bridge UI <-> inference
│   └── widgets.py              # Reusable UI components
├── docs/                       # Run logs & reports
├── data/
│   ├── dataset1/               # Training + Testing (4-class)
│   └── dataset2/               # Br35H (unlabeled, SSL)
└── saved_model/
    └── ssl_best_model.pth      # Best model checkpoint
```
