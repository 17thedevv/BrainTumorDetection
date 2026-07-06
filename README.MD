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
8. Grad-CAM overlay *(Phase 3)*

> GUI **không chứa business logic** — toàn bộ logic inference nằm ở `inference/predictor.py`.

---

## Cấu trúc dự án

```
BrainTumorDetection/
├── main.py                     # Entry point huấn luyện
├── gui_app.py                  # Entry point GUI
├── experiments/
│   └── baseline.yaml           # Điều chỉnh mode ở đây
├── configs/
│   └── config.py               # Dataclasses cấu hình
├── models/
│   └── cnn.py                  # Custom CNN (from scratch)
├── datasets/
│   ├── brain_mri_dataset.py
│   ├── data_module.py          # Stratified Sampling + DataLoaders
│   └── transforms.py
├── training/
│   ├── trainer.py              # AMP, tqdm, time tracking
│   └── metrics.py
├── inference/
│   ├── predictor.py            # Inference engine
│   └── preprocessing.py       # Image loading + transform
├── gui/
│   ├── main_window.py          # PyQt5 UI layout
│   ├── controller.py           # Bridge UI <-> inference
│   └── widgets.py              # Reusable UI components
├── data/
│   ├── dataset1/               # Training + Testing
│   └── dataset2/               # Br35H (SSL Phase 3)
└── saved_model/
    └── best_model.pth
```
