# Brain Tumor Detection

## Mô tả

Dự án dùng TensorFlow để huấn luyện và kiểm thử mô hình phân loại u não từ ảnh MRI. Dữ liệu được lưu trong `data/Dataset` với hai thư mục con `Training` và `Testing`.

## Cài đặt

```powershell
cd c:\Users\84387\Documents\webcuatra\SIC\BrainTumorDetection

# Cài dependencies
python -m pip install -r requirements.txt
```

## Chạy Desktop App (PyQt5 GUI)

```powershell
python gui_app.py
```

Đây là ứng dụng Desktop GUI chuyên nghiệp hỗ trợ:
- Kéo thả (Drag & drop) ảnh MRI.
- Xem kết quả phân tích khối u (Glioma, Meningioma, Pituitary, No Tumor) trực quan.
- Chạy nền (Threaded Inference) không bị đơ giật ứng dụng.

## Chạy huấn luyện

```powershell
python train.py --dataset-dir data/Dataset
```

## Chạy test

```powershell
pytest src/test/test_model.py
```

## Cấu trúc dự án

```
BrainTumorDetection/
├── gui_app.py                      # 🖥️ PyQt5 Desktop App (GUI)
├── train.py                        # Entry point huấn luyện
├── requirements.txt
├── data/
│   └── Dataset/
│       ├── Training/               # glioma, meningioma, notumor, pituitary
│       └── Testing/
├── saved_model/
│   └── brain_tumor_detector/
│       └── best_model.h5           # ~155 MB
└── src/
    ├── model/brain_tumor_model.py  # CNN architecture
    ├── training/train.py           # Training pipeline
    ├── utils/
    │   ├── data_loader.py
    │   └── preprocessing.py
    └── test/
        ├── test_model.py
        └── test_preprocessing.py
```
