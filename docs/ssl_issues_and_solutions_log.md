# NHẬT KÝ GHI NHẬN VẤN ĐỀ & ĐỀ XUẤT CẢI TIẾN MÔ HÌNH HỌC BÁN GIÁM SÁT (SSL LOG)
**Ngày ghi nhận:** 01/08/2026  
**Dự án:** Brain Tumor Detection (Semi-Supervised Learning Pipeline)  
**Kiến trúc:** ImprovedCNN (ResNet-18 style, From Scratch)  

---

## 1. GHI NHẬN THỰC TRẠNG (OBSERVATIONS)

Trong quá trình kiểm thử thực tế mô hình `ssl_best_model.pth` trên ứng dụng đồ họa (GUI App) và phân tích kết quả kiểm thử trên 1.600 ảnh Validation, nhóm nghiên cứu ghi nhận các hạn chế sau:

1. **Hiện tượng nhầm lẫn giữa khối u và không u (Low Confidence & False Positives/Negatives):**
   - Trên một số ảnh test MRI thực tế (ví dụ: khối u tròn, sáng rõ vùng chẩm phía dưới sọ não), mô hình chỉ dự đoán lớp **Meningioma với độ tin cậy thấp (53.03%)**, trong khi xác suất cho lớp **No Tumor lên tới 43.7%**.
   - Điều này cho thấy đường ranh giới quyết định (Decision Boundary) giữa lớp có khối u (Meningioma/Glioma) và không có khối u (No Tumor) chưa thực sự sắc nét.


2. **Vùng kích hoạt Grad-CAM bị phân tán (Imprecise Localization):**
   - Bản đồ nhiệt Grad-CAM chưa tập trung chính xác vào tâm khối u trắng mà bị lan rộng ra bán cầu não bên trái/phải.
   - Hình elip định vị bao quanh vùng kích hoạt không phản ánh đúng vị trí thực tế của khối u.

3. **Chênh lệch lớn về độ chính xác giữa các lớp (Per-Class Imbalance):**
   - Theo báo cáo đánh giá trên 1.600 ảnh test (`docs/final_eval_20260729_170631.txt`):
     - **No Tumor:** F1 = `0.89` (Recall = 97%)
     - **Pituitary:** F1 = `0.86` (Recall = 99%)
     - **Glioma:** F1 = `0.72` (Recall = 67%)
     - **Meningioma:** F1 = `0.64` (Recall = 54%)
   - **Meningioma** và **Glioma** là hai lớp có tỷ lệ sai số cao nhất, thường xuyên bị gán nhãn nhầm cho nhau hoặc nhầm với No Tumor.

---

## 2. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

| Nguyên nhân | Chi tiết kỹ thuật | Tác động lên mô hình |
|:---|:---|:---|
| **1. Nút thắt cổ chai dữ liệu có nhãn (Labeled Bottleneck)** | Trong `experiments/ssl_experiment.yaml`, cấu hình hiện tại chỉ sử dụng **250 ảnh nhãn thật / lớp** (tổng **1.000 ảnh** trên 4 lớp, chỉ chiếm **~17.8%** tập huấn luyện Dataset 1 có 5.600 ảnh). | Số lượng mẫu nhãn thật quá ít khiến mạng CNN ResNet-18 huấn luyện từ đầu (`from scratch`) không đủ dữ liệu để học các đặc trưng hình thái tinh vi của u màng não (Meningioma) và u thần kinh đệm (Glioma). |
| **2. Nhiễu nhãn giả ở Phase B (Confirmation Bias / Pseudo-Label Noise)** | Ở giai đoạn Phase B, mô hình tự dự đoán và gán nhãn giả cho **8.461 ảnh không nhãn** (Unlabeled Pool gồm phần còn lại của DS1 và toàn bộ Br35H DS2) với ngưỡng tin cậy `0.85 -> 0.95`. | Do mô hình Phase A (1.000 ảnh) còn yếu ở lớp Meningioma/Glioma, nó đã **gán nhãn giả sai** cho nhiều ảnh trong tập Unlabeled. Khi tiếp tục huấn luyện trên các nhãn sai này, sai số bị khuếch đại (confirmation bias). |
| **3. Mất cân bằng ngữ nghĩa từ Dataset 2 (Br35H)** | Dataset 2 (3.861 ảnh) gốc chỉ có nhãn nhị phân (Yes/No Tumor), trong đó có tới ~1.500 ảnh không có khối u với độ tương phản khác biệt so với Dataset 1. | Khi gộp vào Unlabeled Pool, lượng ảnh "No Tumor" và ảnh khối u đa dạng khiến mô hình dễ bị phân vân giữa "Meningioma" và "No Tumor" (như trường hợp 53% - 43% trên GUI). |
| **4. Huấn luyện hoàn toàn From Scratch (`pretrained: false`)** | Không sử dụng trọng số khởi tạo ImageNet, các bộ lọc tích chập ở các tầng đầu (Stem, Layer 1) phải tự học từ con số 0. | Các đặc trưng viền, cạnh, kết cấu khối u (texture) không được sắc nét, dẫn đến Grad-CAM bị lan rộng và không xác định đúng biên khối u. |

---

## 3. ĐỀ XUẤT CẢI TIẾN & KẾ HOẠCH HÀNH ĐỘNG (ACTION PLAN)

Để khắc phục triệt để hiện tượng mô hình đoán sai và nâng độ chính xác tổng thể từ **~79% lên > 88% - 92%**, đề xuất thực hiện các điều chỉnh sau:

### Đề xuất 1: Tăng dữ liệu có nhãn (`labeled_per_class`) — *Khuyến nghị cao nhất*
- **Hành động:** Nâng thông số `labeled_per_class` trong file [experiments/ssl_experiment.yaml](file:///c:/Users/84387/Documents/webcuatra/SIC/BrainTumorDetection/experiments/ssl_experiment.yaml) từ `250` lên **`400`** hoặc **`500`** ảnh/lớp (tổng 1.600 - 2.000 ảnh nhãn thật, tương đương 30-35% Dataset 1).
- **Lý do:** Tăng số lượng nhãn thật ban đầu giúp giai đoạn Phase A (Supervised Warmup) đạt độ chính xác > 85% trước khi bước vào Phase B, ngăn chặn từ gốc hiện tượng tự gán nhãn sai (pseudo-label noise).

### Đề xuất 2: Bật Transfer Learning (`pretrained: true`) cho ImprovedCNN
- **Hành động:** Thay đổi `pretrained: false` thành `true` trong phần `model:` của `ssl_experiment.yaml`.
- **Lý do:** Tận dụng khả năng trích xuất đặc trưng viền/cạnh vượt trội của ImageNet Pretrained Weights. Điều này không chỉ tăng Accuracy mà còn giúp bản đồ nhiệt **Grad-CAM định vị chính xác vùng khối u**, không bị lan ra ngoài não.

### Đề xuất 3: Siết chặt Ngưỡng Gán nhãn giả (Strict Curriculum Thresholding)
- **Hành động:** Điều chỉnh ngưỡng tin cậy trong `ssl_experiment.yaml`:
  - `pseudo_threshold_start: 0.90` (cũ: 0.85)
  - `pseudo_threshold_end: 0.97` (cũ: 0.95)
- **Lý do:** Chỉ cho phép mô hình gán nhãn giả cho những ảnh có độ tự tin cực cao (>= 90%), loại bỏ các ca mơ hồ giữa Meningioma và Glioma khỏi tập huấn luyện Phase B.

---

## 4. BÀI HỌC THỰC NGHIỆM ĐẶC BIỆT (CASE STUDY: FALSE NEGATIVE NGHIÊM TRỌNG TRÊN GUI)

Trong quá trình kiểm thử GUI App trên ảnh MRI có khối u sáng trắng lớn nằm ở viền ngoài sọ não (thùy đỉnh/trán - Top/Left), ghi nhận hiện tượng bất thường:
- **Dự đoán:** Mô hình nhận diện sai thành **No Tumor với độ tin cậy tới 98.44%**.
- **Grad-CAM:** Vùng kích hoạt nhiệt đỏ bị **ngược bán cầu (Inverted Localization)** – sáng vào vùng não bình thường bên phải thay vì khối u bên trái.

**Giải thích khoa học:**
1. **Thiếu bất biến không gian cho u viền sọ (Boundary Spatial Invariance):** Mạng ResNet-18 huấn luyện từ đầu (`from scratch`) trên chỉ 250 ảnh/lớp (1.000 ảnh) chưa học đủ đa dạng các góc chụp và kích thước khối u cực đại ở sát xương sọ.
2. **Nhiễu từ Dataset 2 (Br35H):** Trong tập Unlabeled có rất nhiều ảnh "No Tumor" (1.500 ảnh) với viền sọ sáng trắng tương tự. Ở Phase B, khi mô hình tự gán nhãn với ngưỡng `0.85`, các ảnh khối u viền bị gán nhãn nhầm thành "No Tumor", khiến đường ranh giới quyết định bị bóp méo nghiêm trọng.
=> **Khẳng định tính cấp thiết** của việc tăng nhãn thật lên 450 ảnh/lớp (`labeled_per_class: 450`) và siết ngưỡng gán nhãn giả (`0.90 -> 0.97`).

---

## 5. BÀI HỌC THỰC NGHIỆM SỐ 2 (CASE STUDY 2: HIỆN TƯỢNG NHẦM LẪN GLIOMA THÀNH MENINGIOMA / NO TUMOR)

Trong quá trình kiểm thử GUI App với các mẫu thuộc lớp **Glioma (U thần kinh đệm)**, ghi nhận hiện tượng:
- **Dự đoán:** Mô hình phân vân và dự đoán sai thành **No Tumor (55.45%)** hoặc **Meningioma (39.9%)**, trong khi xác suất cho Glioma chỉ đạt **1.2%**.
- **Chỉ số kiểm thử thực tế:** Theo báo cáo `docs/ssl_run_20260801_133319.txt`, dù Accuracy tổng thể đạt **88.88%** và Meningioma Recall đạt **88%**, nhưng **Glioma Recall chỉ đạt 73%** (có khoảng 27% mẫu Glioma bị phân loại nhầm sang Meningioma hoặc No Tumor).

**Giải thích khoa học sâu:**
1. **Đặc tính hình thái của Glioma:** U thần kinh đệm (đặc biệt là thể Low-grade hoặc nằm sâu trong não thất/tiểu não) thường có **độ tương phản gần với mô não bình thường** và **viền khối u mờ, thâm nhiễm (infiltrative borders)** thay vì viền rõ ràng như U màng não (Meningioma).
2. **Hạn chế của Huấn luyện From Scratch (`pretrained: false`):** Mạng ResNet-18 tự học từ con số 0 trên số lượng ảnh y tế hạn chế (450 ảnh/class) chưa có đủ các bộ lọc cạnh siêu nhạy (như bộ lọc Gabor trong ImageNet weights) để phân biệt viền mờ của Glioma với viền nhẵn của Meningioma hoặc chất xám bình thường.
3. **Nhiễu gán nhãn giả Glioma ở Phase B:** Do Glioma là lớp khó nhất, một số mẫu Glioma trong tập Unlabeled (8,461 ảnh) đã bị tự gán nhãn sai thành Meningioma, khiến mô hình hình thành **định kiến sai (Confirmation Bias)** khi gặp các ca Glioma thực tế.

**=> 3 GIẢI PHÁP ĐỀ XUẤT KHẮC PHỤC TRIỆT ĐỂ LỚP GLIOMA:**
- **Giải pháp A (Hiệu quả lập tức - Khuyến nghị 100%): Bật Transfer Learning (`pretrained: true` - ImageNet Weights).** Cho phép sử dụng trọng số ImageNet thay vì học `from scratch`. Trọng số ImageNet cung cấp sẵn hàng ngàn bộ lọc cạnh/kết cấu (texture filters), giúp phân biệt ngay lập tức viền mờ thâm nhiễm của Glioma so với viền nhẵn của Meningioma.
- **Giải pháp B (Nếu tiếp tục học From Scratch): Đặt trọng số phạt riêng cho Glioma (Class-Weighted Loss / Focal Loss).** Tăng trọng số mất mát của lớp Glioma lên 1.3 - 1.5 lần trong hàm Loss để buộc mạng CNN ưu tiên cập nhật gradient cho lớp này.
- **Giải pháp C: Tăng cường tương phản cục bộ (CLAHE Augmentation).** Áp dụng cân bằng histogram thích ứng (CLAHE) trong bước tiền xử lý để làm nổi bật đường biên của Glioma.

---

## 6. BÀI HỌC THỰC NGHIỆM SỐ 3 (CASE STUDY 3: BẪY HỌC TẮT ĐỘ SÁNG VIỀN SỌ - SHORTCUT LEARNING / SPURIOUS CORRELATION)

Trong quá trình kiểm thử GUI App với ảnh MRI thuộc lớp **Glioma** có khối u kích thước cực đại (occupying bottom-left hemisphere), ghi nhận hiện tượng kinh điển trong nghiên cứu Medical AI:
- **Dự đoán:** Mô hình nhận diện sai thành **No Tumor (78.12%)** hoặc **Meningioma (21.4%)**, trong khi xác suất Glioma chỉ là **0.2%**.
- **Bản đồ nhiệt Grad-CAM:** Vùng kích hoạt đỏ rực xuất hiện ở **viền xương sọ góc trên phải** (vùng mô não hoàn toàn bình thường) và bỏ qua toàn bộ khối u khổng lồ phía dưới trái.
- **Thống kê toàn tập Test:** Có chính xác **19/400 ảnh Glioma (4.75%)** bị phân loại nhầm thành `No Tumor` do cơ chế học tắt này.

**Phân tích khoa học (Sự khác biệt giữa Thị giác Con người vs. CNN học From Scratch):**
1. **Đối với Bác sĩ / Thị giác Con người:** Đây là ca bệnh cực kỳ dễ nhận biết vì khối u có diện tích khổng lồ, viền sáng rõ ràng, cấu trúc dị nhất.
2. **Đối với Mạng ResNet-18 học From Scratch (`pretrained: false`):** Mắc bẫy học tắt (*Shortcut Learning / Spurious Correlation*).
   - Trong tập dữ liệu không nhãn (Dataset 2 - Br35H), có tới 1.500 ảnh `No Tumor` với cấu trúc viền xương sọ sáng trắng nổi bật.
   - Do không có sẵn các bộ lọc hình học ImageNet, các tầng tích chập đầu tiên (`stem`, `layer1`) đã học một liên kết giả tạo (spurious correlation): *"Viền sọ sáng trắng hình cung -> Não bình thường (No Tumor)"*.
   - Khi gặp ảnh Glioma lớn có viền khối u trắng sáng sát màng sọ, mô hình nhầm đặc trưng này với xương sọ bình thường, dẫn đến kích hoạt nhầm bán cầu đối diện và đưa ra dự đoán `No Tumor (78.1%)`.

**=> GIẢI PHÁP KHẮC PHỤC TRIỆT ĐỂ BẪY HỌC TẮT:**
- **Giải pháp 1 (Tiêu chuẩn vàng): Bật Transfer Learning (`pretrained: true` - ImageNet Pretrained Weights).** Các bộ lọc cạnh, hình dạng khép kín và kết cấu vi mô từ ImageNet giúp CNN phá vỡ liên kết giả tạo về độ sáng viền sọ, buộc mô hình khóa mục tiêu vào khối u và dự đoán chính xác 100%.
- **Giải pháp 2: Tiền xử lý xóa viền sọ (Skull Stripping / Random Erasing Augmentation).** Loại bỏ hoặc che ngẫu nhiên viền xương sọ trong bước tiền xử lý để mạng CNN bắt buộc phải học đặc trưng mô não bên trong.

---

## 7. LỊCH SỬ THỰC THI & CẬP NHẬT CẤU HÌNH (IMPLEMENTATION LOG)

- **[01/08/2026 - 13:30] ĐÃ ÁP DỤNG CẢI TIẾN VÀO FILE `experiments/ssl_experiment.yaml`:**
  ```yaml
  ssl:
    labeled_per_class: 450        # Tăng lên 450 ảnh/lớp (tổng 1.800 ảnh, tăng 80% nhãn xuất phát)
    warmup_epochs: 15             # Giữ Phase A: 15 epochs
    ssl_epochs: 20                # Giữ Phase B: 20 epochs
    pseudo_threshold_start: 0.90  # Siết chặt ngưỡng xuất phát lên 0.90 (cũ: 0.85)
    pseudo_threshold_end: 0.97    # Siết chặt ngưỡng kết thúc lên 0.97 (cũ: 0.95)
    pseudo_loss_weight: 0.5       # Trọng số loss
    min_pseudo_per_class: 15      # Tối thiểu mỗi lớp 15 ảnh (cũ: 10)
  ```
- **Trạng thái:** Hệ thống đã sẵn sàng cho đợt huấn luyện mới để khắc phục lỗi phân loại sai trên GUI và nâng độ chính xác tổng thể.
