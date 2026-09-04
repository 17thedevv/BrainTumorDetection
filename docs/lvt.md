# BỘ CÂU HỎI BẢO VỆ DỰ ÁN DÀNH CHO: LƯƠNG VĂN TRÀ
Vai trò: Thu thập dữ liệu, Xây dựng mô hình ImprovedCNN kết hợp Semi-Supervised Learning (Cốt lõi thuật toán AI).

---

### PHẦN 1: DỮ LIỆU (DATA COLLECTION & PROCESSING)

**Q1: Dữ liệu của nhóm lấy từ đâu? Tại sao lại gộp 2 Dataset?**
> **Trả lời:** Dạ, chúng em kết hợp 2 bộ dữ liệu MRI công khai. 
> - **Dataset 1:** Có đầy đủ 4 nhãn (Glioma, Meningioma, Pituitary, No Tumor), được chia thành tập Train và Test độc lập.
> - **Dataset 2 (Br35H):** Chỉ có nhãn "Yes/No" (Có u hay không có u), không phân loại 4 lớp nên không thể train supervised trực tiếp.
> - **Cách giải quyết:** Chúng em biến toàn bộ Dataset 2 (và một phần lớn Dataset 1) thành tập **Unlabeled (Không nhãn)** để dùng cho thuật toán Semi-Supervised Learning, mô phỏng đúng thực tế y tế là "rất nhiều ảnh MRI nhưng rất ít ảnh được bác sĩ gán nhãn chi tiết 4 loại".

**Q2: Tập dữ liệu huấn luyện được chia (split) như thế nào?**
> **Trả lời:** Em cấu hình tham số `labeled_per_class = 450`. Nghĩa là em chỉ lấy đúng **1,800 ảnh có nhãn** (450 ảnh x 4 loại khối u) để làm tập "Nhãn thật" (Labeled Data). Toàn bộ **7,661 ảnh còn lại** được đẩy vào tập "Không nhãn" (Unlabeled Data). Tập Test độc lập gồm **1,600 ảnh**.

---

### PHẦN 2: MÔ HÌNH CẢI TIẾN (IMPROVED CNN)

**Q3: Mô hình "ImprovedCNN" của em có gì khác biệt so với một mạng CNN cơ bản (Baseline)?**
> **Trả lời:** Dạ, thay vì dùng CNN truyền thống xếp chồng các lớp Conv, em đã tự xây dựng `ImprovedCNN` lấy cảm hứng từ cấu trúc **ResNet-18** kết hợp với **Squeeze-and-Excitation (SE) Block**:
> - **Residual Connection (Kết nối tắt):** Giải quyết triệt để lỗi mất mát đạo hàm (Vanishing Gradient) khi mạng sâu.
> - **SE-Block (Attention):** Đây là điểm "ăn tiền" nhất. Nó giúp mạng tự học được Channel nào quan trọng. Đối với ảnh MRI, SE-Block giúp mô hình **tập trung vào các vùng chứa khối u** và "bỏ qua" các vùng viền xương sọ không mang ý nghĩa chẩn đoán.

**Q4: Tại sao em lại tự train từ đầu (Train from scratch) mà không dùng Pretrained Models (như ResNet50 ImageNet) cho dễ đạt độ chính xác cao?**
> **Trả lời:** Dạ, nếu dùng Pretrained ImageNet thì độ chính xác dễ dàng vượt 95%. Nhưng mục tiêu cốt lõi trong phần việc của em là **đánh giá sự hiệu quả của thuật toán Semi-Supervised Learning (SSL)**. Nếu dùng Pretrained, em sẽ không thể phân định rõ độ chính xác cao là do SSL hay do trọng số có sẵn của ImageNet. Train "from scratch" giúp em chứng minh minh bạch rằng: Nhờ cơ chế SSL, độ chính xác đã **tăng từ 79.13% lên 88.88% (nhờ học thêm ảnh không nhãn)**.

---

### PHẦN 3: THUẬT TOÁN SEMI-SUPERVISED LEARNING (SSL)

**Q5: Quá trình SSL của em diễn ra như thế nào?**
> **Trả lời:** Em chia làm 2 giai đoạn:
> - **Phase A (Warmup - 15 epochs):** Train mô hình giám sát thuần túy trên 1,800 ảnh có nhãn để mô hình học "đường ranh giới quyết định" cơ bản.
> - **Phase B (Pseudo-Labeling - 20 epochs):** Ở mỗi epoch, em lấy mô hình đi dự đoán trên 7,661 ảnh không nhãn. Những ảnh nào mô hình dự đoán có độ tự tin (Confidence) cực cao sẽ được gán nhãn giả (Pseudo-labels) và đem vào train chung với 1,800 ảnh gốc ở lượt tiếp theo.

**Q6: Làm sao em tránh được việc gán "nhãn giả sai" (Pseudo-label Noise) khiến mô hình học bậy?**
> **Trả lời:** Em áp dụng cơ chế **Curriculum Learning (Siết chặt ngưỡng tin cậy)**. 
> Em tăng Threshold từ `0.90` nhích dần lên `0.97` qua các epoch. Đồng thời em thêm tham số `min_pseudo_per_class = 15` để đảm bảo 4 loại khối u được lấy nhãn giả cân bằng, không bị lớp No Tumor lấn át.

---

### PHẦN 4: ĐÁNH GIÁ CHỈ SỐ (METRICS)

**Q7: Tại sao em lại dùng cả F1-Score mà không dùng mỗi Accuracy (Độ chính xác) để báo cáo?**
> **Trả lời:** Dạ, trong bài toán y tế, Accuracy rất dễ gây đánh lừa nếu dữ liệu bị mất cân bằng (Imbalanced data). Dùng **F1-Score** (trung bình điều hòa giữa Precision và Recall) giúp em đánh giá mô hình khách quan hơn, đảm bảo mô hình không bị thiên vị vào một lớp cụ thể nào, đặc biệt là không được bỏ sót lớp U màng não (Meningioma).
