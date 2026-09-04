# BỘ CÂU HỎI BẢO VỆ DỰ ÁN DÀNH CHO CÁC THÀNH VIÊN KHÁC (BẢN MỞ RỘNG)

---

## 1. DƯƠNG CÔNG HÒA
**Vai trò:** Thu thập dữ liệu, tham gia xây dựng mô hình, chuẩn bị dữ liệu MRI, pipeline huấn luyện Deep Learning và đánh giá hiệu quả.

**Q1: Quá trình chuẩn bị dữ liệu (Data Pipeline) trước khi đưa vào huấn luyện diễn ra như thế nào?**
> **Trả lời:** Dữ liệu MRI gốc thường có kích thước và độ tương phản khác nhau. Em sử dụng PyTorch `DataLoader` kết hợp module `transforms`. Ảnh được Resize về kích thước thống nhất `224x224`, chuyển sang định dạng Tensor để tính toán GPU/CPU, và thực hiện Chuẩn hóa (Normalize) theo mean/std để đẩy nhanh quá trình hội tụ của mô hình.

**Q2: Trong pipeline huấn luyện, Data Augmentation (Tăng cường dữ liệu) nào được áp dụng và tại sao?**
> **Trả lời:** Em sử dụng `RandomHorizontalFlip` và `ColorJitter` (thay đổi nhẹ độ sáng/tương phản). Đối với ảnh y tế MRI, ta không được áp dụng các phép biến đổi quá đà như lật dọc (VerticalFlip) hay cắt ghép ngẫu nhiên làm mất cấu trúc não. Việc lật ngang và thay đổi độ sáng giúp đa dạng hóa dữ liệu, giúp mô hình không bị học vẹt (Overfitting).

**Q3: Nhóm đánh giá hiệu quả mô hình dựa trên các chỉ số nào ngoài Accuracy?**
> **Trả lời:** Nhóm sử dụng Confusion Matrix (Ma trận nhầm lẫn) để theo dõi True Positive, False Positive, v.v. Từ đó tính ra **Precision** (Độ chính xác) và **Recall** (Độ phủ). Việc theo dõi Recall rất quan trọng để đảm bảo mô hình không dự đoán sót bệnh nhân có u (False Negative).

**Q4: Hàm Loss (Mất mát) và Optimizer (Tối ưu hóa) nào được sử dụng trong quá trình huấn luyện? Tại sao?**
> **Trả lời:** Em sử dụng hàm `CrossEntropyLoss` vì đây là hàm tiêu chuẩn và tốt nhất cho bài toán phân loại nhiều lớp (Multi-class Classification). Về thuật toán tối ưu, nhóm dùng `AdamW` với Learning Rate là `0.0001`. AdamW giúp mô hình hội tụ nhanh hơn Adam truyền thống và tích hợp sẵn cơ chế Weight Decay để phạt các trọng số quá lớn, giúp giảm Overfitting.

**Q5: Quá trình Early Stopping (Dừng sớm) hoạt động ra sao để tối ưu hóa mô hình?**
> **Trả lời:** Trong lúc huấn luyện, em theo dõi chỉ số `Validation Loss`. Nếu qua 7 epochs liên tiếp (patience=7) mà Loss không giảm thêm, chứng tỏ mô hình đã đạt "đỉnh" hội tụ và bắt đầu học vẹt (Overfitting). Hệ thống sẽ tự động dừng huấn luyện ngay lập tức và nạp lại trọng số tốt nhất (`ssl_best_model.pth`) để lưu lại.

---

## 2. ĐỖ TRƯỜNG GIANG
**Vai trò:** Tiền xử lý dữ liệu, thiết kế giao diện GUI (PyQt5), chuẩn hóa dữ liệu đầu vào và phát triển hiển thị Grad-CAM.

**Q1: Giao diện GUI được xây dựng bằng công nghệ gì và kiến trúc luồng dữ liệu ra sao?**
> **Trả lời:** Giao diện được xây dựng bằng framework `PyQt5`. Em áp dụng mô hình kiến trúc MVC/Controller-View tách biệt (`gui/main_window.py` quản lý UI, `gui/controller.py` quản lý logic AI). Điểm đặc biệt là em dùng **QThread** để chạy mô hình AI ở dưới nền (Background Thread), giúp cho giao diện người dùng không bị "đơ" (freeze) trong lúc chờ máy tính dự đoán ảnh.

**Q2: Khi người dùng đưa một ảnh bất kỳ vào GUI, tiền xử lý (Preprocessing) diễn ra thế nào để model đọc được?**
> **Trả lời:** Ảnh người dùng tải lên thường là JPEG/PNG bất kỳ. Trước khi truyền vào model, em dùng thư viện đọc ảnh, chuyển về không gian màu RGB, Resize về đúng `224x224` pixel như lúc huấn luyện, sau đó ép kiểu thành Torch Tensor và Normalize (chuẩn hóa) y hệt như trong tập training. Nếu không làm bước này, AI sẽ dự đoán sai hoàn toàn do khác phân bố dữ liệu.

**Q3: Grad-CAM là gì? Tại sao phải thêm chức năng Grad-CAM vào giao diện?**
> **Trả lời:** Grad-CAM là kỹ thuật giải thích AI (Explainable AI). Nó dùng đạo hàm từ lớp Conv cuối cùng của mạng CNN để vẽ ra một "Bản đồ nhiệt" (Heatmap). Mục đích là để "Minh bạch hóa AI" — cho bác sĩ thấy mô hình đang tập trung nhìn vào vùng nào trên não (vùng màu đỏ/vàng) để đưa ra quyết định có khối u. 

**Q4: Tại sao lại chọn framework PyQt5 thay vì thiết kế ứng dụng Web?**
> **Trả lời:** Ứng dụng y tế thường đòi hỏi tính bảo mật dữ liệu cao (ảnh MRI của bệnh nhân không được phép tự ý upload lên web server bên thứ 3 do luật bảo vệ quyền riêng tư). Việc xây dựng Desktop App bằng PyQt5 giúp xử lý dữ liệu 100% Offline trên máy cục bộ của bác sĩ, đảm bảo tuyệt đối an toàn thông tin mà giao diện vẫn hiện đại, chuyên nghiệp.

**Q5: GUI xử lý thế nào nếu người dùng đưa vào một ảnh chụp rất lớn (ví dụ 4K) hoặc rất nhỏ?**
> **Trả lời:** Lớp tiền xử lý của em sẽ tự động nội suy (Interpolation) bằng thuật toán Bilinear. Dù ảnh 4K nặng hàng chục MB, nó cũng sẽ được nén và Resize xuống Tensor `3x224x224` trực tiếp trên RAM. Điều này đảm bảo tốc độ dự đoán luôn cực nhanh (dưới 1 giây) và không bao giờ làm tràn bộ nhớ (VRAM) của card đồ họa.

---

## 3. TRẦN VĂN ĐỨC
**Vai trò:** Hỗ trợ chức năng hiển thị kết quả, kiểm thử giao diện, đảm bảo tính ổn định của GUI và phối hợp ghép nối (Integration).

**Q1: Việc đảm bảo tương tác ổn định trên GUI (Testing/Validation) được thực hiện như thế nào?**
> **Trả lời:** Quá trình kiểm thử giao diện tập trung vào việc ngăn chặn lỗi người dùng (User Error). Ví dụ: 
> 1. Disable nút "Predict" nếu người dùng chưa chọn ảnh.
> 2. Disable toàn bộ nút bấm trong lúc AI đang chạy dự đoán để tránh spam click làm crash phần mềm.
> 3. Tự động tìm và load mô hình `ssl_best_model.pth` khi vừa khởi động để người dùng không phải mất công tìm file thủ công.

**Q2: Quá trình từ lúc model xuất ra con số Tensor đến lúc hiển thị tên bệnh và thanh Confidence (%) diễn ra thế nào?**
> **Trả lời:** Output của model là một mảng 4 con số thô. Em truyền qua hàm `Softmax` để chuyển đổi chúng thành phần trăm xác suất (tổng = 100%). Sau đó dùng hàm `Argmax` để tìm ra vị trí có xác suất cao nhất, từ đó map với danh sách Tên Bệnh (Glioma, Meningioma, No Tumor, Pituitary). Thanh tiến trình (Confidence Bar) sẽ vẽ dựa trên chính giá trị phần trăm này.

**Q3: Khó khăn lớn nhất khi ghép nối (Integrate) mô hình AI đã train vào phần mềm Desktop là gì?**
> **Trả lời:** Khó khăn lớn nhất là quản lý đường dẫn tương đối (Relative Paths) khi gọi thư viện PyTorch trong ứng dụng đồ họa Windows. Hơn nữa, việc phải tích hợp luồng vẽ hình Grad-CAM từ mô hình, đè chắp lớp (Overlay) lên ảnh MRI gốc và chuyển đổi nó thành định dạng `QPixmap` của PyQt5 để hiển thị lên màn hình đòi hỏi việc quản lý bộ nhớ ảnh rất tỉ mỉ.

**Q4: Ứng dụng này khi dự đoán sử dụng CPU hay GPU? Tốc độ ra sao?**
> **Trả lời:** Hàm Load Model được thiết kế tự động phát hiện (Auto-detect): Nếu máy tính bác sĩ có Card đồ họa NVIDIA (CUDA), nó sẽ nạp lên GPU. Nếu máy tính văn phòng bình thường, nó tự động fallback chạy bằng CPU. Trọng số mạng `ImprovedCNN` khá tối ưu (chỉ ~11 triệu tham số), nên dù chạy trên CPU thì tốc độ dự đoán và vẽ Grad-CAM chỉ mất chưa tới 0.5 giây cho mỗi ảnh.

**Q5: Việc thiết kế thanh Confidence Bar (Độ tự tin) mang lại lợi ích thực tiễn gì?**
> **Trả lời:** Nó cung cấp thước đo "Độ chắc chắn" của AI. Nếu máy báo Meningioma nhưng Confidence chỉ 55%, bác sĩ sẽ biết AI đang phân vân (do có nhiều nhiễu) và tự mình xem xét kỹ hơn. Ngược lại nếu Confidence 99%, kết quả có độ tin cậy rất cao. Đây là tính năng không thể thiếu trong các ứng dụng AI Y tế thực tế.
