# Dự án Data Mining: Hệ thống Phát hiện Gian lận Giao dịch Thẻ tín dụng (XGBoost, HAT & ARF)

> Báo cáo và hướng dẫn triển khai theo cấu trúc 5 chương, tích hợp ứng dụng web Django cho phép chuyển đổi trực tiếp giữa **XGBoost**, **Hoeffding Adaptive Tree (HAT)** và **Adaptive Random Forest (ARF)** qua Dropdown để đối chiếu kết quả, độ tin cậy và giải thích mô hình (XAI).

---

## Chương 1: Mở đầu

### 1.1. Đặt vấn đề
Trong lĩnh vực tài chính – ngân hàng, giao dịch gian lận thẻ tín dụng gây thiệt hại lớn nhưng chỉ chiếm tỷ lệ rất nhỏ so với tổng số giao dịch hợp lệ (bài toán **mất cân bằng dữ liệu cực đoan** — khoảng `0.17%–0.19%`). Bên cạnh đó, hành vi gian lận liên tục thay đổi theo thời gian (**Concept Drift**), khiến các mô hình học máy tĩnh (Batch Learning) có nguy cơ giảm hiệu quả nếu không được đối chiếu hoặc thích ứng theo luồng dữ liệu (Streaming Learning).

### 1.2. Tổng quan tài liệu
- **Hướng tiếp cận Batch Learning (XGBoost):** Huấn luyện toàn cục trên tập dữ liệu lịch sử với cơ chế cân bằng trọng số lớp (`scale_pos_weight`), tối ưu hàm mất mát bậc hai và cho khả năng tách lớp mạnh.
- **Hướng tiếp cận Online / Streaming Learning (`river` — HAT & ARF):** Học tuần tự từng giao dịch (`learn_one`), sử dụng bất đẳng thức Hoeffding và bộ phát hiện trôi khái niệm **ADWIN** để tự động thích ứng khi phân phối dữ liệu thay đổi.

### 1.3. Mục đích, mục tiêu, đối tượng và phạm vi nghiên cứu
- **Mục đích:** Xây dựng hệ thống phát hiện gian lận giao dịch minh bạch, có khả năng **giải thích kết quả** theo mạch tư duy khoa học (**Ý tưởng → Dẫn dắt → Phản biện → Minh chứng**).
- **Mục tiêu cụ thể:**
  1. Khai thác bộ dữ liệu đủ lớn, chất lượng (`creditcard_prefix.csv` và `creditcard_suffix.csv`) và trích xuất bảng **14 dòng mẫu đại diện** phản ánh đầy đủ đặc điểm bài toán.
  2. Đánh giá mức độ ảnh hưởng của từng thuộc tính (`Time`, `V1`–`V28`, `Amount`) lên mô hình dự báo.
  3. Huấn luyện, lưu trữ và đọc lại 3 mô hình (**XGBoost**, **HAT**, **ARF**) để dự báo thời gian thực.
  4. Đánh giá độ tin cậy của mô hình qua các chỉ số chuẩn cho dữ liệu mất cân bằng (`PR-AUC`, `ROC-AUC`, `F1-Score`, `Precision`, `Recall`, `Confusion Matrix`).
  5. Xây dựng trang web **Django** có **Dropdown** chọn mô hình để trực quan hóa sự khác biệt giữa 3 mô hình.
- **Đối tượng & phạm vi:** 284,807 giao dịch thẻ tín dụng (chia thành `data/creditcard_prefix.csv` với 142,403 dòng và `data/creditcard_suffix.csv` với 142,404 dòng).

---

## Chương 2: Cơ sở lý thuyết

### 2.1. Bài toán mất cân bằng dữ liệu & Concept Drift
- Với tỷ lệ gian lận ~`0.19%`, độ chính xác tổng thể (`Accuracy`) bị vô nghĩa (một mô hình luôn đoán `0` vẫn đạt `99.8%` Accuracy). Do đó, cần dùng **Precision**, **Recall**, **F1-Score**, **ROC-AUC** và đặc biệt là **PR-AUC (Average Precision)**.
- **Concept Drift:** Phân phối $P(X, y)$ của giao dịch thay đổi theo thời gian (`Time`), đòi hỏi so sánh giữa mô hình học tĩnh (Batch) và mô hình học luồng thích ứng (Online Adaptive).

### 2.2. Các mô hình sử dụng
1. **XGBoost (`xgboost.XGBClassifier`):**
   - Mô hình Gradient Boosted Decision Trees theo lô (Batch), sử dụng `scale_pos_weight = neg_count / pos_count` để phạt nặng sai số trên lớp gian lận.
2. **Cây thích ứng HAT (`river.tree.HoeffdingAdaptiveTreeClassifier`):**
   - Cây quyết định trực tuyến dựa trên **bất đẳng thức Hoeffding** để quyết định tách nhánh với số mẫu tối thiểu, tích hợp bộ dò **ADWIN** nhằm thay thế nhánh cây cũ khi có Concept Drift.
3. **Adaptive Random Forest (`river.forest.ARFClassifier`):**
   - Quần thể nhiều cây Hoeffding học trên các tập con đặc trưng ngẫu nhiên, kết hợp cơ chế lấy mẫu Poisson và **ADWIN** cho từng cây để tạo cây nền (background tree) thay thế khi trôi khái niệm xảy ra.

---

## Chương 3: Phương pháp nghiên cứu

### 3.1. Phương pháp triển khai thu thập, tiền xử lý và chuẩn bị 14 dòng dữ liệu mẫu
- **Tập huấn luyện khởi tạo (`data/creditcard_prefix.csv`):** `142,403` giao dịch, gồm `269` giao dịch gian lận (`0.189%`).
- **Tập luồng thích ứng & kiểm thử (`data/creditcard_suffix.csv`):** `142,404` giao dịch, gồm `223` giao dịch gian lận (`0.157%`).
- **Chuẩn hóa dữ liệu (`sklearn.preprocessing.StandardScaler`):** Chỉ `fit` trên tập `prefix` rồi `transform` cho `suffix` (và lưu vào `scaler.pkl`) để tránh rò rỉ dữ liệu (Data Leakage).
- **Tiền xử lý riêng cho XGBoost:**
  - *Ở tầng Pipeline:* Giữ nguyên phân phối gốc (không dùng SMOTE tránh overfitting), sử dụng tham số $\text{scale\_pos\_weight} = \frac{142{,}134}{269} \approx 528.38$ để tăng trọng số hàm mất mát `logloss` cho lớp gian lận. Dù cây quyết định không bắt buộc chuẩn hóa thang đo, XGBoost vẫn dùng chung đầu vào đã qua `StandardScaler` để đồng bộ công bằng với **HAT** (dùng Naive Bayes ở lá nhạy với thang đo) / **ARF** và tính điểm bất thường $|z\text{-score}| \times \text{importance}$ cho phần giải thích XAI.
  - *Ở tầng nội tại thuật toán:* Khi gọi `.fit()`, XGBoost tự động rời rạc hóa giá trị liên tục theo phân vị (**Quantile Sketch / Histogram Binning**) và tự học hướng rẽ tối ưu cho giá trị thiếu (**Sparsity-aware Split Finding**).
- **Trích xuất 14 dòng mẫu đại diện (`fraud_web/ml_models/samples.json`):** Hàm `build_14_representative_samples` trong `ml_engine/train_models.py` chọn **7 giao dịch gian lận** và **7 giao dịch hợp lệ** từ tập `suffix` trải đều theo dải giá trị `Amount` (từ nhỏ đến lớn) và `Time`, phản ánh tính đa dạng của bài toán khi hiển thị trên web.

### 3.2. Phương pháp đánh giá ảnh hưởng thuộc tính & đọc mô hình đã huấn luyện
- **Ảnh hưởng thuộc tính (Feature Importance):**
  - Với **XGBoost**: trích xuất trực tiếp `feature_importances_` (Gain chuẩn hóa).
  - Với **HAT** và **ARF**: tính **Permutation Importance** trên tập con phân tầng của `suffix` (đo độ suy giảm `PR-AUC` khi hoán vị từng cột đặc trưng).
- **Lưu và đọc mô hình để dự báo (`fraud_web/fraud_app/ml_service.py`):**
  - `train_models.py` lưu `scaler.pkl`, `xgb.pkl`, `hat.pkl`, `arf.pkl`, `metrics.json`, `samples.json`, `dataset_summary.json` vào `fraud_web/ml_models/`.
  - Lớp singleton `FraudPredictionService` nạp các artifact này một lần khi khởi động Django, chuẩn hóa vector đầu vào (30 chiều: `Time`, `V1`–`V28`, `Amount`) bằng `scaler.pkl` và gọi `predict_proba` (XGBoost) hoặc `predict_one` / `predict_proba_one` (HAT, ARF).

### 3.3. Xây dựng trang web Django và các ứng dụng
- **Kiến trúc dự án Django (`fraud_web/`):**
  - `fraud_web/fraud_web/settings.py` & `urls.py`: Cấu hình ứng dụng `fraud_app` và định tuyến `/` cùng `/api/predict/`.
  - `fraud_web/fraud_app/views.py`:
    - `IndexView` (`GET /`): Truyền `metrics_json`, `samples_json`, `dataset_summary_json` xuống giao diện.
    - `PredictView` (`POST /api/predict/`): Nhận `{ "model": "XGBoost|HAT|ARF", "features": {...} }`, trả về nhãn dự báo, xác suất gian lận, độ tin cậy, top 5 thuộc tính tác động và giải thích 4 bước.
  - `fraud_web/fraud_app/templates/fraud_app/index.html`:
    - **Dropdown chọn mô hình** (`XGBoost`, `HAT`, `ARF`): Tự động cập nhật bảng độ tin cậy, biểu đồ Top 10 Feature Importance (Chart.js) và tự động dự báo lại giao dịch đang chọn để thấy rõ sự khác biệt giữa các mô hình.
    - **Bảng 14 dòng mẫu** + **Form nhập nhanh thông số** (`Time`, `Amount`, `V14`, `V17`, `V12`, `V10`).

---

## Chương 4: Kết quả và thảo luận

### 4.1. Kết quả thực nghiệm trên tập kiểm thử `creditcard_suffix.csv` (142,404 giao dịch, 223 gian lận)

| Mô hình | Chế độ học | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Confusion Matrix `[TN, FP; FN, TP]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | Batch (`prefix` → `suffix`) | 68.20% | 79.82% | 73.55% | 0.9721 | 0.7983 | `[[142098, 83], [45, 178]]` |
| **ARF** | Online Ensemble Streaming | 100.00% | 22.87% | 37.23% | 0.8904 | 0.5937 | `[[142181, 0], [172, 51]]` |
| **HAT** | Online Single Tree + Minority Weighting | 0.56% | 55.61% | 1.12% | 0.7642 | 0.0124 | `[[120312, 21869], [99, 124]]` |

### 4.2. Phân tích & Giải thích kết quả (Ý tưởng → Dẫn dắt → Phản biện → Minh chứng)
- **Ý tưởng:** Kết hợp giữa học theo lô tối ưu toàn cục (**XGBoost**) và học trực tuyến thích ứng trôi khái niệm (**HAT**, **ARF**) giúp đánh giá toàn diện cả độ chính xác tĩnh lẫn khả năng thích ứng luồng dữ liệu.
- **Dẫn dắt:**
  - Trên XGBoost, hai thuộc tính PCA có ảnh hưởng lớn nhất là `V14` (`32.1%`) và `V10` (`18.8%`).
  - Trên ARF (quần thể 5 cây Hoeffding), các thuộc tính đóng góp nổi bật qua Permutation Importance là `V16` (`39.1%`), `V4` (`39.1%`) và `V3` (`16.5%`).
  - Khi dự báo từng giao dịch trên web, hệ thống tính điểm tác động $|z\text{-score}| \times \text{importance}$ để chỉ ra đúng thuộc tính nào đẩy xác suất gian lận tăng cao.
- **Phản biện:**
  - **XGBoost** đạt `PR-AUC = 0.7983` và `F1 = 73.55%` cao nhất, nhưng là mô hình tĩnh — nếu hành vi gian lận biến đổi mạnh trong tương lai, mô hình cần được `retrain` định kỳ.
  - **HAT** (cây đơn) khi không cân bằng lớp sẽ dễ sụp đổ về lớp đa số (`0`); khi áp dụng tăng trọng số lớp thiểu số (`repeat=15`), `Recall` tăng lên `55.61%` (bắt được `124/223` ca gian lận), nhưng biên quyết định của một cây đơn còn thô nên tạo ra `21,869` cảnh báo nhầm (`Precision = 0.56%`).
  - **ARF** khắc phục nhược điểm của cây đơn nhờ cơ chế quần thể + ADWIN: đạt `Precision = 100%` (`FP = 0`, không làm phiền khách hàng hợp lệ), nhưng `Recall = 22.87%` vẫn bỏ sót `172` ca gian lận.
- **Minh chứng:** Số liệu thực nghiệm từ `fraud_web/ml_models/metrics.json` và kiểm thử trực tiếp qua Dropdown trên trang web Django xác nhận rõ sự đánh đổi giữa **Batch Precision/Recall (XGBoost)**, **Online Precision tuyệt đối (ARF)** và **Online Sensitivity của cây đơn (HAT)**.

### 4.3. Giải thích chi tiết hiện tượng HAT chỉ đạt F1-Score = 1.12%
Dù `Recall` của **HAT** đạt `55.61%` (bắt đúng `124/223` ca gian lận) và `ROC-AUC` đạt `0.7642`, chỉ số `F1-Score` chỉ còn `1.12%` vì các nguyên nhân sau:
1. **Tác động của trung bình điều hòa khi Precision quá thấp:**
   - Từ Confusion Matrix của HAT (`TN = 120,312`, `FP = 21,869`, `FN = 99`, `TP = 124`), tỷ lệ cảnh báo đúng trên tổng số cảnh báo là $\text{Precision} = \frac{124}{124 + 21869} = 0.5638\%$.
   - Theo công thức $F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.005638 \times 0.556054}{0.005638 + 0.556054} \approx 1.12\%$, lượng `FP = 21,869` lớn gấp ~`176` lần `TP` đã kéo `F1` xuống rất thấp.
2. **Đánh đổi kỹ thuật trên cây đơn trực tuyến:**
   - Với tỷ lệ mất cân bằng cực đoan (`~0.16%–0.19%`), HAT mặc định sẽ sụp đổ về lớp đa số (`F1 = 0%`). Khi bật tăng trọng số lớp thiểu số (`repeat = 15` lần `learn_one` cho `Class = 1`, `grace_period = 50`, `delta = 0.01`, `leaf_prediction = "nb"`), xác suất tiên nghiệm của lớp gian lận ở các lá Naive Bayes tăng mạnh giúp nhạy với gian lận hơn, nhưng biên quyết định của **một cây đơn lẻ** còn thô nên dễ báo nhầm giao dịch hợp lệ.
   - Khi chuyển sang mô hình quần thể thích ứng (**ARF** với 5 cây Hoeffding), hiện tượng báo nhầm này được triệt tiêu (`FP = 0`, `Precision = 100%`, `F1 = 37.23%`).

---

## Chương 5: Kết luận và hướng phát triển đề tài

### 5.1. Kết luận
Dự án đã hoàn thành đầy đủ quy trình Data Mining từ tiền xử lý dữ liệu lớn (`creditcard_prefix.csv` / `creditcard_suffix.csv`), huấn luyện và đánh giá 3 mô hình đại diện cho Batch Learning (**XGBoost**) và Online Streaming Learning (**HAT**, **ARF**), đến xây dựng ứng dụng web **Django** trực quan với Dropdown chuyển đổi mô hình và cơ chế giải thích kết quả 4 bước.

### 5.2. Hướng phát triển
- Kết hợp mô hình lai (Hybrid): dùng **XGBoost** làm bộ lọc chính và **ARF** chạy song song trên luồng giao dịch để cảnh báo sớm Concept Drift.
- Tích hợp **SHAP (SHapley Additive exPlanations)** cho từng giao dịch và hỗ trợ tải lên tệp CSV hàng loạt trên giao diện Django.

---

## Tài liệu tham khảo & Phụ lục

### Tài liệu tham khảo
1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
3. Bifet, A., & Gavaldà, R. (2009). *Adaptive Learning from Evolving Data Streams* (Hoeffding Adaptive Tree & ADWIN). IDA.
4. Gomes, H. M., et al. (2017). *Adaptive Random Forests for Evolving Data Stream Classification*. Machine Learning, Springer.
5. Tài liệu thư viện: `scikit-learn`, `xgboost`, `river`, `Django`.

### Phụ lục: Hướng dẫn cài đặt và chạy hệ thống
```powershell
# 1. Kích hoạt môi trường ảo (đã cài sẵn django, xgboost, scikit-learn, river, pandas, numpy)
.\venv\Scripts\Activate.ps1

# 2. (Tuỳ chọn) Huấn luyện lại 3 mô hình và xuất metrics + 14 mẫu
.\venv\Scripts\python.exe ml_engine\train_models.py

# 3. Chạy Unit Tests của Django
.\venv\Scripts\python.exe fraud_web\manage.py test fraud_app

# 4. Khởi động máy chủ web Django
.\venv\Scripts\python.exe fraud_web\manage.py runserver
```
Sau khi chạy `runserver`, mở trình duyệt tại `http://127.0.0.1:8000/` để sử dụng Dropdown chọn mô hình (`XGBoost`, `HAT`, `ARF`) và kiểm tra các giao dịch mẫu.
