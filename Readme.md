# Vietnamese License Plate Recognition Pipeline

## 📦 Mô tả dự án

Hệ thống nhận diện biển số xe máy/ô tô Việt Nam từ ảnh hoặc video, gồm các bước:
1. **Tăng cường ảnh (Enhancement)**: Sử dụng PReNet để khử mưa, tăng chất lượng ảnh.
2. **Nhận diện biển số (YOLO)**: Phát hiện và crop vùng biển số bằng YOLOv11.
3. **Nhận diện ký tự (OCR)**: Sử dụng PaddleOCR để đọc ký tự trên biển số.
4. **Giao diện web**: Upload ảnh/video, xem kết quả crop, enhanced, text trực tiếp trên web.

---

## 🗂️ Cấu trúc thư mục chính

```
.
├── main_test.py                # Backend FastAPI: API xử lý ảnh/video
├── enhancement_prenet_crop.py  # Module tăng cường ảnh bằng PReNet
├── yolo_detect.py              # Module phát hiện biển số bằng YOLO
├── ocr_infer.py                # Module nhận diện ký tự bằng PaddleOCR
├── outputs/
│   └── enhanced/               # Lưu ảnh enhanced kết quả
│   └── crops/                  # (nếu dùng) Lưu ảnh crop biển số
├── UI/
│   └── my-app/                 # Frontend React
│       └── src/App.js          # Giao diện upload, hiển thị kết quả
├── requirements.txt            # Thư viện Python cần thiết cho backend
├── Readme.md                   # (file này)
└── ...
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Chuẩn bị môi trường backend (FastAPI)

- **Cài đặt Python 3.10.16**
- **Cài đặt các thư viện cần thiết:**
  ```bash
  pip install -r requirements.txt
  ```
- **Tải các file weights (YOLO, PReNet, PaddleOCR) theo hướng dẫn ở cuối README.**
- **Chạy server FastAPI:**
  ```bash
  python main_test.py
  ```
  - Server mặc định chạy ở `http://localhost:8001`
  - Ảnh enhanced sẽ được lưu ở `outputs/enhanced/`
  - API chính: `POST /process` (upload file ảnh/video)

### 2. Chuẩn bị môi trường frontend (React)

```bash
cd UI/my-app
npm install
npm start
```
- Giao diện web chạy ở `http://localhost:3000`
- Kết nối trực tiếp với backend qua API

---

## 🛠️ Các endpoint chính (backend)

- `POST /process`  
  Nhận file ảnh/video, trả về:
  ```json
  {
    "cropped_files": [...],      // (nếu có) các file crop biển số
    "enhanced_files": [...],     // các file enhanced (đường dẫn tĩnh)
    "ocr_results": [
      {
        "enhanced_file": "enhanced/51A-9763.png",
        "text": "51A-9763",
        "frame": 0
      }
    ]
  }
  ```
- `GET /enhanced/{filename}`  
  Trả về file enhanced (dùng cho UI hiển thị)

---

## 🖼️ Giao diện web (UI)

- **Upload** ảnh hoặc video
- **Xem** ảnh crop biển số (nếu có)
- **Xem** ảnh enhanced và text nhận diện được
- **Download** từng ảnh kết quả

---

## 💡 Lưu ý

- Tên file enhanced sẽ tự động thay thế mọi khoảng trắng trong text bằng dấu `_` để tránh lỗi khi lưu/truy cập file.
- Đảm bảo các file weights đã được đặt đúng vị trí như hướng dẫn.
- Nếu muốn nhận diện crop biển số, cần bổ sung logic lưu crop vào `cropped_files` trong backend.

---

## 🔗 Pretrained Weights

Bạn có thể tải các file weights được fine-tuned hoặc pretrained tại đường dẫn sau:  
👉 [Google Drive](https://drive.google.com/drive/folders/1tqchFpQig-Q3iDR3kzcSPCYQ-UGNSHvC?usp=sharing)

Link drive chứa các file sau:

- `yolov11_200_epochs.pth`  
  → YOLO finetuning weights cho biển số xe Việt Nam.

- `original_weights_paddle.pth`  
  → Original PaddleOCRv5 weights.

- `finetuning.zip`  
  → PReNet finetuning weights cho dataset mưa.

> ✅ Hãy **tải và đặt đường dẫn trọng số** một cách phù hợp trước khi train hoặc inference.


