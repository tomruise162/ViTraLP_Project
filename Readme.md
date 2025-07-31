# Vietnamese License Plate Recognition Pipeline

## 📋 Mô tả dự án

Hệ thống nhận diện biển số xe máy/ô tô Việt Nam từ ảnh **hoặc video**, được xây dựng với pipeline xử lý tiên tiến:

### 🔄 Pipeline xử lý
1. **Tăng cường ảnh (Enhancement)**: Sử dụng PReNet để khử mưa và tăng chất lượng ảnh
2. **Phát hiện biển số (Detection)**: Sử dụng YOLOv11 fine-tuned cho biển số Việt Nam
3. **Nhận diện ký tự (OCR)**: Sử dụng PaddleOCR để đọc ký tự trên biển số
4. **Lưu trữ dữ liệu**: Kết quả được lưu vào SQL Server với khả năng tránh trùng lặp
5. **Giao diện web**: Upload ảnh/video và xem kết quả trực tiếp

### ✨ Tính năng chính
- ✅ Xử lý cả ảnh và video
- ✅ Tăng cường chất lượng ảnh trong điều kiện mưa
- ✅ Phát hiện chính xác biển số Việt Nam
- ✅ OCR với độ chính xác cao
- ✅ Giao diện web thân thiện
- ✅ Tìm kiếm biển số đã nhận diện
- ✅ Tránh lưu trùng lặp dữ liệu

---

## 🗂️ Cấu trúc dự án

```
<<<<<<< HEAD
.
├── main_test.py                # Backend FastAPI: API xử lý ảnh/video, lưu SQL Server
├── script.sql                  # Script SQL để khởi tạo database và table
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
=======
ViTraLP/
├── 📁 Backend Servers (FastAPI)
│   ├── main_test.py                    # Server chính với ROI processing
│   ├── main_full_image.py              # Server xử lý toàn bộ ảnh
│   ├── main_half_bottom.py             # Server xử lý nửa dưới ảnh
│   └── requirements.txt                 # Thư viện Python cần thiết
│
├── 📁 Core Modules
│   ├── enhancement_prenet_crop.py      # Module tăng cường ảnh PReNet
│   ├── yolo_detect.py                  # Module phát hiện biển số YOLO
│   ├── ocr_infer.py                    # Module OCR PaddleOCR
│   ├── license_plate_validator.py      # Validator biển số Việt Nam
│   ├── networks.py                      # Kiến trúc mạng PReNet
│   └── utils.py                         # Tiện ích hỗ trợ
│
├── 📁 Frontend (React)
│   └── UI/my-app/
│       ├── src/App.js                   # Giao diện chính
│       ├── package.json                 # Dependencies React
│       └── README.md                    # Hướng dẫn frontend
│
├── 📁 Outputs
│   ├── outputs/enhanced/                # Ảnh đã tăng cường
│   ├── outputs/crops/                   # Ảnh crop biển số
│   ├── output_yolo_crop_test/           # Ảnh crop test
│   └── uploaded_images/                 # Ảnh upload tạm thời
│
├── 📁 Models & Weights
│   ├── yolo_finetuned_weights/          # YOLO weights fine-tuned
│   ├── original_weights_paddle/         # PaddleOCR weights
│   └── original_weights_paddle.zip      # PaddleOCR weights (zipped)
│
├── 📁 Notebooks & Testing
│   ├── test_pipeline.ipynb              # Test pipeline
│   └── Inference_prenet.ipynb           # Test PReNet
│
└── 📄 Documentation & Config
    ├── Readme.md                        # File này
    ├── environment.yml                   # Conda environment
    ├── .gitignore                       # Git ignore rules
    ├── structure                        # Cấu trúc dự án gốc
    └── config/                          # Thư mục cấu hình
>>>>>>> 3923d7d (feat: validate text, update readme, modify ROI)
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 📋 Yêu cầu hệ thống
- **Python**: 3.10.16
- **Node.js**: 16+ (cho frontend)
- **SQL Server**: Để lưu trữ dữ liệu
- **RAM**: Tối thiểu 16GB
- **GPU**: Khuyến nghị có GPU để tăng tốc độ xử lý

<<<<<<< HEAD
- **Cài đặt Python 3.10.16**
- **Cài đặt các thư viện cần thiết:**
  ```bash
  pip install -r requirements.txt
  ```
- **Tải các file weights (YOLO, PReNet, PaddleOCR) theo hướng dẫn ở cuối README.**
- **Chuẩn bị và cấu hình SQL Server:**
  - Cài đặt SQL Server (nếu bạn chưa có).
  - Chạy script trong file `script.sql` để tạo database `OCR_DB` và bảng `DETECTED_NUMBER`.
  - Chỉnh thông tin kết nối SQL Server trong `main_test.py` cho đúng với máy của bạn (server name, user, password).
- **Chạy server FastAPI:**
  ```bash
  python main_test.py
  ```
  - Server mặc định chạy ở `http://localhost:8001`
  - Ảnh enhanced sẽ được lưu ở `outputs/enhanced/`
  - API chính: `POST /process` (upload file ảnh/video)
=======
### 🔧 Cài đặt Backend
>>>>>>> 3923d7d (feat: validate text, update readme, modify ROI)

#### 1. Tạo môi trường Python
```bash
# Tạo conda environment (khuyến nghị)
conda env create -f environment.yml

# Hoặc tạo môi trường mới
conda create -n vitralp python=3.10.16
conda activate vitralp
```

#### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

#### 3. Tải pretrained weights
Tải các file weights từ [Google Drive](https://drive.google.com/drive/folders/1tqchFpQig-Q3iDR3kzcSPCYQ-UGNSHvC?usp=sharing):

- `yolo11_medium_rainy_200_best.pth` → Đặt trong `yolo_finetuned_weights/`
- `original_weights_paddle.pth` → Đặt trong `original_weights_paddle/`
- `finetuning.zip` → Giải nén cho PReNet

#### 4. Cấu hình SQL Server
Chỉnh sửa thông tin kết nối trong các file server:
```python
def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=YOUR_SERVER_NAME;"      # Thay đổi
        "DATABASE=YOUR_DATABASE_NAME;"   # Thay đổi
        "UID=YOUR_USERNAME;"             # Thay đổi
        "PWD=YOUR_PASSWORD;"             # Thay đổi
    )
    return conn
```

#### 5. Tạo bảng SQL Server
```sql
CREATE TABLE DETECTED_NUMBER (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    Recognized_Text NVARCHAR(50) NOT NULL,
    Enhanced_File_Path NVARCHAR(500) NOT NULL,
    Created_At DATETIME DEFAULT GETDATE()
);
```

#### 6. Chạy server
Có 3 lựa chọn server tùy theo nhu cầu:

**Server chính (ROI processing):**
```bash
python main_test.py
```

**Server xử lý toàn ảnh:**
```bash
python main_full_image.py
```

**Server xử lý nửa dưới:**
```bash
python main_half_bottom.py
```

- Server chạy tại: `http://localhost:8001`
- API docs: `http://localhost:8001/docs`

### 🎨 Cài đặt Frontend

#### 1. Cài đặt dependencies
```bash
cd UI/my-app
npm install
```

#### 2. Chạy development server
```bash
npm start
```
- Giao diện web: `http://localhost:3000`

---

## 🔌 API Endpoints

### `POST /process`
**Xử lý upload ảnh/video**

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File ảnh (.jpg, .jpeg, .png, .bmp) hoặc video (.mp4, .avi, .mov, .mkv)

**Response:**
```json
{
  "processing_time_seconds": 2.45,
  "enhanced_files": [
    "enhanced/51A-9763_0.png",
    "enhanced/30A-12345_1.png"
  ],
  "ocr_results": [
    {
      "enhanced_file": "enhanced/51A-9763_0.png",
      "text": "51A-9763",
      "frame": 0
    }
  ],
  "existed_results": [
    {
      "text": "30A-12345",
      "enhanced_file": "enhanced/30A-12345_1.png"
    }
  ],
  "invalid_results": [
    {
      "text": "INVALID-123",
      "message": "Biển số xe không hợp lệ"
    }
  ]
}
```

### `GET /search?q=TEXT`
**Tìm kiếm biển số đã nhận diện**

**Response:**
```json
[
  {
    "recognized_text": "51A-9763",
    "enhanced_file_path": "outputs/enhanced/51A-9763_0.png"
  }
]
```

### `GET /enhanced/{filename}`
**Tải ảnh đã tăng cường**

### `GET /outputs/enhanced/{filename}`
**Tải ảnh từ thư mục outputs**

---

## 🖼️ Giao diện web

### Tính năng chính:
- **Upload**: Hỗ trợ drag & drop ảnh/video
- **Preview**: Xem trước ảnh trước khi xử lý
- **Results**: Hiển thị kết quả OCR và ảnh enhanced
- **Download**: Tải từng ảnh kết quả
- **Search**: Tìm kiếm biển số đã nhận diện
- **History**: Xem lịch sử xử lý

### Cách sử dụng:
1. Truy cập `http://localhost:3000`
2. Upload ảnh hoặc video
3. Chờ xử lý (có progress bar)
4. Xem kết quả OCR và ảnh enhanced
5. Tải ảnh hoặc tìm kiếm biển số

---

## 🔧 Cấu hình nâng cao

### Chọn server phù hợp:

#### 1. `main_test.py` (Khuyến nghị)
- Xử lý ROI với tỷ lệ cố định
- Tối ưu cho biển số ở vị trí cụ thể
- Nhanh nhất trong 3 server

#### 2. `main_full_image.py`
- Xử lý toàn bộ ảnh/video
- Phù hợp khi biển số có thể ở bất kỳ đâu
- Chậm hơn nhưng chính xác hơn

#### 3. `main_half_bottom.py`
- Xử lý nửa dưới của ảnh/video
- Phù hợp cho camera an ninh
- Cân bằng giữa tốc độ và độ chính xác

### Thay đổi model paths
Trong các file server:
```python
model_path = "path/to/your/yolo/weights.pt"
```

### Tùy chỉnh ROI (chỉ cho main_test.py)
```python
ROI_RATIO = (0.32, 0.63, 0.432, 0.336)  # (x1, y1, x2, y2)
```

### Thay đổi port
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## 🐛 Troubleshooting

### Lỗi thường gặp:

#### 1. "Module not found"
```bash
pip install -r requirements.txt
```

#### 2. "CUDA out of memory"
- Giảm batch size trong code
- Sử dụng CPU thay vì GPU
- Đóng các ứng dụng khác

#### 3. "SQL Server connection failed"
- Kiểm tra thông tin kết nối
- Đảm bảo SQL Server đang chạy
- Cài đặt ODBC Driver 17

#### 4. "Model weights not found"
- Tải weights từ Google Drive
- Kiểm tra đường dẫn trong code

### Logs và Debug:
```bash
# Chạy với debug mode
python main_test.py --debug

# Xem logs chi tiết
tail -f logs/app.log
```

---

## 🤝 Đóng góp

### Cách đóng góp:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

### Guidelines:
- Tuân thủ PEP 8 cho Python
- Comment code rõ ràng
- Test trước khi commit
- Update documentation

---

## 📄 License

Dự án này được phát hành dưới MIT License.

---

## 📞 Liên hệ

- **Email**: [tonybdg2061@gmail.com]
- **GitHub**: [tomruise162]
- **LinkedIn**: [https://www.linkedin.com/in/giabao16/]

---

## 🙏 Acknowledgments

- PReNet authors cho mô hình tăng cường ảnh
- YOLOv11 team cho mô hình detection
- PaddleOCR team cho OCR engine
- FastAPI team cho web framework

---

*Cập nhật lần cuối: Tháng 7, 2025*


