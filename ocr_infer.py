import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ép PaddleOCR chỉ dùng CPU
from paddleocr import PaddleOCR
import cv2
import numpy as np

# Khởi tạo PaddleOCR với model custom, chạy trên CPU
ocr_model = PaddleOCR(
    rec_model_dir=r"D:/ViTraLP/original_weights_paddle",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def recognize_text(img_bgr):
    """
    Nhận diện ký tự từ ảnh BGR (numpy) hoặc đường dẫn file bằng PaddleOCR
    """
    if isinstance(img_bgr, str):
        # Nếu đầu vào là đường dẫn file
        result = ocr_model.ocr(img_bgr, cls=False)
    else:
        # Nếu đầu vào là ảnh numpy
        result = ocr_model.ocr(img_bgr, cls=False)

    if result and result[0]:
        texts = [line[1][0] for line in result[0]]
        return ' '.join(texts)
    else:
        return "[No text detected]"