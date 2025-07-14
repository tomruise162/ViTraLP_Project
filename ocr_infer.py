from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

# Khởi tạo PaddleOCR model với model đã export từ Kaggle
ocr_model = PaddleOCR(
    rec_model_dir='D:\ViTraLP\original_weights_paddle',  # thư mục chứa model export
    use_angle_cls=False,
    use_space_char=True,
    use_gpu=False  # nếu bạn có GPU và cài PaddlePaddle-GPU thì để True
)

def recognize_text(img_bgr):
    """
    Nhận diện ký tự từ ảnh BGR bằng PaddleOCRv5 đã fine-tuned.
    Input:
        - img_bgr: ảnh biển số dạng numpy BGR (từ OpenCV)
    Output:
        - text: chuỗi ký tự được nhận dạng
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
