import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ép PaddleOCR chỉ dùng CPU
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re

# Khởi tạo PaddleOCR với model custom, chạy trên CPU
ocr_model = PaddleOCR(
    rec_model_dir=r"D:/ViTraLP/original_weights_paddle",
    rec_char_dict_path="D:\ViTraLP\original_weights_paddle\ppocrv5_dict.txt",  # <-- Đường dẫn tới file dict ký tự
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def contains_letter(text):
    """
    Kiểm tra xem text có chứa ít nhất một chữ cái hay không
    """
    return bool(re.search(r'[a-zA-Z]', text))

def correct_first_two_digits(text):
    """
    Đảm bảo 2 ký tự đầu là số, nếu không thì chuyển các chữ dễ nhầm thành số tương ứng.
    """
    char_to_digit = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', 'L': '1',
        'Z': '2', 'S': '5', 'B': '8', 'G': '6'
    }
    text = text.strip()
    if len(text) < 2:
        return text
    new_text = list(text)
    for i in range(2):
        if not new_text[i].isdigit():
            new_text[i] = char_to_digit.get(new_text[i], '0')  # Nếu không map được thì cho về '0'
    return ''.join(new_text)

def correct_third_character(text):
    """
    Fix ký tự thứ 3 bị đọc nhầm từ số thành chữ cái.
    Ví dụ: 29A-1234 -> 29A1234, nhưng OCR có thể đọc nhầm thành 291234
    """
    digit_to_char = {
        '0': 'U', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', 
        '6': 'G', '7': 'T', '8': 'B', '9': 'G'
    }
    
    text = text.strip()
    if len(text) < 3:
        return text
    
    new_text = list(text)
    
    # Kiểm tra nếu 2 ký tự đầu là số và ký tự thứ 3 cũng là số
    if (new_text[0].isdigit() and new_text[1].isdigit() and 
        new_text[2].isdigit() and len(text) >= 7):
        # Có thể ký tự thứ 3 bị đọc nhầm từ chữ thành số
        # Thử chuyển số thành chữ cái tương ứng
        possible_char = digit_to_char.get(new_text[2], 'A')
        new_text[2] = possible_char
    
    return ''.join(new_text)

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
        text = ' '.join(texts)
        text = correct_first_two_digits(text)
        text = correct_third_character(text)  # Thêm fix ký tự thứ 3
        return text
    else:
        return "frame"