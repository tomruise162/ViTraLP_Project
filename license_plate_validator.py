import re

def is_valid_vietnamese_license_plate(text):
    """
    Kiểm tra xem text có phải là biển số xe Việt Nam hợp lệ hay không
    
    Logic đơn giản:
    1. Loại bỏ dấu - và . 
    2. Kiểm tra độ dài chuỗi (7-9 ký tự)
    
    Ví dụ:
    - 29A-1234 -> 29A1234 (7 ký tự) ✅
    - 51A-123.45 -> 51A12345 (8 ký tự) ✅
    - 36B1-246.80 -> 36B124680 (9 ký tự) ✅
    - AB123 -> AB123 (5 ký tự) ❌ (quá ngắn)
    
    Args:
        text (str): Text cần kiểm tra
        
    Returns:
        bool: True nếu là biển số hợp lệ, False nếu không
    """
    if not text:
        return False
    
    # Loại bỏ khoảng trắng và chuyển về chữ hoa
    text = text.strip().upper()
    
    # Loại bỏ dấu - và .
    cleaned_text = text.replace('-', '').replace('.', '')
    
    # Kiểm tra độ dài chuỗi (7-9 ký tự)
    if len(cleaned_text) < 7 or len(cleaned_text) > 9:
        return False
    
    return True

def validate_and_clean_license_plate(text):
    """
    Validate và làm sạch text biển số xe
    
    Args:
        text (str): Text cần validate
        
    Returns:
        tuple: (is_valid, cleaned_text, message)
        - is_valid: bool - có hợp lệ hay không
        - cleaned_text: str - text đã được làm sạch
        - message: str - thông báo lỗi nếu có
    """
    if not text:
        return False, "", "Không có ký tự biển số xe hợp lệ"
    
    # Loại bỏ khoảng trắng và chuyển về chữ hoa
    cleaned_text = text.strip().upper()
    
    # Loại bỏ các ký tự không cần thiết, giữ lại chữ cái, số, dấu gạch ngang và dấu chấm
    cleaned_text = re.sub(r'[^A-Z0-9\-\.]', '', cleaned_text)
    
    # Loại bỏ dấu - và . để kiểm tra độ dài
    length_check_text = cleaned_text.replace('-', '').replace('.', '')
    
    # Kiểm tra độ dài tối thiểu (7 ký tự)
    if len(length_check_text) < 7:
        return False, cleaned_text, "Biển số xe quá ngắn"
    
    # Kiểm tra độ dài tối đa (9 ký tự)
    if len(length_check_text) > 9:
        return False, cleaned_text, "Biển số xe quá dài"
    
    # Kiểm tra định dạng
    if is_valid_vietnamese_license_plate(cleaned_text):
        return True, cleaned_text, "Biển số xe hợp lệ"
    else:
        return False, cleaned_text, "Không có ký tự biển số xe hợp lệ" 