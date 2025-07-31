import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import numpy as np
from collections import defaultdict
from enhancement_prenet_crop import enhance_image_prenet_np
from yolo_detect import detect_license_plates
from ocr_infer import recognize_text
from license_plate_validator import validate_and_clean_license_plate
import pyodbc
import time

app = FastAPI()

# Mount static files for outputs directory

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"D:\ViTraLP\yolo_finetuned_weights\yolo11_medium_rainy_200_best.pt"
OUTPUT_DIR = "outputs/enhanced"
app.mount("/enhanced", StaticFiles(directory=OUTPUT_DIR), name="enhanced")
app.mount("/outputs/enhanced", StaticFiles(directory=OUTPUT_DIR), name="outputs-enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=MSI;"
        "DATABASE=OCR_DB;"
        "UID=sa;"
        "PWD=123456"
    )
    return conn

def insert_detected_number(image_path, text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO DETECTED_NUMBER (Recognized_Text, Enhanced_File_Path) VALUES (?, ?)",
        (text, image_path)
    )
    conn.commit()
    cursor.close()
    conn.close()

def recognized_text_exists(text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT TOP 1 1 FROM DETECTED_NUMBER WHERE Recognized_Text = ?",
        (text,)
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists

def allowed_video(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

def allowed_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

@app.post("/process")
async def process(file: UploadFile = File(...)):
    start_time = time.time()  # Bắt đầu đo thời gian
    
    file_location = os.path.join(OUTPUT_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    ocr_results = set()  # Đổi sang set để kiểm tra trùng lặp text trong batch
    txt_lines = []
    result_files = []
    existed_results = []  # Lưu các text đã tồn tại trong DB
    invalid_results = []  # Lưu các text không hợp lệ

    if allowed_video(file.filename):
        cap = cv2.VideoCapture(file_location)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Xử lý toàn frame thay vì ROI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced_frame = enhance_image_prenet_np(frame_rgb)
            yolo_results = detect_license_plates(enhanced_frame, model_path, frame_idx=frame_idx)
            for result in yolo_results:
                for crop_path in result["crops"]:
                    crop_img = cv2.imread(crop_path) if isinstance(crop_path, str) else crop_path
                    if crop_img is None:
                        continue
                    text = recognize_text(crop_img)
                    if text not in ocr_results:
                        ocr_results.add(text)
                        
                        # Validate biển số xe
                        is_valid, cleaned_text, message = validate_and_clean_license_plate(text)
                        
                        if not is_valid:
                            invalid_results.append({"text": text, "message": message})
                            continue  # Bỏ qua, không lưu frame này
                        
                        # Chỉ lưu nếu hợp lệ
                        if recognized_text_exists(cleaned_text):
                            existed_results.append({"text": cleaned_text})
                        else:
                            safe_text = cleaned_text.replace(' ', '_')
                            out_path = os.path.join(OUTPUT_DIR, f"{safe_text}_{frame_idx}.png")
                            cv2.imwrite(out_path, cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR))
                            txt_lines.append(f"{out_path}\t{cleaned_text}\tframe:{frame_idx}")
                            result_files.append({"file": out_path, "text": cleaned_text, "frame": frame_idx})
                            insert_detected_number(out_path, cleaned_text)
            frame_idx += 1
        cap.release()
    elif allowed_image(file.filename):
        img = cv2.imread(file_location)
        if img is None:
            return {"error": "Cannot read image"}
        # Xử lý toàn ảnh thay vì ROI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced_img = enhance_image_prenet_np(img_rgb)
        yolo_results = detect_license_plates(enhanced_img, model_path, frame_idx=0)
        for result in yolo_results:
            for crop_path in result["crops"]:
                crop_img = cv2.imread(crop_path) if isinstance(crop_path, str) else crop_path
                if crop_img is None:
                    continue
                text = recognize_text(crop_img)
                if text not in ocr_results:
                    ocr_results.add(text)
                    
                    # Validate biển số xe
                    is_valid, cleaned_text, message = validate_and_clean_license_plate(text)
                    
                    if not is_valid:
                        invalid_results.append({"text": text, "message": message})
                        continue  # Bỏ qua, không lưu frame này
                    
                    # Chỉ lưu nếu hợp lệ
                    if recognized_text_exists(cleaned_text):
                        existed_results.append({"text": cleaned_text})
                    else:
                        out_path = os.path.join(OUTPUT_DIR, f"{cleaned_text}_0.png")
                        cv2.imwrite(out_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
                        txt_lines.append(f"{out_path}\t{cleaned_text}\tframe:0")
                        result_files.append({"file": out_path, "text": cleaned_text, "frame": 0})
                        insert_detected_number(out_path, cleaned_text)
    else:
        os.remove(file_location)
        return {"error": "Unsupported file type"}

    txt_path = os.path.join(OUTPUT_DIR, "ocr_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in txt_lines:
            f.write(line + "\n")
    
    os.remove(file_location)

    cropped_files = []
    enhanced_files = []
    ocr_results_list = []
    
    for item in result_files:
        # file là file enhanced
        enhanced_file = item["file"].replace("\\", "/").split("/")[-1]
        text = item["text"]
        frame = item["frame"]
        enhanced_path = f"enhanced/{enhanced_file}"
        enhanced_files.append(enhanced_path)
        ocr_results_list.append({
            "enhanced_file": enhanced_path,
            "text": text,
            "frame": frame
        })
        # Nếu có crop_path, thêm vào cropped_files (nếu muốn trả về)
        # cropped_files.append(crop_path)
    
    end_time = time.time()  # Kết thúc đo thời gian
    processing_time = end_time - start_time  # Tính thời gian xử lý
    
    print(f"Tổng thời gian xử lý: {processing_time:.2f} giây")
    
    return {
        "cropped_files": cropped_files,  # Nếu muốn trả về crop, cần lưu crop_path ở trên
        "enhanced_files": enhanced_files,
        "ocr_results": ocr_results_list,
        "existed_results": existed_results,
        "invalid_results": invalid_results,  # Thêm kết quả không hợp lệ
        "processing_time_seconds": round(processing_time, 2)  # Thêm thời gian xử lý vào response
    }

@app.get("/search")
def search_recognized_text(q: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Recognized_Text, Enhanced_File_Path FROM DETECTED_NUMBER WHERE Recognized_Text LIKE ?",
        (f"%{q}%",)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{
        "recognized_text": row[0],
        "enhanced_file_path": row[1]
    } for row in results]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_full_image:app", host="0.0.0.0", port=8001, reload=True) 