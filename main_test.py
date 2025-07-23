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

app = FastAPI()

# Mount static files for outputs directory


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"D:/DSP_Project/Src/yolov11_200_epochs.pt"
OUTPUT_DIR = "outputs/enhanced"
app.mount("/enhanced", StaticFiles(directory=OUTPUT_DIR), name="enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_FRAMES_PER_TEXT = 5
ROI_RATIO = (0.32, 0.63, 0.432, 0.336)

def allowed_video(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

def allowed_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

@app.post("/process")
async def process(file: UploadFile = File(...)):
    file_location = os.path.join(OUTPUT_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    ocr_results = defaultdict(list)
    txt_lines = []
    result_files = []

    if allowed_video(file.filename):
        cap = cv2.VideoCapture(file_location)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x = int(ROI_RATIO[0] * width)
            y = int(ROI_RATIO[1] * height)
            w = int(ROI_RATIO[2] * width)
            h = int(ROI_RATIO[3] * height)
            roi = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            enhanced_roi = enhance_image_prenet_np(roi_rgb)
            yolo_results = detect_license_plates(enhanced_roi, model_path)
            for result in yolo_results:
                for crop_path in result["crops"]:
                    crop_img = cv2.imread(crop_path) if isinstance(crop_path, str) else crop_path
                    if crop_img is None:
                        continue
                    text = recognize_text(crop_img)
                    if len(ocr_results[text]) < MAX_FRAMES_PER_TEXT:
                        safe_text = text.replace(' ', '_')
                        out_path = os.path.join(OUTPUT_DIR, f"{safe_text}_{frame_idx}.png")
                        cv2.imwrite(out_path, cv2.cvtColor(enhanced_roi, cv2.COLOR_RGB2BGR))
                        txt_lines.append(f"{out_path}\t{text}\tframe:{frame_idx}")
                        result_files.append({"file": out_path, "text": text, "frame": frame_idx})
            frame_idx += 1
        cap.release()
    elif allowed_image(file.filename):
        img = cv2.imread(file_location)
        if img is None:
            return {"error": "Cannot read image"}
        height, width = img.shape[:2]
        x = int(ROI_RATIO[0] * width)
        y = int(ROI_RATIO[1] * height)
        w = int(ROI_RATIO[2] * width)
        h = int(ROI_RATIO[3] * height)
        roi = img[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        enhanced_roi = enhance_image_prenet_np(roi_rgb)
        yolo_results = detect_license_plates(enhanced_roi, model_path)
        for result in yolo_results:
            for crop_path in result["crops"]:
                crop_img = cv2.imread(crop_path) if isinstance(crop_path, str) else crop_path
                if crop_img is None:
                    continue
                text = recognize_text(crop_img)
                if len(ocr_results[text]) < MAX_FRAMES_PER_TEXT:
                    ocr_results[text].append(0)
                    out_path = os.path.join(OUTPUT_DIR, f"{text}_0.png")
                    cv2.imwrite(out_path, cv2.cvtColor(enhanced_roi, cv2.COLOR_RGB2BGR))
                    txt_lines.append(f"{out_path}\t{text}\tframe:0")
                    result_files.append({"file": out_path, "text": text, "frame": 0})
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
    
    return {
        "cropped_files": cropped_files,  # Nếu muốn trả về crop, cần lưu crop_path ở trên
        "enhanced_files": enhanced_files,
        "ocr_results": ocr_results_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_test:app", host="0.0.0.0", port=8001, reload=True) 