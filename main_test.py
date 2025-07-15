from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from yolo_detect import detect_license_plates
from enhancement import enhance_image
from ocr_infer import recognize_text
import cv2

app = FastAPI()

model_path = "yolo_finetuned_weights/yolo11_rainy_200_best.pt"
UPLOAD_DIR = "uploaded_images"
ENHANCED_DIR = "outputs/enhanced"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    # Lưu file tạm
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: YOLO detection
    results = detect_license_plates(file_location, model_path)
    crop_files = []
    enhanced_files = []
    ocr_results = []
    
    for result in results:
        for crop_path in result["crops"]:
            crop_files.append(crop_path)
            # Step 2: Enhancement
            enhanced_img = enhance_image(crop_path)
            # Lưu ảnh đã enhance
            enhanced_name = os.path.splitext(os.path.basename(crop_path))[0] + "_enhanced.jpg"
            enhanced_path = os.path.join(ENHANCED_DIR, enhanced_name)
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
            enhanced_files.append(enhanced_path)
            # Step 3: OCR
            text = recognize_text(enhanced_path)
            ocr_results.append({
                "enhanced_file": enhanced_path,
                "text": text
            })

    os.remove(file_location)

    return JSONResponse(content={
        "cropped_files": crop_files,
        "enhanced_files": enhanced_files,
        "ocr_results": ocr_results
    })

if __name__ == "__main__":
    uvicorn.run("main_test:app", host="0.0.0.0", port=8001, reload=True) 