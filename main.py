from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from yolo_detect import detect_license_plates
from enhancement import enhance_image
from ocr_infer import recognize_text

app = FastAPI()

yolo_weights = "yolo_finetuned_weights/yolo11_rainy_200_best.pt"  # Sửa lại đường dẫn weights phù hợp
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Lưu file tạm
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Bước 1: Detect biển số và crop
    cropped_results = detect_license_plates(file_location, yolo_weights)
    results = []
    for crop_info in cropped_results:
        for crop_path in crop_info["crops"]:
            # Bước 2: Enhancement
            enhanced_img = enhance_image(crop_path)
            # Bước 3: OCR
            text = recognize_text(enhanced_img)
            results.append({
                "file": crop_path,
                "text": text
            })

    # Xóa file tạm nếu muốn
    # os.remove(file_location)

    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
