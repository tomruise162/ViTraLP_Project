from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from yolo_detect import detect_license_plates

app = FastAPI()

model_path = "yolo_finetuned_weights/yolo11_rainy_200_best.pt"
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    # Lưu file tạm
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Chạy YOLO detection
    results = detect_license_plates(file_location, model_path)
    crop_files = []
    for result in results:
        crop_files.extend(result["crops"])

    # Xóa file tạm nếu muốn
    os.remove(file_location)

    return JSONResponse(content={"cropped_files": crop_files})

if __name__ == "__main__":
    uvicorn.run("main_test:app", host="0.0.0.0", port=8001, reload=True) 