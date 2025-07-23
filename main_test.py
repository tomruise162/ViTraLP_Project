import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from yolo_detect import detect_license_plates
from enhancement import enhance_image
from ocr_infer import recognize_text
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"D:\DSP_Project\Src\yolov11_200_epochs.pt"
UPLOAD_DIR = "uploaded_images"
ENHANCED_DIR = "outputs/enhanced"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

CROPS_DIR = "outputs/crops"
os.makedirs(CROPS_DIR, exist_ok=True)
app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")
# Mount static files for enhanced images
app.mount("/enhanced", StaticFiles(directory=ENHANCED_DIR), name="enhanced")

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
            crop_name = os.path.basename(crop_path)
            crop_files.append(f"crops/{crop_name}")
            # Step 2: Enhancement
            enhanced_img = enhance_image(crop_path)
            # Lưu ảnh đã enhance
            enhanced_name = os.path.splitext(crop_name)[0] + "_enhanced.jpg"
            enhanced_path = os.path.join(ENHANCED_DIR, enhanced_name)
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
            enhanced_files.append(f"enhanced/{enhanced_name}")
            # Step 3: OCR
            text = recognize_text(enhanced_path)
            ocr_results.append({
                "enhanced_file": f"enhanced/{enhanced_name}",
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