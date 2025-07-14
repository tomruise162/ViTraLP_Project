from yolo_detect import detect_license_plates
from enhancement import enhance_image
from ocr_infer import recognize_text
import os
import glob

def run_pipeline(image_path, yolo_weights_path):
    # Bước 1: Detect biển số và crop
    cropped_paths = detect_license_plates(image_path, yolo_weights_path)

    results = []
    for path in cropped_paths:
        # Bước 2: Enhancement
        enhanced_img = enhance_image(path)

        # Bước 3: OCR
        text = recognize_text(enhanced_img)
        results.append({
            "file": path,
            "text": text
        })

    return results

if __name__ == "__main__":
    yolo_weights = "weights/yolov11_200epochs.pt"
    input_folder = "input_images"

    # Lấy tất cả ảnh từ folder
    image_paths = glob.glob(os.path.join(input_folder, "*.[jp][pn]g"))  # jpg, jpeg, png

    if not image_paths:
        print("⚠️ Không tìm thấy ảnh nào trong thư mục input_images/")
        exit()

    for image_path in image_paths:
        print(f"\n🔍 Xử lý ảnh: {image_path}")
        final_results = run_pipeline(image_path, yolo_weights)

        for res in final_results:
            print(f"📦 [{res['file']}] → 📖: {res['text']}")
