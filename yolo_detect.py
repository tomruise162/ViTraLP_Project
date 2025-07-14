from ultralytics import YOLO
import cv2
import os
from typing import Union, List

def detect_license_plates(image_paths: Union[str, List[str]], model_path, save_dir="outputs/crops", conf_thresh=0.4):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Đảm bảo image_paths là list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    all_cropped_paths = []

    for image_path in image_paths:
        # Run inference
        results = model(image_path, conf=conf_thresh)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()

        # Load ảnh gốc
        image = cv2.imread(image_path)
        cropped_paths = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = image[y1:y2, x1:x2]
            crop_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_plate_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            cropped_paths.append(crop_path)

        all_cropped_paths.append({"image": image_path, "crops": cropped_paths})

    return all_cropped_paths

# Example usage
if __name__ == "__main__":
    image_paths = [r"C:\Users\ADMIN\Desktop\test_real\bien-so-xe-co-quan-nha-nuoc_0905103106.png"]  # Có thể là 1 hoặc nhiều ảnh
    model_path = "yolo_finetuned_weights/yolo11_rainy_200_best.pt"  # Đường dẫn mới
    results = detect_license_plates(image_paths, model_path)
    for result in results:
        print(f"Cropped plates for {result['image']}:")
        for path in result['crops']:
            print(" -", path)
