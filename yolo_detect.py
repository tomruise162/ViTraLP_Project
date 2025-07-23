from ultralytics import YOLO
import cv2
import os
import numpy as np
from typing import Union, List

def detect_license_plates(image_input: Union[str, np.ndarray, List[Union[str, np.ndarray]]], model_path, save_dir="outputs/crops", conf_thresh=0.4):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Đảm bảo image_input là list
    if not isinstance(image_input, list):
        image_input = [image_input]

    all_cropped_paths = []

    for idx, img in enumerate(image_input):
        # Run inference
        results = model(img, conf=conf_thresh)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()

        # Load ảnh gốc
        if isinstance(img, str):
            image = cv2.imread(img)
            img_name = os.path.splitext(os.path.basename(img))[0]
        else:
            image = img
            img_name = f"input_{idx}"

        cropped_paths = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = image[y1:y2, x1:x2]
            crop_path = os.path.join(save_dir, f"{img_name}_plate_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            cropped_paths.append(crop_path)

        all_cropped_paths.append({"image": img_name, "crops": cropped_paths})

    return all_cropped_paths

# Example usage
if __name__ == "__main__":
    image_paths = [r"D:\DSP_Project\New_dataset\heavy_rainy\train\images\greenpack_0571_png.rf.16bba9369e13dc15cb6c6750d38ac5bb.jpg"]  
    model_path = r"D:\DSP_Project\Src\yolov11_200_epochs.pt"  
    results = detect_license_plates(image_paths, model_path)
    for result in results:
        print(f"Cropped plates for {result['image']}:")
        for path in result['crops']:
            print(" -", path)
