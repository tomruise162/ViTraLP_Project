import cv2
import os
import numpy as np
from collections import defaultdict
from enhancement_prenet_crop import enhance_image_prenet_np
from yolo_detect import detect_license_plates
from ocr_infer import recognize_text

def process_video_roi_yolo_ocr(input_video_path, output_dir, model_path, roi_ratio=(0.32, 0.63, 0.432, 0.336), max_frames_per_text=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ocr_results = defaultdict(list)
    frame_idx = 0
    saved_count = defaultdict(int)
    txt_lines = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop ROI
        x = int(roi_ratio[0] * width)
        y = int(roi_ratio[1] * height)
        w = int(roi_ratio[2] * width)
        h = int(roi_ratio[3] * height)
        roi = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Enhance ROI
        enhanced_roi = enhance_image_prenet_np(roi_rgb)
        # Chạy YOLO trên enhanced ROI
        yolo_results = detect_license_plates(enhanced_roi, model_path)
        for result in yolo_results:
            for crop_path in result["crops"]:
                # Nếu crop_path là file, đọc ảnh; nếu là array, dùng luôn
                crop_img = cv2.imread(crop_path) if isinstance(crop_path, str) else crop_path
                if crop_img is None:
                    continue
                text = recognize_text(crop_img)
                if len(ocr_results[text]) < max_frames_per_text:
                    ocr_results[text].append(frame_idx)
                    saved_count[text] += 1
                    out_path = os.path.join(output_dir, f"{text}_{frame_idx}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(enhanced_roi, cv2.COLOR_RGB2BGR))
                    txt_lines.append(f"{out_path}\t{text}\tframe:{frame_idx}")
        frame_idx += 1
    cap.release()
    # Lưu kết quả ra file txt
    txt_path = os.path.join(output_dir, "ocr_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in txt_lines:
            f.write(line + "\n")
    print(f"Done! Kết quả lưu ở: {output_dir}, file txt: {txt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video pipeline: Crop ROI, enhance ROI, YOLO detect, OCR, lọc tối đa 3 frame/ký tự.")
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default='output_frames', help='Output directory for frames')
    parser.add_argument('--model', '-m', type=str, required=True, help='YOLO model path')
    args = parser.parse_args()
    process_video_roi_yolo_ocr(args.input, args.output, args.model) 