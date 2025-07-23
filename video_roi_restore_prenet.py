import cv2
import os
import numpy as np
from enhancement_prenet_crop import enhance_image_prenet_np

def process_video(input_video_path, output_dir, roi_ratio=(0.32, 0.63, 0.432, 0.336), fps=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Input video: {input_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {input_fps}, Total frames: {frame_count}")
    print(f"Output frames will be saved to: {output_dir}")

    frame_idx = 0
    saved_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB for restoration
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Calculate ROI
        x = int(roi_ratio[0] * width)
        y = int(roi_ratio[1] * height)
        w = int(roi_ratio[2] * width)
        h = int(roi_ratio[3] * height)
        roi = frame_rgb[y:y+h, x:x+w]
        # Process ROI bằng PReNet
        processed_roi = enhance_image_prenet_np(roi)
        # Paste back
        frame_rgb[y:y+h, x:x+w] = processed_roi
        # Convert back to BGR for saving
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Save frame
        out_path = os.path.join(output_dir, f"frame_{saved_idx:05d}.png")
        cv2.imwrite(out_path, frame_bgr)
        saved_idx += 1
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")
    cap.release()
    print(f"Done! {saved_idx-1} frames saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process video, restore only a region of interest in each frame using PReNet, and save frames as images.")
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default='output_frames', help='Output directory for frames')
    args = parser.parse_args()
    process_video(args.input, args.output) 