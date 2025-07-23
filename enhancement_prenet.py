import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from utils import normalize, print_network
from networks import PReNet

# Cấu hình
MODEL_PATH = r"D:\DSP_Project\derain_PReNet\PReNet\logs\finetuning\medium\net_epoch49_med.pth"
RECURRENT_ITER = 6
USE_GPU = True
GPU_ID = "0"

# Thiết lập GPU
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def load_prenet_model():
    model = PReNet(RECURRENT_ITER, USE_GPU)
    print_network(model)
    if USE_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda' if USE_GPU else 'cpu'))
    model.eval()
    return model

# Tải model 1 lần khi import
prenet_model = load_prenet_model()

def enhance_image_prenet(image_path):
    """
    Enhance ảnh bằng PReNet. Trả về numpy array RGB (không lưu file).
    """
    # Đọc và chuẩn hóa ảnh
    y = cv2.imread(image_path)
    if y is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    b, g, r = cv2.split(y)
    y = cv2.merge([r, g, b])  # BGR -> RGB
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)  # (1, 3, H, W)
    y = Variable(torch.Tensor(y))
    if USE_GPU:
        y = y.cuda()
    with torch.no_grad():
        if USE_GPU:
            torch.cuda.synchronize()
        out, _ = prenet_model(y)
        out = torch.clamp(out, 0., 1.)
        if USE_GPU:
            torch.cuda.synchronize()
    out_np = out.data.cpu().numpy().squeeze() if USE_GPU else out.data.numpy().squeeze()
    save_out = np.uint8(255 * out_np).transpose(1, 2, 0)  # (H, W, 3)
    # Chuyển lại RGB (giữ nguyên, vì đã là RGB)
    return save_out

# if __name__ == "__main__":
#     # Test thử
#     test_image = r"C:\Users\ADMIN\Desktop\test_real\rainy\heavy\hv2.jpg"
#     if os.path.exists(test_image):
#         enhanced = enhance_image_prenet(test_image)
#         print(f"Enhanced image shape: {enhanced.shape}")
#         cv2.imwrite("enhanced_test.jpg", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
#         print("Saved as enhanced_test.jpg")
#     else:
#         print(f"Test image not found: {test_image}") 

if __name__ == "__main__":
    input_folder = r"C:\Users\ADMIN\Desktop\test_real\rainy\medium"
    output_folder = r"C:\Users\ADMIN\Desktop\test_real\results\medium"
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_exts]

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            enhanced = enhance_image_prenet(input_path)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_bgr)
            print(f"Enhanced: {filename}")
        except Exception as e:
            print(f"Failed: {filename} - {e}")