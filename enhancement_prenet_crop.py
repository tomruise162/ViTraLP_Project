import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from utils import normalize, print_network
from networks import PReNet

def _to_rgb_array(image):
    # Nếu là path thì đọc ảnh, nếu là numpy array thì kiểm tra và chuyển về RGB
    if isinstance(image, str):
        y = cv2.imread(image)
        if y is None:
            raise ValueError(f"Không thể đọc ảnh: {image}")
        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])  # BGR -> RGB
        return y
    elif isinstance(image, np.ndarray):
        if image.shape[2] == 3:
            return image
        else:
            raise ValueError("Input numpy array phải có shape (H, W, 3)")
    else:
        raise ValueError("Input phải là path hoặc numpy array RGB")

# Cấu hình
MODEL_PATH = r"D:/DSP_Project/derain_PReNet/PReNet/logs/finetuning/medium/net_epoch49_med.pth"
RECURRENT_ITER = 6
USE_GPU = True
GPU_ID = "0"

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

prenet_model = load_prenet_model()

def enhance_image_prenet_np(image):
    """
    Enhance ảnh bằng PReNet. Nhận vào numpy array RGB hoặc path, trả về numpy array RGB.
    """
    y = _to_rgb_array(image)
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
    return save_out

if __name__ == "__main__":
    # Test thử với path
    test_image = r"C:/Users/ADMIN/Desktop/test_real/rainy/heavy/hv2.jpg"
    if os.path.exists(test_image):
        enhanced = enhance_image_prenet_np(test_image)
        print(f"Enhanced image shape: {enhanced.shape}")
        cv2.imwrite("enhanced_test.jpg", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        print("Saved as enhanced_test.jpg")
    # Test thử với numpy array
    img = cv2.imread(test_image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced2 = enhance_image_prenet_np(img_rgb)
    cv2.imwrite("enhanced_test2.jpg", cv2.cvtColor(enhanced2, cv2.COLOR_RGB2BGR)) 