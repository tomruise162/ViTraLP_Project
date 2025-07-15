import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_unet(img_shape=(160, 320, 3)):
    inputs = Input(shape=img_shape)

    # Downsampling
    c1 = Conv2D(64, 3, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(512, 3, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

    # Upsampling
    u1 = UpSampling2D()(c4)
    u1 = concatenate([u1, c3])
    c5 = Conv2D(256, 3, padding='same')(u1)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    u2 = UpSampling2D()(c5)
    u2 = concatenate([u2, c2])
    c6 = Conv2D(128, 3, padding='same')(u2)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u3 = UpSampling2D()(c6)
    u3 = concatenate([u3, c1])
    c7 = Conv2D(64, 3, padding='same')(u3)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    outputs = Conv2D(3, 1, activation='tanh')(c7)

    return Model(inputs, outputs)

def preprocess_for_model(image_path, target_size=(160, 320)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Resize đúng (OpenCV dùng width, height)
    image = cv2.resize(image, (target_size[1], target_size[0]))

    # Down-up scale
    img_lr = cv2.resize(
        cv2.resize(image, (target_size[1] // 4, target_size[0] // 4)),
        (target_size[1], target_size[0])
    )

    # Blur + noise
    img_lr = cv2.GaussianBlur(img_lr, (7, 7), 0)
    noise = np.random.normal(0, 75, img_lr.shape).astype(np.float32)
    img_lr = np.clip(img_lr + noise, 0, 255).astype(np.uint8)

    # Normalize về [-1, 1]
    img_lr = img_lr.astype("float32") / 127.5 - 1

    return img_lr

def postprocess_output(output_image):
    """
    Hậu xử lý output từ model U-Net
    """
    # Đưa từ [-1, 1] → [0, 255]
    denorm = ((output_image + 1) * 127.5).astype(np.uint8)
    
    # CLAHE tăng tương phản
    lab = cv2.cvtColor(denorm, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(3, 3))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    return enhanced

model = build_unet(img_shape=(160, 320, 3))
model.load_weights("unet_model_full.h5")

def enhance_image(image_path):
    """
    Hàm chính để enhance ảnh bằng U-Net
    """
    if model is None:
        raise RuntimeError("Model enhancement chưa được load thành công!")
    
    # Tiền xử lý
    input_tensor = preprocess_for_model(image_path)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Dự đoán
    pred = model.predict(input_tensor)[0]
    
    # Hậu xử lý
    final = postprocess_output(pred)
    
    return final

if __name__ == "__main__":
    test_image = r""

    # Thư mục lưu kết quả
    save_dir = "enhanced_LP"
    os.makedirs(save_dir, exist_ok=True)  # Tự tạo nếu chưa tồn tại

    if os.path.exists(test_image):
        enhanced = enhance_image(test_image)
        print(f"Enhanced image shape: {enhanced.shape}")
        
        # Tạo tên file output dựa trên tên ảnh gốc
        filename = os.path.basename(test_image)
        save_path = os.path.join(save_dir, f"enhanced_{filename}")

        # Lưu ảnh
        cv2.imwrite(save_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        print(f"Enhanced image saved as: {save_path}")
    else:
        print(f"Test image not found: {test_image}")