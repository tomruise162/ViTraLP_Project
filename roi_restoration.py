import sys, os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "universal-image-restoration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "universal-image-restoration/config/daclip-sde"))

import options as option
from models import create_model
import open_clip
import utils as util

def setup_restoration_model(config_path='universal-image-restoration/config/daclip-sde/options/test.yml'):
    opt = option.parse(config_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    device = model.device
    clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
    clip_model = clip_model.to(device)
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
    return model, clip_model, sde, device

model, clip_model, sde, device = setup_restoration_model()

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image)

def restore(image):
    # image: numpy array, HWC, RGB, 0-255
    h, w, _ = image.shape
    image = image.astype(np.float32) / 255.
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()
    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    # output is HWC, BGR, 0-255
    output = output[:, :, [2, 1, 0]]  # Convert BGR to RGB
    # Đảm bảo output có cùng kích thước với ROI đầu vào
    if output.shape[0] != h or output.shape[1] != w:
        import cv2
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)
    return output.astype(np.uint8) 