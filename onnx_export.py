import sys
import os
import time
from PIL import Image
import numpy as np
import torch
from cfg import Cfg
import clip

path = 'test.jpg'
weight_path = 'checkpoints/173_0.0_model.pth'

def load_source(cfg):
    model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    image = preprocess(Image.open(path)).unsqueeze(0).cpu() # [1, 3, 224, 224]]
    text = clip.tokenize([(f"a photo of a {i}") for i in range(cfg.num_class)])
    return model, image, text

def convert_inference(model, image, text, image_onnx, text_onnx):
    from clip_onnx import clip_onnx, attention
    clip.model.ResidualAttentionBlock.attention = attention
    onnx_model = clip_onnx(model)
    onnx_model.convert2onnx(image, text, verbose=True)
    print("convert is success") 
    
    ### inference ###
    start_time = time.time()
    onnx_model.start_sessions(providers=["CPUExecutionProvider"]) # cpu mode
    image_features = onnx_model.encode_image(image_onnx)
    text_features = onnx_model.encode_text(text_onnx)

    logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    print("Label probs:", probs)
    print("Prediction took {:.2f} seconds".format(time.time() - start_time))

def main(cfg):
    model, image, text = load_source(cfg)
    image_onnx = image.detach().cpu().numpy().astype(np.float32)
    text_onnx = text.detach().cpu().numpy().astype(np.int32)
    
    convert_inference(model, image, text, image_onnx, text_onnx)

if __name__ == '__main__':
    cfg = Cfg
    main(cfg)
