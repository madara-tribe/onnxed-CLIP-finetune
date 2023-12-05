import sys
import time
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify

from typing import Tuple
from cfg import Cfg
from utils.augmentations import transforms_
import clip


CLIP_ONNX_EXPORT_PATH = 'clip_resnet.onnx'
CLIP_ONNX_EXPORT_PATH_SIMP = 'clip_resnet_simplified.onnx'

ONNX_INPUT_NAMES = ["IMAGE", "TEXT"]
ONNX_OUTPUT_NAMES = ["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT"]
ONNX_DYNAMIC_AXES = {
    "IMAGE": {
        0: "image_batch_size",
    },
    "TEXT": {
        0: "text_batch_size",
    },
    "LOGITS_PER_IMAGE": {
        0: "image_batch_size",
        1: "text_batch_size",
    },
    "LOGITS_PER_TEXT": {
        0: "text_batch_size",
        1: "image_batch_size",
    },
}


def return_brand_ec(brand, ec):
    brand_ec = 0
    if brand==0 and ec==0:
        brand_ec = 0
    elif brand==0 and ec==1:
        brand_ec = 1
    elif brand==1 and ec==0:
        brand_ec = 2
    elif brand==1 and ec==1:
        brand_ec = 3
    elif brand==2 and ec==0:
        brand_ec = 4
    elif brand==2 and ec==1:
        brand_ec = 5
    elif brand==3 and ec==0:
        brand_ec = 6
    elif brand==3 and ec==1:
        brand_ec = 7
    elif brand==4 and ec==0:
        brand_ec = 8
    elif brand==4 and ec==1:
        brand_ec = 9
    return brand_ec

def load_clip(cfg, device, weight_path) -> Tuple[clip.model.CLIP, Tuple[torch.Tensor, torch.Tensor]]:
    num_cls = cfg.num_class
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.float()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to('cpu')
    npx = model.visual.input_resolution
    dummy_image = torch.randn(10, 3, npx, npx) #.to(device)
    dummy_texts = dummy_texts = clip.tokenize(["quick brown fox", "lorem ipsum"]) #.to(device) #torch.cat([clip.tokenize(f"a photo of a {i}") for i in range(num_cls)]).to(device)
    return model, (dummy_image, dummy_texts)


def export_onnx(
    model, 
    inputs, 
    input_names,
    output_names,
    dynamic_axes,
    export_path
) -> None:
    torch.onnx.export(
        model=model, 
        args=inputs, 
        f=export_path, 
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        dynamic_axes=dynamic_axes
    )



def main(cfg, device, weight_path):
    model, dummy_input = load_clip(cfg, device, weight_path)
    model.eval()
    
    export_onnx(
        model=model,
        inputs=dummy_input,
        input_names=ONNX_INPUT_NAMES,
        output_names=ONNX_OUTPUT_NAMES,
        dynamic_axes=ONNX_DYNAMIC_AXES,
        export_path=CLIP_ONNX_EXPORT_PATH,
    )
    
    ## run check
    onnx_model = onnx.load(CLIP_ONNX_EXPORT_PATH)
    onnx.checker.check_model(onnx_model)
    # run additional checks and simplify
    model_simp, check = simplify(onnx_model, skip_fuse_bn=True)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, CLIP_ONNX_EXPORT_PATH_SIMP)
    #ort_sess = ort.InferenceSession(CLIP_ONNX_EXPORT_PATH)
    #with torch.no_grad():
    #    pytorch_output = model(*dummy_input)
    #onnx_output = ort_sess.run(ONNX_OUTPUT_NAMES, {"IMAGE": dummy_input[0].numpy(), "TEXT": dummy_input[1].numpy()})
    #assert all([torch.allclose(pt_pred, torch.tensor(onnx_pred)) for pt_pred, onnx_pred in zip(pytorch_output, onnx_output)])

    #print(f'Pytorch output: {pytorch_output}\n\nONNX output: {onnx_output}')

if __name__=="__main__":
    if len(sys.argv)>1:
        weight_path = sys.argv[1]
    else:
        print("weight_path should be difined")
        sys.exit(1)
    cfg = Cfg
    img_path = 'teset.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(cfg, device, weight_path)



