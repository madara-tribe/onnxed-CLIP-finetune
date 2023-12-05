# Abstract

This is CLIP finetune prototype for image Classification task.


# Perfomance

## Dataset
- [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

| Model | Head | Pretrain | class | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |        ---: |
| resnext50d_32x4d(timm) | fc | imageNet |69|74.09%|
| CLIP-finetune | None | CLIP |69|90.42%|


## train/eval
```bash
python3 main.py --mode train (-w <weight path>)
python3 main.py --mode eval -w <weight path>
```

## ONNX convert (test step)
```bash
python3 onnx_export.py <weight_path>
```

# improvent methods
- resize image size during training
- augumentation that fit dataset

# References
- [clip logic](https://arxiv.org/pdf/2103.00020.pdf)
- [kaggle](https://www.kaggle.com/code/zacchaeus/clip-finetune)
- [CLIP Onnx](https://www.kaggle.com/code/ivanpan/pytorch-clip-onnx-to-speed-up-inference)
- [OpenAI-CLIP](https://github.com/openai/CLIP)
