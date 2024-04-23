# Abstract

This is CLIP finetune model for image Classification task.

## About CLIP
<img src="https://github.com/madara-tribe/CLIP-finetune/assets/48679574/170d1b6f-0738-4932-8409-656dd17354d8" width="700px" height="300px"/>

## CLIP Accuracy
<img src="https://github.com/madara-tribe/CLIP-finetune/assets/48679574/b8714835-5ea5-42b4-9f42-993376099f81" width="700px" height="300px"/>


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
python3 onnx_export.py
```

```bash
# inference time
Prediction took 1.01 seconds
```


# improvent methods
- resize image size during training
- augumentation that fit dataset

# References
- [clip logic](https://arxiv.org/pdf/2103.00020.pdf)
- [kaggle](https://www.kaggle.com/code/zacchaeus/clip-finetune)
- [CLIP Onnx](https://www.kaggle.com/code/ivanpan/pytorch-clip-onnx-to-speed-up-inference)
- [OpenAI-CLIP](https://github.com/openai/CLIP)
