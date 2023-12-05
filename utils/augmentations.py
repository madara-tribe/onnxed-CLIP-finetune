import albumentations as A
from albumentations.pytorch import ToTensorV2

def transforms_(cfg):
    train_transform = A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.HorizontalFlip(),
            A.Rotate((0, 20)),
            # Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    val_transform = A.Compose([
                A.Resize(cfg.input_size, cfg.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])
    return train_transform, val_transform

