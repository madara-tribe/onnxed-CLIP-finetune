import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.pin_memory=True
Cfg.num_worker = 2
Cfg.train_batch = 16 # must higher than num class
Cfg.val_batch = 1
Cfg.epochs = 100
Cfg.eval_step = 100
Cfg.input_size = 224


## hyperparameter
Cfg.scale_size = 1
Cfg.SCALE_SIZE = [0, 30]
Cfg.lr = 1e-5

## dataset
ROOT = "../place/conchita/datasets/fainal_proto/brand_ec_classes" #"datasets/dataset"
Cfg.num_class = 10
Cfg.root_train = os.path.join(ROOT, 'train/*/*.jpg')
Cfg.root_valid = os.path.join(ROOT, 'valid/*/*.jpg')
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')
