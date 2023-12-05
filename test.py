import os
import torch
import time
from tqdm import tqdm
from torch.utils import data
from utils.augmentations import transforms_
from utils.dataloader import DataLoader
from tools import create_cofusion_matrix
import clip


class BasePredictor:
    def __init__(self, cfg, device, weight_path):
        print("start predict")

    def create_data_loader(self, config):
        _, val_transform = transforms_(config)
        val_dst = DataLoader(root=config.root_valid, transform=val_transform)

        val_loader = data.DataLoader(
                    val_dst, batch_size=config.val_batch, shuffle=None, num_workers=config.num_worker,
                    pin_memory=None)
        print(" Query set: %d" %(len(val_dst)))
        return val_loader, val_dst

    def load_trained_model(self, cfg, device, weight_path):
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.float()
        model.load_state_dict(torch.load(weight_path, map_location=device))
        return model

class Predictor(BasePredictor):
    def __init__(self, cfg, device, weight_path):
        model = self.load_trained_model(cfg, device, weight_path)
        self.predict(model, cfg, device)
    
    def predict(self, model, cfg, device):
        filename='results'
        sub_dir = 'miss'
        main_label_dir = os.path.join(filename, 'main')
        val_loader, _ = self.create_data_loader(cfg)
        model.eval()
        num_cls = cfg.num_class
        y_pred, y_test = [], []
        start = time.time()
        for i, (imgs, label) in tqdm(enumerate(val_loader)):
            imgs = imgs.to(device=device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {i}") for i in range(num_cls)]).to(device)
            logits_per_image, logits_per_text = model(imgs, text_inputs) 
            label = int(label.to('cpu').detach().numpy().copy())
            pred_label = int(torch.argmax(logits_per_image.softmax(dim=-1)).to('cpu').detach().numpy().copy())
            # brand accuracy
            y_pred.append(pred_label)
            y_test.append(label)
        latency = time.time() - start
        print(f'Prediction Latency: {latency}') 
        # cm
        brand_name = [str(i) for i in range(10)]
        create_cofusion_matrix(y_test, y_pred, target_names=brand_name, filename=main_label_dir)