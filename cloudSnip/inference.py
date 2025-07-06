import torch
import tqdm
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from dataset import train_dataset, val_dataset, NoDataAware_RandomSampler
from torch.utils.data import DataLoader
# from model import PanopticonUNet
from f2p_unet_model import PanopticonUNet
from loss import CloudShadowLoss
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples
from dataset import Liss4
import yaml
import matplotlib.pyplot as plt
import numpy as np
from dataset import val_transforms

experiment = "increasing_alpha"
run_id = "afbbd4b9d73041e28b69eca999850be0"
model_no = "20"
def read_yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


parameters = read_yaml_to_dict("cloudSnip/config.yml")['parameters']

inference_dataset = Liss4("data/unprocessed_data/val/img")
inference_sampler = GridGeoSampler(inference_dataset, size=224, stride=224)
inference_sampler.transforms = val_transforms

inference_dataloader = DataLoader(inference_dataset, batch_size=parameters['batch_size'], sampler=inference_sampler, collate_fn=stack_samples)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = PanopticonUNet(num_classes=3)
model.to(device)
model.load_state_dict(torch.load(f"models/{experiment}/{run_id}/{model_no}.pth", map_location=device), strict=True)

with torch.no_grad():
    model.eval()
    for batch in tqdm.tqdm(inference_dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # print(batch["mask"])
        outputs = model(batch)

        # plt.imshow
        # print(outputs[0].cpu().numpy().transpose(1, 2, 0))
        idx = 21
        fix, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(batch['imgs'][idx].cpu().numpy().transpose(1, 2, 0))
        ax[0].set_title("Input Image")
        ax[1].imshow(outputs[idx].argmax(dim=0).cpu().numpy())
        ax[1].set_title("Predicted Mask")
        plt.show()
        plt.savefig(f"{experiment}.png")
        break
ed = time.time()