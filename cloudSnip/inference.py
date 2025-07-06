import torch
import tqdm
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from dataset import train_dataset, val_dataset, NoDataAware_RandomSampler
from torch.utils.data import DataLoader
from model import PanopticonUNet
from loss import CloudShadowLoss
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples
from dataset import Liss4
import yaml
import matplotlib.pyplot as plt
import numpy as np
from dataset import val_transforms
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import rasterio as rio
from torchvision.transforms import v2, Compose


model_path = "/home/nischal/projects/cloudSnip/models/increasing_data/548a21f452d84499a3c84f94a07a8137/best/1.pth"

def read_yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


parameters = read_yaml_to_dict("cloudSnip/config.yml")['parameters']
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = PanopticonUNet(num_classes=3)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)


def get_image_profile(path):
    with rio.open(path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def preprocess_image(image):
    image = torch.tensor(image)
    image = torch.nan_to_num(image, nan=0.0)
    transforms = Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.36576813, 0.3658635, 0.3988132],
                     std=[0.16295877, 0.17293826, 0.15380774]),
    ])
    return transforms(image)

def pad_to_fit(image, patch_size, stride):
    _, h, w = image.shape
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    return F.pad(image, (0, pad_w, 0, pad_h))


def get_patches(image, patch_size=224, stride=160):  # overlap = patch_size - stride
    _, h, w = image.shape
    patches = []
    positions = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    return torch.stack(patches), positions

def predict_on_patches(model, patches, device):
    model.eval()
    preds = []


    with torch.no_grad():
        for patch in patches:
            patch_dict = dict(
                imgs = patch.unsqueeze(0).to(device),
                chn_ids = torch.tensor([[842, 665, 560]]).to(device),
            ) 
            pred = model(patch_dict)  # shape: [1, C, H, W]
            preds.append(pred.cpu().squeeze(0))  # shape: [C, H, W]
    
    return preds  # list of tensors


def merge_patches(preds, positions, full_shape, patch_size=224, stride=160):
    C, H, W = preds[0].shape[0], full_shape[0], full_shape[1]
    output = torch.zeros((C, H, W))
    count = torch.zeros((C, H, W))

    for pred, (y, x) in zip(preds, positions):
        output[:, y:y+patch_size, x:x+patch_size] += pred
        count[:, y:y+patch_size, x:x+patch_size] += 1
    
    output /= count.clamp(min=1e-8)
    return output  # shape: [C, H, W]

def save_map(path, segmentation_map, profile):
    clip_map = segmentation_map.numpy().astype(np.uint8)[:profile['height'], :profile['width']]
    with rio.open(path, 'w', **profile) as dst:
        dst.write(clip_map, 1)  # Assuming single channel output


def get_segmentation_map(input_img, output_img):
    image, profile = get_image_profile(input_img)
    image = preprocess_image(image)  # shape: [C, H, W]
    image = pad_to_fit(image, patch_size=224, stride=160)  # shape: [C, H, W]
    patches, positions = get_patches(image, patch_size=224, stride=160)
    preds = predict_on_patches(model, patches, device)
    output = merge_patches(preds, positions, image.shape[1:], patch_size=224, stride=160)
    segmentation_map = torch.argmax(output, dim=0)  # shape [H, W]
    save_map(output_img, segmentation_map, profile)

if __name__ == "__main__":
    for input_imag in Path("data/unprocessed_data/val/img").glob("*.tif"):
        output_imag = Path("test_segmentation_map/") / input_imag.name
        get_segmentation_map(input_imag, output_imag)