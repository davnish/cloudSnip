from torchgeo.datasets import RasterDataset
import torch
from torchvision.transforms import Compose
from torchgeo.samplers import BatchGeoSampler, RandomGeoSampler
import random
# from torchvision.transforms import Compose, ToTensor, Normalize, 
# from torchvision.transforms import Lambda
import torchvision.transforms.functional as F
from torchvision.transforms import v2
import torch.nn as nn
from torchgeo.transforms import AppendNDWI, AppendGRNDVI, AppendNDVI, AppendGNDVI
from transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# from rasvec import patchify_raster


train_transforms = Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    RandomHorizontalFlip(p=0.3), 
    RandomVerticalFlip(p=0.3), 
    RandomRotation(degrees=30),  

    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.2), 
    v2.RandomAutocontrast(p=0.2),  
    v2.ColorJitter(
        brightness=0.4,  
        contrast=0.4,
        saturation=0.4,
        hue=0.2
    ),
    v2.RandomApply(
        [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
        p=0.2 
    ),
    
    v2.Normalize(mean=[0.36576813, 0.3658635, 0.3988132],
                 std=[0.16295877, 0.17293826, 0.15380774]),
    # AppendNDWI(index_nir=0, index_green=2),
    # AppendNDVI(index_nir=0, index_red=1),
])

val_transforms = Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.36576813, 0.3658635, 0.3988132],
                 std=[0.16295877, 0.17293826, 0.15380774]),
])

class NoDataAware_RandomSampler(RandomGeoSampler):
    def __init__(self, dataset, size, length, nodata_value=0, max_nodata_ratio=0.4, **kwargs):
        super().__init__(dataset, size, length, **kwargs)
        self.nodata_value = nodata_value
        self.max_nodata_ratio = max_nodata_ratio
        self.dataset = dataset
    
    def __iter__(self):
        generated = 0
        while generated < self.length:
            # Get a random query from parent sampler
            query = next(super().__iter__())
            
            try:
                # Check the sample for nodata
                sample = self.dataset[query]
                if 'mask' in sample:
                    mask = sample['mask']
                    nodata_ratio = (mask == self.nodata_value).float().mean().item()
                    # Skip if too much nodata
                    if nodata_ratio >= self.max_nodata_ratio:
                        continue
                
                yield query
                generated += 1
            except Exception as e:
                # Skip problematic samples
                print(f"Skipping sample {query} due to error: {e}")
                continue

class Liss4(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^.{3}(?P<date>\d{2}[A-Z]{3}\d{4})'
    date_format = '%d%b%Y'
    single_band = False
    is_image = True
    separate_files = False
    # transforms = transforms

    def __getitem__(self, query):
        sample = super().__getitem__(query)

        image = sample["image"]
        image = torch.nan_to_num(image, nan=0.0)
        sample['chn_ids'] = torch.tensor([842, 665, 560])
        sample["imgs"] = image
        return sample

class Liss4_GT(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^.{3}(?P<date>\d{2}[A-Z]{3}\d{4})'
    date_format = '%d%b%Y'
    single_band = False
    is_image = False
    separate_files = False

    def __getitem__(self, query):
        sample = super().__getitem__(query)
        mask = sample["mask"]
        mask = torch.nan_to_num(mask, nan=0.0)
        mask = torch.where(mask < 0, torch.tensor(0, dtype=torch.int64), mask)
        mask = torch.where(mask > 2, torch.tensor(0, dtype=torch.int64), mask)
        sample["mask"] = mask
        return sample


train_img = Liss4("data/unprocessed_data/train/img")
train_label = Liss4_GT("data/unprocessed_data/train/label")

val_img = Liss4("data/unprocessed_data/test/img")
val_label = Liss4_GT("data/unprocessed_data/test/label")


train_dataset = train_img & train_label
train_dataset.transforms = train_transforms

val_dataset = val_img & val_label
val_dataset.transforms = val_transforms