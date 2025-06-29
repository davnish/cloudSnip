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

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if random.random() < self.p:
            if 'image' in sample:
                sample['image'] = F.hflip(sample['image'])
            if 'mask' in sample:
                sample['mask'] = F.hflip(sample['mask'])
        return sample

class RandomVerticalFlip(nn.Module):
    """Randomly flips the image and mask vertically with a given probability.
    Args:
        p (float): Probability of flipping the image and mask. Default is 0.5
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if random.random() < self.p:
            if 'image' in sample:
                sample['image'] = F.vflip(sample['image'])
            if 'mask' in sample:
                sample['mask'] = F.vflip(sample['mask'])
        return sample

class RandomRotation(nn.Module):
    """Randomly rotates image and mask by an angle within (-degrees, +degrees)."""
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)

    def forward(self, sample):
        angle = random.uniform(*self.degrees)
        
        if 'image' in sample:
            sample['image'] = F.rotate(
                sample['image'],
                angle,
                interpolation=F.InterpolationMode.BILINEAR,
                expand=False,
                fill=0
            )
        if 'mask' in sample:
            sample['mask'] = F.rotate(
                sample['mask'].unsqueeze(0),  # Ensure mask is 4D
                angle,
                interpolation=F.InterpolationMode.NEAREST,  # Important!
                expand=False,
                fill=0
            ).squeeze(0)  # Remove the added dimension
        return sample


train_transforms = Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    RandomHorizontalFlip(p=0.3),  # less frequent flipping
    RandomVerticalFlip(p=0.3),  # optional: very rare in natural scenes
    RandomRotation(degrees=30),  # subtle rotation if needed

    v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.1), 
    v2.RandomAutocontrast(p=0.1),  # reduced chance
    v2.ColorJitter(
        brightness=0.1,  # small tweaks
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    v2.RandomApply(
        [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
        p=0.1  # less frequent and gentler blur
    ),
  
    v2.Normalize(mean=[0.36576813, 0.3658635, 0.3988132],
                 std=[0.16295877, 0.17293826, 0.15380774]),
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