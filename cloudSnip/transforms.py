import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import v2

class AppendNDWI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        if 'image' in sample:
            band_green = sample['image'][2]
            band_nir = sample['image'][0]
            ndwi = (band_green - band_nir) / (band_green + band_nir + 1e-8)
            ndwi = ndwi.unsqueeze(0)  # Add channel dimension
            sample['image'] = torch.cat((sample['image'], ndwi), dim=0)
        return sample
    
class PerImageMinMaxNormalize:
    def __call__(self, sample):
        min_val = sample['image'].min()
        max_val = sample['image'].max()
        sample['image'] = (sample['image'] - min_val) / (max_val - min_val + 1e-8)  # avoid division by zero
        return sample


class AppendNDWI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        if 'image' in sample:
            band_green = sample['image'][2]
            band_nir = sample['image'][0]
            ndwi = (band_green - band_nir) / (band_green + band_nir + 1e-8)
            ndwi = ndwi.unsqueeze(0)  # Add channel dimension
            sample['image'] = torch.cat((sample['image'], ndwi), dim=0)
        return sample
    
class AppendNDVI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        if 'image' in sample:
            band_red = sample['image'][2]
            band_nir = sample['image'][1]
            ndvi = (band_red - band_nir) / (band_red + band_nir + 1e-8)
            ndvi = ndvi.unsqueeze(0)  # Add channel dimension
            sample['image'] = torch.cat((sample['image'], ndvi), dim=0)
        return sample
    
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
