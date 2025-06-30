import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import v2


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
