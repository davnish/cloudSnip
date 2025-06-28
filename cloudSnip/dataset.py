from torchgeo.datasets import RasterDataset
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, 
from torchgeo.samplers import RandomGeoSampler
# from torchvision.transforms import Compose, ToTensor, Normalize, 
# from torchvision.transforms import Lambda

transforms = Compose([
    ToTensor(),
    # Lambda(lambda x: torch.nan_to_num(x, nan=0.0)),
    Normalize(mean=[0.36576813, 0.3658635, 0.3988132],
              std=[0.16295877, 0.17293826, 0.15380774])
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
    transforms = transforms

    def __getitem__(self, query):
        sample = super().__getitem__(query)

        # if "imgs" in sample:
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
val_dataset = val_img & val_label