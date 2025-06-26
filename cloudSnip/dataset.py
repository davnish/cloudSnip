from torchgeo.datasets import RasterDataset
from torchvision.transforms import Compose, ToTensor, Normalize

transforms = Compose([
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225])
])

class Liss4(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^.{3}(?P<date>\d{2}[A-Z]{3}\d{4})'
    date_format = '%d%b%Y'
    single_band = False
    is_image = True
    separate_files = False
    transforms = transforms

class Liss4_GT(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^.{3}(?P<date>\d{2}[A-Z]{3}\d{4})'
    date_format = '%d%b%Y'
    single_band = False
    is_image = False
    separate_files = False


train_img = Liss4("data/unprocessed_data/train/img")
train_label = Liss4_GT("data/unprocessed_data/train/label")

val_img = Liss4("data/unprocessed_data/test/img")
val_label = Liss4_GT("data/unprocessed_data/test/label")


train_dataset = train_img & train_label
val_dataset = val_img & val_label