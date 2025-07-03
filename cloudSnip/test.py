import torch
import tqdm
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from cloudSnip.main import read_yaml_to_dict
from cloudSnip.dataset import train_dataset, val_dataset, NoDataAware_RandomSampler
from torch.utils.data import DataLoader
from cloudSnip.model import PanopticonUNet
from cloudSnip.loss import CloudShadowLoss
from torchgeo.samplers import GridGeoSampler


parameters = read_yaml_to_dict("cloudSnip/config.yml")['parameters']

test_dataset = "data/unprocessed_data/val/img"
# test_sampler = NoDataAware_RandomSampler(test_dataset, size=224, length=parameters['length'], nodata_value=0, max_nodata_ratio=0.4)
test_sampler = GridGeoSampler(test_dataset, size=224, stride=224)

test_dataloader = DataLoader(test_dataset, batch_size=parameters['batch_size'], sampler=test_sampler)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = PanopticonUNet(num_classes=3)
model.to(device)
model.load_state_dict(torch.load("models/1view_wobitfit/trial_0.pth", map_location=device), strict=True)
criterion = CloudShadowLoss()
# list_outputs = []
# list_masks = []
with torch.no_grad():
    model.eval()
    for batch in tqdm.tqdm(test_dataloader):
        masks = batch['mask'].squeeze().to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(batch)
        # loss = criterion(outputs, masks.long())

        # list_outputs.append(outputs.detach().cpu())
        # list_masks.append(masks.detach().cpu())

        val_loss += loss.detach().cpu().item()
val_loss /= len(val_dataloader)
ed = time.time()