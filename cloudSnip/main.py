import mlflow.pytorch
import torch
import torch.nn as nn
from model import PanopticonUNet
from dataset import train_dataset, val_dataset
from torch.utils.data import DataLoader
import time
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import stack_samples
import mlflow
import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score

parameters = dict(
    lr = 1e-4,
    weight_decay = 1e-4,
    epochs = 10,
    batch_size = 8,
)

mlflow.set_tracking_uri("https://infinite-clear-moose.ngrok-free.app")
mlflow.set_experiment("cloudSnip")

def collate_fn(samples):
    sample = stack_samples(samples)
    sample['imgs'] = sample['image']
    sample['chn_ids'] = torch.tensor([842, 665, 560]).repeat(parameters['batch_size'], 1)
    return sample

train_sampler = RandomGeoSampler(train_dataset, size=224, length=1000)
train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, collate_fn=collate_fn)

val_sampler = GridGeoSampler(val_dataset, size=224, stride=210)
val_dataloader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, collate_fn=collate_fn)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PanopticonUNet(in_ch=768, num_classes=3)
# model = torch.compile(uncompiled_model)

optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
criterion = nn.CrossEntropyLoss()   

def train_test():
    with mlflow.start_run():
        for epoch in range(parameters['epochs']):
            st = time.time()
            model.train()
            for batch in tqdm.tqdm(train_dataloader):
                masks = batch['mask'].squeeze()
                
                optimizer.zero_grad()
                outputs = model(batch)
                
                loss = criterion(outputs, masks.long())
                loss.backward()
                optimizer.step()

            # validation
            list_outputs = []
            list_masks = []
            with torch.no_grad():
                model.eval()
                for batch in tqdm.tqdm(val_dataloader):
                    masks = batch['mask'].squeeze()
                    outputs = model(batch)
                    # loss = criterion(outputs, masks.long())
                    list_outputs.append(outputs)
                    list_masks.append(masks)
            ed = time.time()

            # Concatenate all outputs and masks
            all_outputs = torch.cat(list_outputs)
            all_masks = torch.cat(list_masks)

            # Get predicted classes
            preds = torch.argmax(all_outputs, dim=1).cpu().numpy()
            targets = all_masks.cpu().numpy()

            # Flatten for metric calculation
            preds_flat = preds.flatten()
            targets_flat = targets.flatten()

            f1 = f1_score(targets_flat, preds_flat, average="macro")
            # Classwise precision and recall
            precision = precision_score(targets_flat, preds_flat, average=None, zero_division=0)
            recall = recall_score(targets_flat, preds_flat, average=None, zero_division=0)


            model_info = mlflow.pytorch.log_model(model, "model", registered_model_name="PanopticonUNet", input_example=batch)
            mlflow.log_params(parameters)
            # Log classwise metrics
            mlflow.log_metric("loss", loss.item(), step=epoch, model_id=model_info.model_id)
            for i, (p, r) in enumerate(zip(precision, recall)):
                mlflow.log_metric(f"precision_class_{i}", p, step=epoch, model_id=model_info.model_id)
                mlflow.log_metric(f"recall_class_{i}", r, step=epoch, model_id=model_info.model_id)
            acc = accuracy_score(targets_flat, preds_flat)
            mlflow.log_metric("f1_score", f1, step=epoch, model_id=model_info.model_id)
            mlflow.log_metric("accuracy", acc, step=epoch, model_id=model_info.model_id)
            mlflow.log_metric("epoch", epoch)

        print(f"Epoch [{epoch+1}/{parameters['epochs']}], Loss: {loss.item():.4f}, Time: {ed-st:.2f}s")

if __name__ == "__main__":
    train_test()
    print("Training complete.")
    # Save the model
    torch.save(model.state_dict(), "panopticon_unet.pth")
    print("Model saved as panopticon_unet.pth")