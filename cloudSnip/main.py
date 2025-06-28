# import mlflow.pytorch
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
from mlflow.models.signature import infer_signature
import numpy as np

parameters = dict(
    lr = 1e-4,
    weight_decay = 1e-4,
    epochs = 150,
    batch_size = 64,
    length = 10000,
)

mlflow.login()
# mlflow.set_tracking_uri("https://infinite-clear-moose.ngrok-free.app")
mlflow.set_experiment("/Users/nischal.singh38@gmail.com/cloudSnip")

# input_example = {
#     'imgs': np.random.randn(8, 3, 224, 224),  # Example input tensor
#     'chn_ids': np.array([842, 665, 560]).repeat(8)  # Example channel IDs
# }
# output_example = np.random.randn(8, 3, 224, 224)  # Example output tensor
# signature = infer_signature(input_example, output_example) 

train_sampler = RandomGeoSampler(train_dataset, size=224, length=parameters['length'])
train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=train_sampler, collate_fn=stack_samples, num_workers=4)

val_sampler = GridGeoSampler(val_dataset, size=224, stride=210)
val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], sampler=val_sampler, collate_fn=stack_samples, num_workers=4)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = PanopticonUNet(in_ch=768, num_classes=3).to(device)
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
criterion = nn.CrossEntropyLoss()   

def train_test():
    with mlflow.start_run(run_name="inc_epoch50_dropout0.5"):
        for epoch in range(parameters['epochs']):
            st = time.time()
            model.train()
            train_loss = 0.0
            for batch in tqdm.tqdm(train_dataloader):
                masks = batch['mask'].squeeze().to(device)

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, masks.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().cpu().item()
            train_loss /= len(train_dataloader)

            # validation
            list_outputs = []
            list_masks = []
            val_loss = 0.0
            with torch.no_grad():
                model.eval()
                for batch in tqdm.tqdm(val_dataloader):
                    masks = batch['mask'].squeeze().to(device)
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    outputs = model(batch)
                    loss = criterion(outputs, masks.long())

                    list_outputs.append(outputs.detach().cpu())
                    list_masks.append(masks.detach().cpu())

                    val_loss += loss.detach().cpu().item()
            val_loss /= len(val_dataloader)
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
            accuracy = accuracy_score(targets_flat, preds_flat)

            # signature = infer_signature(input_example.numpy(), outputs.detach().cpu().numpy())
            # model_info = mlflow.pytorch.log_model(model, "workspace.default.v1", registered_model_name="PanopticonUNet", signature=signature)
            mlflow.log_params(parameters)
            # Log classwise metrics

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for i, (p, r) in enumerate(zip(precision, recall)):
                mlflow.log_metric(f"precision_class_{i}", p, step=epoch)
                mlflow.log_metric(f"recall_class_{i}", r, step=epoch)

            mlflow.log_metric("f1_score", f1, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("epoch", epoch)

            print(f"Epoch [{epoch+1}/{parameters['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}")
            print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy:.4f}")
            print(f"Time: {ed-st:.2f}s")
            schedule.step()

if __name__ == "__main__":
    train_test()
    print("Training complete.")
    # Save the model
    torch.save(model.state_dict(), "panopticon_unet.pth")
    print("Model saved as panopticon_unet.pth")