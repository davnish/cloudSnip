# import mlflow.pytorch
import torch
import torch.nn as nn
# from model import PanopticonUNet
from f2p_unet_model import PanopticonUNet
from dataset import train_dataset, val_dataset, NoDataAware_RandomSampler
from torch.utils.data import DataLoader
import time
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.datasets import stack_samples
import mlflow
import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
import optuna
from loss import CloudShadowLoss
from pathlib import Path
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss

mlflow.login()

experiment_name = "increasing_alpha"
mlflow.set_experiment(f"/Users/nischal.singh38@gmail.com/{experiment_name}")

def read_yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data

parameters = read_yaml_to_dict("cloudSnip/config.yml")['parameters']

def objective():

    # train_sampler = Rand(train_dataset, size=224, length=parameters['length'], nodata_value=0, max_nodata_ratio=0.4)
    train_sampler = RandomGeoSampler(train_dataset, length=parameters['length'], size=224)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=train_sampler, collate_fn=stack_samples, drop_last=True)
    val_sampler = GridGeoSampler(val_dataset, size=224, stride=210)
    val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], sampler=val_sampler, collate_fn=stack_samples, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PanopticonUNet(num_classes=3).to(device)
    # model.load_state_dict(torch.load("models/transform_augmentation/5dded9e090db4614830145f260aa5376/30.pth", map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
    # scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.00005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=0.00008)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = CloudShadowLoss()
    # criterion = CrossEntropyLoss()

    with mlflow.start_run(nested=True) as run:
        for k, v in parameters.items():
            mlflow.log_param(k, v)

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
            # model_info = mlflow.pytorch.log_model(model, "model", registered_model_name="PanopticonUNet", signature=signature)
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
            scheduler.step(val_loss)

            # new_weight_decay = parameters['weight_decay'] * (parameters['decay_rate'] ** epoch)
            # optimizer.param_groups[0]['weight_decay'] = new_weight_decay
            lr = optimizer.param_groups[0]['lr']

            print(f"Current: Learning rate = {lr:.6f}")
            if (epoch+1) % 10 == 0:
                # Save model checkpoint
                # mlflow.pytorch.log_mosdel(model, artifact_path=f"model_epoch_{epoch}")
                print(f"Saving model at epoch {epoch+1}...")
                # Save the model state_dict
                model_dir = Path(f"models/{experiment_name}/{run.info.run_id}")
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)

            # mlflow.log_param("model_path", model_path)
        # mlflow.pytorch.log_model(model, artifact_path="model")
    return val_loss

if __name__ == "__main__":

    objective()