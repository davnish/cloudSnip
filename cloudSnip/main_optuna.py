# import mlflow.pytorch
import torch
import torch.nn as nn
from model import PanopticonUNet
from dataset import train_dataset, val_dataset, NoDataAware_RandomSampler
from torch.utils.data import DataLoader
import time
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples
import mlflow
import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
import optuna

mlflow.login()

experiment_name = "cloudSnip_transforms"
mlflow.set_experiment(f"/Users/nischal.singh38@gmail.com/{experiment_name}")


def objective(trial):
    # Define the hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 50, 200)
    batch_size = trial.suggest_int('batch_size', 32, 64)
    length = trial.suggest_int('length', 1000, 5000)

    parameters = dict(
        lr = lr,
        weight_decay = weight_decay,
        epochs = epochs,
        batch_size = batch_size,
        length = length,
    )

    train_sampler = NoDataAware_RandomSampler(train_dataset, size=224, length=parameters['length'], nodata_value=0, max_nodata_ratio=0.4)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=train_sampler, collate_fn=stack_samples, num_workers=4, drop_last=True)

    val_sampler = GridGeoSampler(val_dataset, size=224, stride=210)
    val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], sampler=val_sampler, collate_fn=stack_samples, num_workers=4, drop_last=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = PanopticonUNet(num_classes=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()



    with mlflow.start_run(nested=True) as run:


        for epoch in range(epochs):
            mlflow.log_param(parameters)
            mlflow.log_param("lr", optimizer.param_groups[0]['lr'])
            
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
            schedule.step()

            trial.report(val_loss, epoch)

            if trial.should_prune():
                mlflow.log_param("pruned", True)
                raise optuna.exceptions.TrialPruned()
            

        model_path = f"models/{experiment_name}/trial_{trial.number}.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_param("model_path", model_path)
        # mlflow.pytorch.log_model(model, artifact_path="model")
    return val_loss

if __name__ == "__main__":

    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
                                )  # or 'maximize' if using accuracy directly
    study.optimize(objective, n_trials=20)

    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")