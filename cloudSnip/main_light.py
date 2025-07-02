
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
from torchgeo.models import panopticon_vitb14, Panopticon_Weights
from typing import Optional, Union, List, Tuple
from model import DoubleConv, UNetDecoder
from loss import CloudShadowLoss
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from dataset import NoDataAware_RandomSampler, train_dataset, val_dataset
from torchgeo.datasets import stack_samples
from pathlib import Path
from lightning import Trainer, seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
seed_everything(42, workers=True)

model_params = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'step_size': 50,
    'num_classes': 3,
    'dropout': 0.1
}

dataset_params = {
    'length': 224,
    'batch_size': 16,
}

mlflow.login()
experiment_name = "lightning"
mlf_logger = MLFlowLogger(f"/Users/nischal.singh38@gmail.com/{experiment_name}")

train_sampler = NoDataAware_RandomSampler(train_dataset, size=224, length=dataset_params['length'], nodata_value=0, max_nodata_ratio=0.4)
train_loader = DataLoader(train_dataset, batch_size=dataset_params['batch_size'], sampler=train_sampler, collate_fn=stack_samples, drop_last=True)

val_sampler = NoDataAware_RandomSampler(val_dataset, size=224, length=dataset_params['length'], nodata_value=0, max_nodata_ratio=0.4)
val_loader = DataLoader(val_dataset, batch_size=dataset_params['batch_size'], sampler=val_sampler, collate_fn=stack_samples, drop_last=True)

class PanopticonUNet(L.LightningModule):
    def __init__(self, lr, weight_decay, dropout, num_classes=3):
        super().__init__()

        self.save_hyperparameters()
        self.criterion = CloudShadowLoss()


        #Model
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder.model
        self.neck = DoubleConv(768, 512)
        self.decoder = UNetDecoder(512, dropout=dropout)
        self.head = nn.Conv2d(32, num_classes, 3, padding=1)

        # Metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.train_recall = MulticlassRecall(num_classes=num_classes, average='macro')

        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')

    def forward(self, x):
        B, _, _,_ = x['imgs'].shape
        x = self.encoder_intermediates(x, indices = [11], norm=True)[0]
        x = x.permute(0, 2, 1).reshape(B, -1, 16, 16)
        x = self.neck(x)
        x = self.decoder(x)
        x = self.head(x)
        return x
    
    def encoder_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = [3, 5, 7, 11],
            norm: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Whether to apply normalization to the output intermediates
        Returns:

        """
        intermediates = []
        take_indices = indices
        # forward pass
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in take_indices:
                intermediates.append(self.encoder.norm(x) if norm else x)
        intermediates = [y[:, self.encoder.num_prefix_tokens:] for y in intermediates]
        return intermediates
    
    def training_step(self, batch, batch_idx):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = self(batch)
        loss = self.criterion(preds, batch['labels'])
        self.log('train_loss', loss)
        self.log("train_f1", self.train_f1(preds, batch['labels']), on_epoch=True)
        self.log("train_precision", self.train_precision(preds, batch['labels']), on_epoch=True)
        self.log("train_recall", self.train_recall(preds, batch['labels']), on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = self(batch)
        loss = self.criterion(preds, batch['labels'])
        self.log('val_loss', loss)
        self.log("val_f1", self.val_f1(preds, batch['labels']), on_epoch=True)
        self.log("val_precision", self.val_precision(preds, batch['labels']), on_epoch=True)
        self.log("val_recall", self.val_recall(preds, batch['labels']), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5,threshold=1e-4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}


def objective(trial):
    model = PanopticonUNet(lr=model_params['lr'],
                           weight_decay=model_params['weight_decay'],
                           dropout=model_params['dropout'],
                           num_classes=model_params['num_classes'])
    trainer = Trainer(
        max_epochs=model_params['epochs'],
        accelerator='gpu',
        logger=mlf_logger,
    )
    # Start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == "__main__":

    objective(None)




