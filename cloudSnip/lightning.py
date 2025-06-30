import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from torchgeo.models import panopticon_vitb14, Panopticon_Weights
from typing import Optional, Union, List, Tuple
from models import DoubleConv, UNetDecoder
from transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

class cloudSnip(L.LightningModule):
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        imgs = batch['imgs']
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, batch['labels'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['imgs']
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, batch['labels'])
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


class PanopticonUNet(nn.Module):
    def __init__(self,num_classes=3, dropout=0.3):
        super().__init__()
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)

        for param in encoder.parameters():
            param.requires_grad = False
            
        self.encoder = encoder.model

        self.neck = DoubleConv(768, 512)

        self.decoder = UNetDecoder(512, dropout=dropout)

        self.head = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):

        B, _, _,_ = x['imgs'].shape


        x = self.encoder_intermediates(x, norm=True)

        x = torch.cat(x, dim=1)

        x = x.permute(0, 2, 1).reshape(B, -1, 32, 32)

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