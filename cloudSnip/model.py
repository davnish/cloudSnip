import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import panopticon_vitb14, Panopticon_Weights
from typing import Optional
import torch
from typing import Union, List, Tuple
# from torchgeo.models import Panopticon

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dprob=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dprob),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dprob)
        )
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dprob=0.3, scale_factor=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=scale_factor, stride=scale_factor)
        self.conv = DoubleConv(in_ch, out_ch, dprob)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_ch, dropout=0.3):
        super().__init__()
        self.dec1 = DecoderBlock(in_ch, 256, dprob=dropout)

        self.bridge = nn.Conv2d(256, 256, 3, padding=1)

        self.dec2 = DecoderBlock(256, 128, dprob=dropout)
        self.dec3 = DecoderBlock(128, 64, dprob=dropout)


    def forward(self, x):
        x = self.dec1(x)  # 32x32 -> 64x64

        x = F.interpolate(x, size=(56,56), mode='bilinear', align_corners=False) # 64x64 → 56x56
        x = self.bridge(x)  # DoubleConv to process the features

        x = self.dec2(x) #56x56 -> 112x112

        x = self.dec3(x) # 112x112 -> 224x224
        return x
    
class PanopticonUNet(nn.Module):
    def __init__(self,num_classes=3, dropout=0.3):
        super().__init__()
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)

        for param in encoder.parameters():

            param.requires_grad = False
        
        # for name, param in encoder.named_parameters():
        #     if ".bias" in name:
        #         continue  # Biases are not frozen
        #     else:
        #         param.requires_grad = False
            

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
    
if __name__ == "__main__":
    import torch
    # Example usage
    img = torch.randn(1,3,224,224)
    chn_ids = torch.tensor([842, 665, 560]).repeat(1, 1)

    data = dict(
        imgs =  img,
        chn_ids = chn_ids
    )

    model = PanopticonUNet(decoder_in_ch=768, num_classes=3)
    # output = model.encoder_intermediates(data, indices=[3, 5, 7, 11], norm=True)
    output = model(data)
    # print([i.shape for i in output])  

    # print(torch.cat(output, dim=1).shape
    
    print(output.shape)  # Should be (1, 3, 224, 224) for the final output