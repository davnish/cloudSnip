import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import panopticon_vitb14, Panopticon_Weights
from typing import Optional
import torch
from typing import Optional, Union, List, Tuple
from torchgeo.models import Panopticon

# class UNetDecoder(nn.Module):
#     """The decoder part of a UNet architecture.
#     This module is designed to stand alone or be used as part of a UNet architecture.
#     It is designed to upsample features from the bottleneck. Every layer upsamples the input by a factor of 2,
#     and optionally can be combined with skip connections from the encoder to produce a high-resolution output.
#     """
#     def __init__(
#         self, 
#         in_channels: int,
#         layer_dimensions: list[int] = [512, 256, 128, 64],
#         encoder_features_dims: Optional[list[int]] = None, # [64, 128, 256, 512]
#     ) -> None:
#         super().__init__()
#         self.encoder_features_dims = encoder_features_dims

#         if self.encoder_features_dims and len(self.encoder_features_dims) != len(layer_dimensions):
#             raise ValueError("encoder_features_dims must match the number of decoder layers.")
        
#         self.upsamples = nn.ModuleList()
#         self.convs = nn.ModuleList()

#         prev_channels = in_channels
#         for i, out_channels in enumerate(layer_dimensions):
#             self.upsamples.append(
#                 nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=2, stride=2)
#             )
#             if self.encoder_features_dims:
#                 self.convs.append(DoubleConv(out_channels + self.encoder_features_dims[-(i+1)], out_channels))
#             else:
#                 self.convs.append(DoubleConv(out_channels, out_channels))

#             prev_channels = out_channels

    
#     def forward(self, x: torch.Tensor, skip_features: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
#         """
#         Args:
#             x: Bottleneck features
#             skip_features (Optional[list[torch.Tensor]]): List of skip features from encoder, must match the number of decoder layers, 
#             if `encoder_features_dims` is set this should be provided.
#         Returns:
#             torch.Tensor: Output tensor after upsampling and convolution.
#         """

#         if self.encoder_features_dims:
#             if skip_features is None:
#                 raise ValueError("skip_features must be provided when encoder_features_dims is set.")
#             elif len(skip_features) != len(self.encoder_features_dims):
#                 raise ValueError(
#                     f"Expected {len(self.encoder_features_dims)} skip connections, "
#                     f"but got {len(skip_features)}."
#                 )
        
#         for i, (upsample, conv) in enumerate(zip(self.upsamples, self.convs)):
#             x = upsample(x)

#             if skip_features:
#                 feature_idx = -(1 + i)

#                 skip = skip_features[feature_idx]
                
#                 # Handle size mismatches
#                 if x.shape[-2:] != skip.shape[-2:]:
#                     diff_h = skip.shape[-2] - x.shape[-2]
#                     diff_w = skip.shape[-1] - x.shape[-1]
                    
#                     if diff_h != 0 or diff_w != 0:
#                         # Center crop skip connection to match x
#                         h_start = max(0, diff_h // 2)
#                         w_start = max(0, diff_w // 2)
#                         h_end = h_start + x.shape[-2]
#                         w_end = w_start + x.shape[-1]
#                         skip = skip[:, :, h_start:h_end, w_start:w_end]
#                 x = torch.cat([x, skip], dim=1)

#             x = conv(x)
#         return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dprob=0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
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
        x = self.up(x)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.dec1 = DecoderBlock(in_ch, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.dec1(x)  # e.g. 16x16 → 32x32
        x = self.dec2(x)  # 32x32 → 64x64
        x = F.interpolate(x, size=(56,56), mode='bilinear', align_corners=False)
        x = self.dec3(x)  # 56x56 → 112x112
        x = self.dec4(x)  # 112x112 → 224x224
        return self.out(x)
    
class PanopticonUNet(nn.Module):
    def __init__(self, decoder_in_ch=768, num_classes=3):
        super().__init__()
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)

        for param in encoder.parameters():
            param.requires_grad = False
        
        self.encoder = encoder.model

        self.decoder = UNetDecoder(decoder_in_ch, num_classes)
    
    def forward(self, x):


        x = self.encoder_intermediates(x)


        x = torch.cat(x, dim=2)
        print(x.shape)
        print(x)


        # print(x == blocks[-1])

        # B, 257, 768 -> B, 768, 16, 16
        # print(x)
        # x = x[:, 1:,:].permute(0, 2, 1).reshape(x.shape[0], -1, 16, 16)
        # x = self.decoder(x)
        return x
    
    def encoder_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Whether to apply normalization to the output intermediates
        Returns:

        """
        B, _, _,_ = x['imgs'].shape

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
        intermediates = [y.reshape(B, 16, 16, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

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
    output = model.encoder_intermediates(data, indices=[3, 5, 7, 11], norm=True)
    print([i.shape for i in output])  

    print(torch.cat(output, dim=1).shape)