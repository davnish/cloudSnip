import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import panopticon_vitb14, Panopticon_Weights
from typing import Optional
import torch
from typing import Optional, Union, List, Tuple
# from mmseg.models.necks import Feature2Pyramid
# from mmseg.models.decode_heads import UPerHead
# from torchgeo.models import Panopticon

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dprob=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout2d(dprob),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout2d(dprob)
        )
    def forward(self, x):
        return self.conv(x)
    
class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)



class UNetDecoder(nn.Module):
    """The decoder part of a UNet architecture."""
    def __init__(
        self, 
        in_channels: int,
        layer_dimensions: List[int],
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            layer_dimensions (List[int]): List of output channels for each layer.
        """
        super().__init__()
        
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.num_layers = len(layer_dimensions)

        prev_channels = in_channels
        for out_channels in layer_dimensions:
            self.upsamples.append(nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=2, stride=2))
            self.convs.append(DoubleConv(out_channels+in_channels, out_channels))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Bottleneck features.
            skip_features (List[torch.Tensor]): List of skip features from encoder. 
        Returns:
            torch.Tensor: Output tensor after upsampling and convolution.
        """
        # More explicit indexing for torch.export compatibility
        for i in range(self.num_layers):
            x = self.upsamples[i](x)
            # Use explicit indexing instead of negative indexing
            skip_idx = len(skip_features) - 1 - i
            skip = skip_features[skip_idx]
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
        return x
    

class PanopticonUNet(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)

        for param in encoder.parameters():
            param.requires_grad = False

        # for name, param in encoder.named_parameters():
        #     if ".bias" in name:
        #         continue  # Biases are not frozen
        #     else:
        #         param.requires_grad = False

            
        self.num_classes = num_classes

        self.encoder = encoder.model
        edim = 768
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UNetDecoder(in_channels=edim, layer_dimensions=[256, 128, 64])
        
        self.head = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):

        B, _, _,_ = x['imgs'].shape


        x = self.encoder_intermediates(x, norm=True)


        x = [x.permute(0, 2, 1).reshape(B, -1, 16,16) for x in x]

        x = self.neck(x)

        x, skip_features = x[-1], x[:-1]

        x = self.decoder(x, skip_features)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

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
        # take_indices = indices

        # forward pass
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)


        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in indices:
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

    model = PanopticonUNet(num_classes=3)
    # output = model.encoder_intermediates(data, indices=[3, 5, 7, 11], norm=True)
    output = model(data)
    # print([i.shape for i in output])  

    # print(torch.cat(output, dim=1).shape
    
    print(output.shape)  # Should be (1, 3, 224, 224) for the final output