import torch
import torch.nn as nn
from typing import List, Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetEncoder(nn.Module):
    """The encoder part of a UNet architecture."""
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

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in layer_dimensions:
            self.convs.append(DoubleConv(prev_channels, out_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Output tensor and intermediate features.
        """
        intermediate_features = []
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x)
            intermediate_features.append(x)
            x = pool(x)
        return x, intermediate_features
    
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
            self.convs.append(DoubleConv(out_channels*2, out_channels))
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
    
class UNet(nn.Module):
    """Vanilla UNet architecture for image segmentation."""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layer_dimensions: List[int] = [64, 128, 256, 512],
    ) -> None:
        """  
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            layer_dimensions (List[int], optional): List of output channels for each layer. 
                Defaults to [64, 128, 256, 512]. `len(layer_dimensions)` will determine the depth of the UNet.
        """
        super().__init__()

        self.encoder = UNetEncoder(in_channels, layer_dimensions)

        self.neck = DoubleConv(layer_dimensions[-1], layer_dimensions[-1]*2)

        # Create decoder dimensions in reverse order
        decoder_dims = list(reversed(layer_dimensions))
        self.decoder = UNetDecoder(in_channels=layer_dimensions[-1]*2, layer_dimensions=decoder_dims)

        self.head = nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Getting features from the encoder
        x, skip_features = self.encoder(x)

        # neck
        x = self.neck(x)

        # Decode with skip connections
        output = self.decoder(x, skip_features)

        return self.head(output)

