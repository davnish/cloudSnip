import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import panopticon_vitb14, Panopticon_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
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
    def __init__(self, in_ch=3, num_classes=3):
        super().__init__()
        encoder = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)
        for param in encoder.parameters():
                param.requires_grad = False
        self.encoder = encoder.model

        self.decoder = UNetDecoder(in_ch, num_classes)
    
    def forward(self, x):

        x = self.encoder.forward_features(x)
        # B, 257, 768 -> B, 768, 16, 16
        x = x[:, 1:,:].permute(0, 2, 1).reshape(x.shape[0], -1, 16, 16)
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    import torch
    # Example usage
    model = PanopticonUNet(in_ch=768, num_classes=3)
    model = torch.compile(model, fullgraph=True)