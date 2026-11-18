import torch
import torch.nn as nn
from torchvision import models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Handle possible slight shape mismatch due to padding
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG16UNet(nn.Module):
    """
    U-Net with ImageNet-pretrained VGG-16 encoder for endoscopic polyp segmentation.
    Produces a single-channel logit map (passed through sigmoid for probability).
    """
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        features = list(vgg.features.children())

        # VGG-16 stages:
        # stage 1: conv1_1, conv1_2 -> indices 0..3 (64 channels)
        # stage 2: pool1, conv2_1, conv2_2 -> indices 4..8 (128 channels)
        # stage 3: pool2, conv3_1, conv3_2, conv3_3 -> indices 9..15 (256 channels)
        # stage 4: pool3, conv4_1, conv4_2, conv4_3 -> indices 16..22 (512 channels)
        # stage 5 / bridge: pool4, conv5_1, conv5_2, conv5_3 -> indices 23..29 (512 channels)
        self.stage1 = nn.Sequential(*features[:4])
        self.stage2 = nn.Sequential(*features[4:9])
        self.stage3 = nn.Sequential(*features[9:16])
        self.stage4 = nn.Sequential(*features[16:23])
        self.stage5 = nn.Sequential(*features[23:30])

        # Bottleneck / Center
        self.center = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder stages
        self.dec5 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.dec4 = DecoderBlock(in_channels=256, skip_channels=512, out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=256, out_channels=64)
        self.dec2 = DecoderBlock(in_channels=64, skip_channels=128, out_channels=32)
        self.dec1 = DecoderBlock(in_channels=32, skip_channels=64, out_channels=32)

        # Final head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.stage1(x)    # [B, 64, H, W]
        s2 = self.stage2(s1)   # [B, 128, H/2, W/2]
        s3 = self.stage3(s2)   # [B, 256, H/4, W/4]
        s4 = self.stage4(s3)   # [B, 512, H/8, W/8]
        s5 = self.stage5(s4)   # [B, 512, H/16, W/16]

        # Bottleneck
        center = self.center(s5) # [B, 512, H/32, W/32]

        # Decoder with skip connections
        d5 = self.dec5(center, s5) # [B, 256, H/16, W/16]
        d4 = self.dec4(d5, s4)     # [B, 128, H/8, W/8]
        d3 = self.dec3(d4, s3)     # [B, 64, H/4, W/4]
        d2 = self.dec2(d3, s2)     # [B, 32, H/2, W/2]
        d1 = self.dec1(d2, s1)     # [B, 32, H, W]

        logits = self.head(d1)
        return logits

