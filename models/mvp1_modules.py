import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FrozenVGG(nn.Module):
    """冻结的 VGG16 特征编码器（MVP-1 用，代替两阶段预训练编码器）。

    输入: [-1, 1] 范围的 RGB 图像 (B,3,H,W)
    输出: [relu2_2, relu3_3, relu4_3] 三层特征（以 512×512 输入为例）
          relu2_2: 128ch, 1/2 分辨率 (256×256) -> 作为 R_sig（残差候选信号来源）
          relu3_3: 256ch, 1/4 分辨率 (128×128) -> NCE 中层特征
          relu4_3: 512ch, 1/8 分辨率 (64×64)   -> 作为 S（结构骨架特征）

    注意：VGG16 的池化在 features 的第 4/9/16/23 层，
    第 8 层之后只池化过 1 次，第 22 层之后池化过 3 次。
    """

    IDX = (8, 15, 22)  # relu2_2, relu3_3, relu4_3 在 vgg16.features 中的层号

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # [-1,1] -> [0,1] -> ImageNet 归一化
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.IDX:
                feats.append(h)
            if i == self.IDX[-1]:
                break
        return feats


class ProjModule(nn.Module):
    """Proj 筛选回归残差模块（唯一可训练的前置模块）。

    输入: R_sig (B,128,H/2,W/2)  浅层特征（混杂蛋白线索与噪声）
    输出: 增量 (B,512,H/8,W/8)，与骨架 S 相加得到 S' = S + Proj(R_sig)
    （两次 stride-2 卷积完成 1/2 -> 1/8 的分辨率对齐）
    """

    def __init__(self, in_ch=128, out_ch=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )

    def forward(self, r):
        return self.net(r)


class _UpsampleBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureDecoder(nn.Module):
    """特征->图像 渲染器（MVP-1 生成器）。只做画风渲染，不接触原始 AF 像素。

    输入: S'.detach() (B,512,H/8,W/8)
    输出: fake IHC (B,3,H,W), tanh 到 [-1,1]
    （3 次上采样：1/8 -> 1/4 -> 1/2 -> 1/1）
    """

    def __init__(self, in_ch=512, ngf=64, out_ch=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ngf * 4, 3, padding=1),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        )
        self.up1 = _UpsampleBlock(ngf * 4, ngf * 2)  # -> 1/4
        self.up2 = _UpsampleBlock(ngf * 2, ngf)      # -> 1/2
        self.up3 = _UpsampleBlock(ngf, ngf)          # -> 1/1
        self.tail = nn.Sequential(nn.Conv2d(ngf, out_ch, 3, padding=1), nn.Tanh())

    def forward(self, s):
        h = self.head(s)
        h = self.up3(self.up2(self.up1(h)))
        return self.tail(h)
