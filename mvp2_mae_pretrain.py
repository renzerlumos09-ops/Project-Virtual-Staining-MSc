# -*- coding: utf-8 -*-
"""
mae_pretrain.py — MVP-2 Stage 1: IHC 域掩码图像建模（MIM）预训练
=========================================================
目的：在 trainB（IHC, ~52k 张 512x512 PNG）上预训练一个与 FrozenVGG 同构的
VGG16 features 编码器，产出权重可无缝替换 MVP-1 中的 ImageNet 冻结编码器

方法：SimMIM 式掩码图像建模
- 输入归一化到 [-1, 1]
- 随机遮住 mask_ratio 比例的 32x32 块（置 0 = 中灰）
- 编码器取 relu4_3 特征（512ch @ 64x64），轻量解码器重建整图
- L1 损失只计算被遮住的像素

用法（CMD，项目根目录下）：
python mvp2_mae_pretrain.py --dataroot ./datasets/prostate_data/trainB \
--name mae_ihc_v1 --batch_size 8 --n_epochs 30

输出：
checkpoints/<name>/mae_encoder_latest.pth      编码器权重（VGG16 features 格式）
checkpoints/<name>/vis/epochXXX.png            每 epoch 重建对比图（原图/遮挡/重建）
checkpoints/<name>/loss_log.txt                训练日志
"""

import argparse
import glob
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, utils as vutils


# ------------------------------------------------------------------ dataset
class IHCDataset(Dataset):
    """读取单目录下的 IHC PNG，归一化到 [-1, 1]。"""

    def __init__(self, roots, max_n=None):
        #self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.paths = []
        for r in roots.split(","):
            self.paths += sorted(glob.glob(os.path.join(r.strip(), "*.png")))
        if max_n is not None:
            self.paths = self.paths[:max_n]
        assert len(self.paths) > 0, f"no png found in {roots}"
        self.tf = transforms.Compose([
            transforms.ToTensor(),                    # [0,1]
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # -> [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


# -------------------------------------------------------------------- mask
def random_block_mask(batch, grid, ratio, size, device):
    """返回 bool 掩码 (B,1,size,size)，True = 被遮住。
    grid x grid 个块，每块 size/grid 像素。"""
    n = grid * grid
    k = int(n * ratio)
    idx = torch.argsort(torch.rand(batch, n), dim=1)[:, :k]          # B,k
    m = torch.zeros(batch, n, dtype=torch.bool)
    m.scatter_(1, idx, True)
    m = m.view(batch, 1, grid, grid).float()
    m = nn.functional.interpolate(m, size=(size, size), mode="nearest")
    return m.bool().to(device)


# ------------------------------------------------------------------- model
class MIMEncoder(nn.Module):
    """VGG16 features 主干（与 FrozenVGG 完全同构）。
    forward 返回三层特征：relu2_2 / relu3_3 / relu4_3。"""

    TAP_LAYERS = (8, 15, 22)  # 与 mvp1_modules.FrozenVGG 保持一致

    def __init__(self, init="imagenet"):
        super().__init__()
        weights = "IMAGENET1K_V1" if init == "imagenet" else None
        self.features = models.vgg16(weights=weights).features

    def forward(self, x):
        outs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.TAP_LAYERS:
                outs[i] = x
        return outs[8], outs[15], outs[22]


class MIMDecoder(nn.Module):
    """轻量重建解码器：512ch@64x64 -> 3ch@512x512。"""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(True),
            )

        self.dec = nn.Sequential(
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, f):
        return self.dec(f)


class MIMModel(nn.Module):
    def __init__(self, init="imagenet"):
        super().__init__()
        self.encoder = MIMEncoder(init)
        self.decoder = MIMDecoder()

    def forward(self, x):
        _, _, f8 = self.encoder(x)   # 只用最深一层做重建
        return self.decoder(f8)


# -------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True, help="IHC png 目录，如 ./datasets/prostate_data/trainB")
    p.add_argument("--name", default="mae_ihc_v1")
    p.add_argument("--checkpoints_dir", default="./checkpoints")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--grid", type=int, default=16, help="16x16 个 32px 块")
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--init", default="imagenet", choices=["imagenet", "scratch"])
    p.add_argument("--max_images", type=int, default=None, help="调试用：只用前 N 张")
    p.add_argument("--num_threads", type=int, default=0, help="Windows 下保持 0")
    p.add_argument("--save_epoch_freq", type=int, default=5)
    p.add_argument("--resume", default="", help="从指定权重继续训练（分段跑用）")
    p.add_argument("--epoch_offset", type=int, default=0, help="从指定 epoch 开始计数（分段跑用）")
    opt = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    vis_dir = os.path.join(save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "loss_log.txt")

    dataset = IHCDataset(opt.dataroot, opt.max_images)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.num_threads, drop_last=True)

    model = MIMModel(opt.init).to(device)
    if opt.resume:
        model.encoder.features.load_state_dict(torch.load(opt.resume, map_location="cpu"))
        print(f"[resume] encoder weights loaded from {opt.resume}")
        dec_path = os.path.join(os.path.dirname(opt.resume), "mae_decoder_latest.pth")
        if os.path.exists(dec_path):
            model.decoder.load_state_dict(torch.load(dec_path, map_location="cpu"))
            print(f"[resume] decoder weights loaded from {dec_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05)
    total_iters = opt.n_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: 0.5 * (1 + torch.cos(torch.tensor(math_pi * it / total_iters)).item()))
    scaler = torch.amp.GradScaler("cuda")

    with open(log_path, "a") as f:
        f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {vars(opt)}\n")

    print(f"dataset: {len(dataset)} images, {len(loader)} iters/epoch, device: {device}")
    it = 0
    for epoch in range(1 + opt.epoch_offset, opt.n_epochs + 1 + opt.epoch_offset):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            mask = random_block_mask(x.size(0), opt.grid, opt.mask_ratio, opt.size, device)
            x_masked = x.masked_fill(mask, 0.0)

            with torch.amp.autocast("cuda"):
                rec = model(x_masked)
                diff = (rec - x).abs() * mask          # 只算被遮区域
                loss = diff.sum() / (mask.sum() * x.size(1) + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item(); nb += 1; it += 1
            if it % 200 == 0:
                print(f"epoch {epoch} iter {it}/{total_iters} loss {run_loss/nb:.4f}")

        msg = f"[epoch {epoch}] loss {run_loss/max(nb,1):.4f} time {time.time()-t0:.0f}s"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        # 每 epoch 存重建对比图（原图 / 遮挡 / 重建）
        model.eval()
        with torch.no_grad():
            x = next(iter(loader)).to(device)
            mask = random_block_mask(x.size(0), opt.grid, opt.mask_ratio, opt.size, device)
            rec = model(x.masked_fill(mask, 0.0)).clamp(-1, 1)
            grid_img = torch.cat([x[:4], x.masked_fill(mask, 0.0)[:4], rec[:4]])
            vutils.save_image(grid_img * 0.5 + 0.5, os.path.join(vis_dir, f"epoch{epoch:03d}.png"), nrow=4, normalize=False)
        # 存编码器权重（只存 features，格式与 torchvision VGG16 一致）
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.n_epochs:
            torch.save(model.encoder.features.state_dict(), os.path.join(save_dir, f"mae_encoder_epoch{epoch}.pth"))
            torch.save(model.encoder.features.state_dict(), os.path.join(save_dir, "mae_encoder_latest.pth"))
            torch.save(model.decoder.state_dict(), os.path.join(save_dir, "mae_decoder_latest.pth"))
            print(f"saved encoder weights at epoch {epoch}")


math_pi = 3.141592653589793

if __name__ == "__main__":
    main()
