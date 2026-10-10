# check_layer_gap.py — 三层特征判别性普查：找出蒸馏/NCE 该用哪一层
# 指标：信噪比 SNR = 域均值差 / (AF域内方差 + IHC域内方差)
#       SNR > 1  说明该层能区分 AF 与 IHC（有判别性，可做蒸馏目标）
#       SNR < 1  说明该层域间差异淹没在域内噪声里（坍缩，无信息）
#
# 用法：复制到项目根目录 D:\Inde_Project\pytorch-CycleGAN-and-pix2pix\
#       python check_layer_gap.py

import glob
import random

import numpy as np
import torch
from PIL import Image

from models.mvp2_modules import FrozenVGG

N_PAIRS = 30
SIZE = 512
LAYER_NAMES = ["relu2_2 (128ch, 1/2)", "relu3_3 (256ch, 1/4)", "relu4_3 (512ch, 1/8)"]


def load(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (x * 2 - 1).unsqueeze(0)          # [-1, 1]，与训练输入一致


enc = FrozenVGG(weights_path="./checkpoints/mae_dual_v1/mae_encoder_latest.pth").cuda().eval()

A_paths = sorted(glob.glob("./datasets/prostate_data/trainA/*.png"))
B_paths = sorted(glob.glob("./datasets/prostate_data/trainB/*.png"))
random.seed(0)
random.shuffle(A_paths)
random.shuffle(B_paths)

# 每层累加：元素级 sum / sumsq（算域均值差与域内方差），以及随机配对 MSE
stats = [dict(sumA=None, sumsqA=None, sumB=None, sumsqB=None, mse=0.0) for _ in range(3)]

with torch.no_grad():
    for pa, pb in zip(A_paths[:N_PAIRS], B_paths[:N_PAIRS]):
        fa = enc(load(pa).cuda())
        fb = enc(load(pb).cuda())
        for li in range(3):
            a = fa[li].flatten(1).double().cpu()   # (1, D)
            b = fb[li].flatten(1).double().cpu()
            st = stats[li]
            if st["sumA"] is None:
                st["sumA"] = torch.zeros_like(a); st["sumsqA"] = torch.zeros_like(a)
                st["sumB"] = torch.zeros_like(b); st["sumsqB"] = torch.zeros_like(b)
            st["sumA"] += a; st["sumsqA"] += a * a
            st["sumB"] += b; st["sumsqB"] += b * b
            st["mse"] += ((a - b) ** 2).mean().item()

print(f"{'layer':<24} {'域均值差':>12} {'AF域内方差':>12} {'IHC域内方差':>12} {'随机配对MSE':>12} {'SNR':>8}")
for li in range(3):
    st = stats[li]
    meanA = st["sumA"] / N_PAIRS
    meanB = st["sumB"] / N_PAIRS
    varA = ((st["sumsqA"] / N_PAIRS) - meanA ** 2).mean().item()
    varB = ((st["sumsqB"] / N_PAIRS) - meanB ** 2).mean().item()
    gap = ((meanA - meanB) ** 2).mean().item()
    mse = st["mse"] / N_PAIRS
    snr = gap / (varA + varB + 1e-12)
    print(f"{LAYER_NAMES[li]:<24} {gap:>12.6f} {varA:>12.6f} {varB:>12.6f} {mse:>12.6f} {snr:>8.2f}")

print()
print("判读：")
print("  SNR > 1   -> 该层能区分 AF/IHC，蒸馏目标挪到该层（改 netEnc(real_B)[i] 的索引）")
print("  SNR < 1   -> 该层坍缩，无判别信息")
print("  三层全 <1 -> MIM 预训练特征整体无域判别性，需重新讨论预训练目标")
