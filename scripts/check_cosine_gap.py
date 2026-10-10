# check_cosine_gap.py — 对比式蒸馏可行性检验：AF/IHC 在高层特征空间里"方向"可分吗？
# 原理：对比蒸馏（normalize 后）不看距离看方向。只要域均值方向有可分夹角，
#       InfoNCE 就有活干；若方向几乎重合（cos≈1），任何蒸馏变体在 relu4_3 都无效。
#
# 用法：复制到项目根目录 D:\Inde_Project\pytorch-CycleGAN-and-pix2pix\
#       python check_cosine_gap.py

import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.mvp2_modules import FrozenVGG

N = 30
SIZE = 512


def load(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (x * 2 - 1).unsqueeze(0)


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()


enc = FrozenVGG(weights_path="./checkpoints/mae_dual_v1/mae_encoder_latest.pth").cuda().eval()

A_paths = sorted(glob.glob("./datasets/prostate_data/trainA/*.png"))
B_paths = sorted(glob.glob("./datasets/prostate_data/trainB/*.png"))
random.seed(0)
random.shuffle(A_paths)
random.shuffle(B_paths)

with torch.no_grad():
    SA = torch.cat([enc(load(p).cuda())[2].cpu() for p in A_paths[:N]])   # (N,512,64,64)
    SB = torch.cat([enc(load(p).cuda())[2].cpu() for p in B_paths[:N]])

meanA = SA.mean(0, keepdim=True)
meanB = SB.mean(0, keepdim=True)

cos_domains = cos(meanA, meanB)                 # 域均值方向相似度（关键指标）
cos_pairwise = cos(SA, SB)                      # 随机 AF-IHC 图对
half = N // 2
cos_within_A = cos(SA[:half], SA[half:])        # AF 域内
cos_within_B = cos(SB[:half], SB[half:])        # IHC 域内

print(f"域均值方向 cos(E[S_A], E[S_B]) = {cos_domains:.4f}   <- 关键指标")
print(f"随机配对   cos(S_A, S_B)       = {cos_pairwise:.4f}")
print(f"AF  域内   cos(S_A1, S_A2)     = {cos_within_A:.4f}")
print(f"IHC 域内   cos(S_B1, S_B2)     = {cos_within_B:.4f}")
print()
print("判读：")
print("  - 域均值 cos 明显低于域内 cos（如 0.90 vs 0.98）-> 方向可分，对比式蒸馏有活干")
print("  - 域均值 cos ≈ 域内 cos（都 0.97+）             -> 方向也不可分，relu4_3 蒸馏无解，")
print("                                                     MVP-2 按'局限已查明'收尾，转 MVP-3")
