# check_feature_gap.py — 判决性检验：Proj 蒸馏任务是"trivial"还是"有信息量"？
# 原理：A1 的蒸馏目标是随机不配对的 S_B。若 Proj 完全不动（S' = S），
# distill 的期望值 = 随机 AF/IHC 图对的 relu4_3 特征 MSE。
# 与 a1_smoke 收敛值（distill/λ = 0.0021）对比即可判定。
#
# 用法：复制到项目根目录 D:\Inde_Project\pytorch-CycleGAN-and-pix2pix\
#       python check_feature_gap.py

import glob
import random

import numpy as np
import torch
from PIL import Image

from models.mvp2_modules import FrozenVGG

N_PAIRS = 30          # 随机图对数，够稳即可
SIZE = 512


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

feats_A, feats_B = [], []
with torch.no_grad():
    for p in A_paths[:N_PAIRS]:
        feats_A.append(enc(load(p).cuda())[2].flatten(1).cpu())   # relu4_3
    for p in B_paths[:N_PAIRS]:
        feats_B.append(enc(load(p).cuda())[2].flatten(1).cpu())

SA = torch.cat(feats_A)   # (N, 512*64*64)
SB = torch.cat(feats_B)

mse_pairwise = ((SA - SB) ** 2).mean().item()              # Proj 不动时的蒸馏基线
mse_meanshift = ((SA.mean(0) - SB.mean(0)) ** 2).mean().item()  # "平均域差"项
var_A = ((SA - SA.mean(0)) ** 2).mean().item()             # AF 域内方差
var_B = ((SB - SB.mean(0)) ** 2).mean().item()             # IHC 域内方差

print(f"随机配对 MSE(S_A, S_B)        = {mse_pairwise:.6f}   <- Proj 蒸馏任务的起点（不干活时的值）")
print(f"域均值差 MSE(E[S_A], E[S_B])  = {mse_meanshift:.6f}   <- Proj 仅靠'平均域差'能压到的地板")
print(f"AF  域内方差                   = {var_A:.6f}")
print(f"IHC 域内方差                   = {var_B:.6f}")
print()
print("判读：")
print("  a1_smoke 收敛值 distill/lambda = 0.0021")
print("  - 若 随机配对 MSE ≈ 0.002     -> 任务起点即终点，Proj 没活干（trivial）")
print("  - 若 随机配对 MSE >> 0.002    -> Proj 压缩了真实差距，学到了映射")
print("  - 若 域内方差也 ≈ 0.001 量级  -> 高层特征整体坍缩，relu4_3 不适合做蒸馏/NCE，需换中层")
