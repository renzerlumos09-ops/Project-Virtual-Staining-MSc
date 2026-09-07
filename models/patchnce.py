import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """PatchNCE 损失（CUT 论文简化移植版，单尺度，采样负样本）。

    query: 生成 IHC 的特征 (B,C,H,W)
    key:   参考 AF 的特征   (B,C,H,W)
    正样本 = 同空间位置的 key；负样本 = 同一 AF 图内随机采样的其他位置。
    索引在 CPU 上生成再搬运，规避部分 Windows/CUDA 组合下 device randperm
    高级索引的不稳定问题；损失计算固定在 float32。
    """

    def __init__(self, tau=0.07, num_patches=256, num_negatives=255):
        super().__init__()
        self.tau = tau
        self.num_patches = num_patches
        self.num_negatives = num_negatives
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        # 防御：若 query/key 空间尺寸不一致（奇数尺寸输入经 VGG 下采样取整所致），
        # 强制对齐，否则按位置索引会越界
        if feat_k.shape[2:] != feat_q.shape[2:]:
            feat_k = F.interpolate(feat_k, size=feat_q.shape[2:], mode="bilinear", align_corners=False)

        B, C, H, W = feat_q.shape
        hw = H * W

        # 索引在 CPU 生成，再搬到 GPU（Windows 稳定性考虑）
        n = min(self.num_patches, hw)
        idx = torch.randperm(hw)[:n].to(feat_q.device)
        m = min(self.num_negatives, hw - 1)
        neg_idx = torch.randperm(hw)[:m].to(feat_q.device)

        q = F.normalize(feat_q.float().flatten(2), dim=1)  # B,C,HW
        k = F.normalize(feat_k.float().flatten(2), dim=1)  # B,C,HW

        q_s = q.index_select(2, idx).transpose(1, 2)       # B,n,C
        pos = k.index_select(2, idx).transpose(1, 2)       # B,n,C
        neg = k.index_select(2, neg_idx)                   # B,C,m

        l_pos = (q_s * pos).sum(-1, keepdim=True)          # B,n,1
        l_neg = torch.matmul(q_s, neg)                     # B,n,m

        logits = torch.cat([l_pos, l_neg], dim=-1) / self.tau  # B,n,1+m
        labels = torch.zeros(B * n, dtype=torch.long, device=q.device)
        return self.ce(logits.reshape(B * n, -1), labels)


def make_mlp(cin, hidden=256):
    """NCE 特征投影头（1x1 卷积实现，Q/K 两侧共享同一个头）。"""
    return nn.Sequential(
        nn.Conv2d(cin, hidden, 1), nn.ReLU(True),
        nn.Conv2d(hidden, hidden, 1),
    )
