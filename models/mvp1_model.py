import torch
import torch.nn.functional as F

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .mvp1_modules import FrozenVGG, ProjModule, FeatureDecoder
from .patchnce import PatchNCELoss, make_mlp


class MVP1Model(BaseModel):
    """MVP-1: 冻结 VGG 编码器 + Proj 残差筛选 + 单向 FastCUT 生成。

    数据流:
        real_A (AF) --FrozenVGG--> S (relu4_3), R_sig (relu2_2)
        S' = S + Proj(R_sig)
        fake_B = G(S'.detach())          # 生成器只收特征，不见原图
        D: fake_B vs real_B (IHC)

    损失分组（物理分开回传）:
        G 组:    L_GAN + lambda_nce * L_NCE          （不回传 Proj）
        Proj 组: lambda_distill*L_distill + lambda_pyramid*L_pyramid(S',S)+ lambda_sparse*L_sparse(increment)
        D 组:    标准 PatchGAN 对抗损失

    数据集模式: --dataset_mode unaligned（is_paired 在 MVP-1 中忽略，只用非配对数据）
    """

    NCE_CHANNELS = [128, 256, 512]  # relu2_2 / relu3_3 / relu4_3

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument("--lambda_nce", type=float, default=1.0, help="PatchNCE 损失权重")
            parser.add_argument("--lambda_distill", type=float, default=10.0, help="Proj 表征蒸馏损失权重")
            parser.add_argument("--lambda_proj_pyramid", type=float, default=5.0, help="Proj 少尺度金字塔损失权重")
            parser.add_argument("--lambda_sparse", type=float, default=1.0, help="空间稀疏加权损失权重")
            parser.add_argument("--nce_temp", type=float, default=0.07, help="PatchNCE 温度系数")
            parser.add_argument("--num_patches", type=int, default=256, help="PatchNCE 采样 patch 数")
            parser.add_argument("--w_tissue", type=float, default=0.1, help="稀疏损失中腺体组织区的权重（背景=1.0）")
            parser.add_argument("--white_thresh", type=float, default=0.9, help="白板率统计阈值（[0,1] 尺度）")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_GAN", "G_NCE", "Proj_distill", "Proj_pyramid", "Proj_sparse", "D", "white_ratio"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if self.isTrain:
            self.model_names = ["G", "Proj", "F", "D"]   # F = NCE 投影头
        else:
            self.model_names = ["G", "Proj"]

        # --- 冻结编码器（不进 model_names，不保存不加载、不优化） ---
        self.netEnc = FrozenVGG().to(self.device)

        # --- 可训练模块 ---
        self.netProj = ProjModule(in_ch=128, out_ch=512).to(self.device)
        self.netG = FeatureDecoder(in_ch=512, ngf=opt.ngf, out_ch=opt.output_nc).to(self.device)
        networks.init_weights(self.netProj, opt.init_type, opt.init_gain)
        networks.init_weights(self.netG, opt.init_type, opt.init_gain)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netF = torch.nn.ModuleList([make_mlp(c).to(self.device) for c in self.NCE_CHANNELS])

            self.fake_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(tau=opt.nce_temp, num_patches=opt.num_patches).to(self.device)

            # 三套优化器物理分开（红线：不要合并成一个总 loss）
            self.optimizer_G = torch.optim.Adam(
                list(self.netG.parameters()) + list(self.netF.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netProj.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_P, self.optimizer_D]

    # ---------------- 数据 ----------------
    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)  # AF
        self.real_B = input["B" if AtoB else "A"].to(self.device)  # IHC
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    # ---------------- 前向 ----------------
    def forward(self):
        with torch.no_grad():  # 冻结编码器提 AF 特征（NCE 参考也用这份）
            feats_A = self.netEnc(self.real_A)
        self.S = feats_A[2]          # 骨架 (B,512,H/16,W/16)
        self.R_sig = feats_A[0]      # 残差候选 (B,128,H/4,W/4)

        self.increment = self.netProj(self.R_sig)
        self.S_prime = self.S + self.increment

        # 红线：detach 截断，GAN/NCE 梯度不回 Proj
        self.fake_B = self.netG(self.S_prime.detach())

        # 生成图特征（梯度要穿回 G，不能用 no_grad；编码器本身冻结即可）
        self.feats_fake = self.netEnc(self.fake_B)

    # ---------------- Proj 组损失 ----------------
    def backward_Proj(self):
        # 1) 蒸馏：S' 不偏离骨架 S
        self.loss_Proj_distill = F.mse_loss(self.S_prime, self.S) * self.opt.lambda_distill

        # 2) 少尺度金字塔：约束 S' 的多尺度结构（2 个尺度即可）
        p = F.l1_loss(self.S_prime, self.S)
        s_ds = F.avg_pool2d(self.S, 2)
        sp_ds = F.avg_pool2d(self.S_prime, 2)
        p = p + F.l1_loss(sp_ds, s_ds)
        self.loss_Proj_pyramid = p / 2.0 * self.opt.lambda_proj_pyramid

        # 3) 空间稀疏加权：背景区把残差压到 0，组织区放宽
        with torch.no_grad():
            gray = (self.real_A + 1) / 2
            gray = gray.mean(dim=1, keepdim=True)
            tissue = (gray < self.opt.white_thresh).float()          # 组织=1
            w = tissue * self.opt.w_tissue + (1 - tissue) * 1.0      # 背景重罚
            w = F.interpolate(w, size=self.increment.shape[2:], mode="nearest")
        self.loss_Proj_sparse = (w * self.increment.abs()).mean() * self.opt.lambda_sparse

        loss = self.loss_Proj_distill + self.loss_Proj_pyramid + self.loss_Proj_sparse
        loss.backward()

    # ---------------- G 组损失 ----------------
    def backward_G(self):
        # 对抗
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True) * self.opt.lambda_gan

        # PatchNCE：fake_B 与 real_A 在三层特征上对齐
        with torch.no_grad():
            feats_A = self.netEnc(self.real_A)
        nce = 0.0
        for i, mlp in enumerate(self.netF):
            q = mlp(self.feats_fake[i])
            k = mlp(feats_A[i])
            nce = nce + self.criterionNCE(q, k)
        self.loss_G_NCE = nce / len(self.netF) * self.opt.lambda_nce

        (self.loss_G_GAN + self.loss_G_NCE).backward()

    # ---------------- D 组损失 ----------------
    def backward_D(self):
        fake = self.fake_pool.query(self.fake_B)
        pred_real = self.netD(self.real_B)
        pred_fake = self.netD(fake.detach())
        self.loss_D = (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False)) * 0.5
        self.loss_D.backward()

    # ---------------- 白板率监控（混入 loss_names 打印，不是损失） ----------------
    def compute_white_ratio(self):
        with torch.no_grad():
            img = (self.fake_B + 1) / 2
            self.loss_white_ratio = (img > self.opt.white_thresh).all(dim=1).float().mean()

    # ---------------- 优化入口 ----------------
    def optimize_parameters(self):
        self.forward()
        self.compute_white_ratio()

        # G + F（冻结 D）
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Proj（独立回传）
        self.optimizer_P.zero_grad()
        self.backward_Proj()
        self.optimizer_P.step()

        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
