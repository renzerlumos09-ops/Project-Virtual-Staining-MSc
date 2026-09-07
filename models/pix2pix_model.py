import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="instance", netG="resnet_9blocks", dataset_mode="aligned", no_dropout=False)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument('--lr_D', type=float, default=0.0002, help='learning rate for discriminator')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的损失名称
        self.loss_names = ['G_GAN', 'G_L1', 'G_Pyramid', 'G_VGG', 'D_real', 'D_fake']
        # 指定要保存/显示的视觉结果
        self.visual_names = ['real_A', 'fake_B', 'real_B']
    
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 定义生成器和判别器
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                              not opt.no_dropout, opt.init_type, opt.init_gain)
    
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
        
            # --- 你的自定义损失 ---
            from .networks import PyramidLoss, VGGPerceptualLoss # 确保在networks.py里定义了
            self.criterionPyramid = PyramidLoss(levels=3).to(self.device)
            self.criterionVGG = VGGPerceptualLoss().to(self.device)
           # ----------------------

           # 定义优化器
            # --- 核心修正：修正优化器变量名 ---
            # 官方 Repo 使用的是 self.optimizer_G 而不是 self.optimizers.append
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        # Calculate GAN loss for the discriminator
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  
        # we use conditional GANs; we need to feed both input and output to the discriminator
        # 这是一个条件 GAN，所以我们需要将输入和输出一起送入判别器

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # 1. GAN Loss (对抗损失)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        raw_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_gan
        #对抗损失是用来衡量生成器生成的图像在判别器眼中有多“真实”，这个损失越小越好；
        #因为我们希望生成器能够欺骗判别器，让它认为假图像是真实的。

        # 2. 组合监督损失 (你的核心改进)
        # L1 Loss (基础像素对齐)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
    
         # Pyramid Loss (多尺度结构)
        self.loss_G_Pyramid = self.criterionPyramid(self.fake_B, self.real_B) * self.opt.lambda_pyramid
    
        # VGG Loss (对抗微小对齐误差)
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_vgg
        
        # 这里我们写一个监督模块，如果 GAN 损失过大，就强行降级到一个合理范围，防止训练不稳定
        # 因为pre-train的目的是为了让生成器学会基本的结构和内容，而不是过度追求对抗损失；
        # 所以我们允许 GAN 损失在一个合理范围内，但如果它过大了，就强行降级到结构损失的一个比例。
        struct_loss_sum = self.loss_G_L1 + self.loss_G_Pyramid
        if raw_G_GAN > struct_loss_sum * 0.8:  # 如果 GAN 损失过大，强行降级
            ratio = (struct_loss_sum * 0.8) / (raw_G_GAN + 1e-7) # 加一个小数防止除零
            self.loss_G_GAN = raw_G_GAN * ratio
        else:
            self.loss_G_GAN = raw_G_GAN

        # 汇总
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Pyramid + self.loss_G_VGG
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        if not hasattr(self, 'internal_count'):
            self.internal_count = 0
        self.internal_count += 1
        #这里是为了降低判别器的更新频率，因为在训练初期，生成器的输出可能非常差，过于频繁地更新判别器可能会导致训练不稳定。
        #通过降低判别器的更新频率，我们可以让生成器有更多的机会去学习基本的结构和内容，而不是过度追求对抗损失，从而提高训练的稳定性和效果。

        # update D
        if self.internal_count % 3 == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
