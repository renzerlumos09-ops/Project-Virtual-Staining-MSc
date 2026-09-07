import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'
        self.dir_paired = os.path.join(opt.dataroot, 'train') #成对数据的路径 

        self.paired_ratio = getattr(opt, 'paired_ratio', 0.0)  # 成对数据占总数据的比例，可以根据需要调整
        self.total_samples = getattr(opt, 'total_samples', 150000)  # 总的训练样本数量，可以根据需要调整
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        if self.paired_ratio > 0:
            self.paired_paths = sorted(make_dataset(self.dir_paired, opt.max_dataset_size)) # load paired images from '/path/to/data/train'
        else:
            self.paired_paths = []
        self.paired_size = len(self.paired_paths)  # 成对数据的数量
        
        self.paired_indices = list(range(self.paired_size))  # 成对数据的索引列表
        self.paired_ptr = 0  # 成对数据的指针，初始为0，独立于A和B数据的索引

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.paired_size = len(self.paired_paths)

        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        self.data_modes =[] # 数据模式列表，包含'paired'和'unpaired'两种模式
        # 根据成对数据的比例计算成对数据和非成对数据的数量
        num_paired = int(self.total_samples * self.paired_ratio)
        num_unpaired = self.total_samples - num_paired
        self.data_modes = ["paired"] * num_paired + ["unpaired"] * num_unpaired

        self.shuffle() # 打乱数据模式列表，使成对数据和非成对数据在训练过程中随机出现

    def shuffle(self):
        random.shuffle(self.data_modes) #每次调用shuffle方法都会打乱数据模式列表，使成对数据和非成对数据在训练过程中随机出现
        random.shuffle(self.paired_indices) #打乱成对数据的索引列表，使成对数据在训练过程中以随机顺序出现
        self.paired_ptr = 0 #重置成对数据的指针，使成对数据从头开始使用

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        mode = self.data_modes[index % len(self.data_modes)]

        if mode == "paired":
            paired_idx = self.paired_indices[self.paired_ptr]
            paired_path = self.paired_paths[paired_idx] 

            self.paired_ptr += 1 # 移动成对数据的指针
            if self.paired_ptr >= self.paired_size: # 如果指针超过成对数据的数量，重置并重新打乱索引
                random.shuffle(self.paired_indices)
                self.paired_ptr = 0

            # 读取成对图像
            try:
                img = Image.open(paired_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"读取成对数据失败: {paired_path}, 错误信息: {str(e)}") from e

            w, h = img.size
            real_A = img.crop((0, 0, w // 2, h))
            real_B = img.crop((w // 2, 0, w, h))  # 右半部分作为B
            real_A = self.transform_A(real_A)
            real_B = self.transform_B(real_B)
            return {"A": real_A, "B": real_B, 
                    "A_paths": paired_path, "B_paths": paired_path, 
                    "is_paired": True
            }
        
        else:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            try:
                A_img = Image.open(A_path).convert("RGB")
                B_img = Image.open(B_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"读取非成对数据失败: {A_path} 或 {B_path}, 错误信息: {str(e)}") from e
            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

        return {"A": A, "B": B, 
                "A_paths": A_path, "B_paths": B_path, 
                "is_paired": False
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.total_samples
