from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.distributed as dist
import math

from .transforms import transforms_imagenet_train, transforms_imagenet_eval, ImageFolderWithEntropy
from .distributed_utils import is_dist_avail_and_initialized, get_world_size, get_rank

@dataclass
class AugmentConfig:
    color_jitter: float = 0.0
    auto_augment: str = 'rand-m9-mstd0.5-inc1'  # RandAugment with magnitude 9
    interpolation: transforms.InterpolationMode = 'bicubic'
    re_prob: float = 0.25  # Random erasing probability
    re_mode: str = 'const'  # Random erasing fill mode ('pixel', 'const', etc.)
    re_count: int = 1  # Number of random erasing regions
    eval_crop_ratio: float = 0.875  # Crop ratio for evaluation
    repeated_aug: bool = False  # Whether to use repeated augmentation

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ImagenetDataModuleWithEntropy(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
    """

    def __init__(
        self,
        data_dir: str = "path/to/imagenet",
        train_val_test_split: Tuple[int, int, int] = (1281167, 50000, 50000),
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        augment: Any = None,
        patch_size: int = 16,
        num_scales: int = 2,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.image_size = img_size
        self.save_hyperparameters(logger=False)
        self.augment = augment

        # data transformations - now returns (pre_transform, post_transform)
        self.train_pre_transform, self.train_post_transform = transforms_imagenet_train(
            img_size=self.image_size,
            mean=mean,
            std=std,
            pre_post_divide=True,
            augment=augment
        )

        self.val_pre_transform, self.val_post_transform = transforms_imagenet_eval(
            img_size=self.image_size,
            mean=mean,
            std=std,
            pre_post_divide=True,
            eval_crop_ratio=augment.eval_crop_ratio if augment is not None else None,
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ImageNet classes (1000).
        """
        return 1000

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass  # ImageNet needs to be downloaded manually

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning before training/validation/test steps.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print("Loading training data from", f"{self.hparams.data_dir}/train")
            self.data_train = ImageFolderWithEntropy(
                root=f"{self.hparams.data_dir}/train",
                transform=(self.train_pre_transform, self.train_post_transform),
                patch_size=self.hparams.patch_size,
                num_scales=self.hparams.num_scales,
            )
            print("Loading validation data from", f"{self.hparams.data_dir}/val")
            self.data_val = ImageFolderWithEntropy(
                root=f"{self.hparams.data_dir}/val",
                transform=(self.val_pre_transform, self.val_post_transform),
                patch_size=self.hparams.patch_size,
                num_scales=self.hparams.num_scales
            )
            #TODO: identify test set separately...?
            self.data_test = self.data_val
    
    def train_dataloader(self):
        total_examples = len(self.data_train)
        batch_size = self.hparams.batch_size
        # Calculate the largest number of examples that is a multiple of batch_size
        num_examples = total_examples - (total_examples % batch_size)
        subset = torch.utils.data.Subset(self.data_train, range(num_examples))
        
        if hasattr(self, "augment") and self.augment is not None and self.augment.repeated_aug:
            sampler_train = RASampler(subset)
        else:
            sampler_train = torch.utils.data.DistributedSampler(subset, shuffle=True) if self.trainer and self.trainer.world_size > 1 else None
        
        return DataLoader(
            dataset=subset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler_train,
        )

    def val_dataloader(self):
        total_examples = len(self.data_val)
        batch_size = self.hparams.batch_size
        # Calculate the largest number of examples that is a multiple of batch_size
        num_examples = total_examples - (total_examples % batch_size)
        subset = torch.utils.data.Subset(self.data_val, range(num_examples))
        return DataLoader(
            dataset=subset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            sampler=torch.utils.data.DistributedSampler(subset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None,
        )

    def test_dataloader(self):
        subset = torch.utils.data.Subset(self.data_test, range(50000))
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            sampler=torch.utils.data.DistributedSampler(subset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None,
        )    

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
    
class ImagenetDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
    """

    def __init__(
        self,
        data_dir: str = "path/to/imagenet",
        train_val_test_split: Tuple[int, int, int] = (1281167, 50000, 50000),
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.image_size = img_size
        self.save_hyperparameters(logger=False)

        # data transformations - now returns (pre_transform, post_transform)
        self.train_transform = transforms_imagenet_train(
            img_size=self.image_size,
            mean=mean,
            std=std,
        )

        self.val_transform = transforms_imagenet_eval(
            img_size=self.image_size,
            mean=mean,
            std=std,
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ImageNet classes (1000).
        """
        return 1000

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass  # ImageNet needs to be downloaded manually

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning before training/validation/test steps.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print("Loading training data from", f"{self.hparams.data_dir}/train")
            self.data_train = ImageFolder(
                root=f"{self.hparams.data_dir}/train",
                transform=self.train_transform,
            )
            print("Loading validation data from", f"{self.hparams.data_dir}/val")
            self.data_val = ImageFolder(
                root=f"{self.hparams.data_dir}/val",
                transform=self.val_transform,
            )
            #TODO: identify test set separately...?
            self.data_test = self.data_val

    # def train_dataloader(self):
    #     total_examples = len(self.data_train)
    #     batch_size = self.hparams.batch_size
    #     # Calculate the largest number of examples that is a multiple of batch_size
    #     num_examples = total_examples - (total_examples % batch_size)
    #     subset = torch.utils.data.Subset(self.data_train, range(num_examples))
    #     return DataLoader(
    #         dataset=subset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=True,
    #         sampler=torch.utils.data.DistributedSampler(subset, shuffle=True) if self.trainer and self.trainer.world_size > 1 else None,
    #     )

    # def val_dataloader(self):
    #     total_examples = len(self.data_val)
    #     batch_size = self.hparams.batch_size
    #     # Calculate the largest number of examples that is a multiple of batch_size
    #     num_examples = total_examples - (total_examples % batch_size)
    #     subset = torch.utils.data.Subset(self.data_val, range(num_examples))
    #     return DataLoader(
    #         dataset=subset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #         sampler=torch.utils.data.DistributedSampler(subset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None,
    #     )

    # def test_dataloader(self):
    #     subset = torch.utils.data.Subset(self.data_test, range(50000))
    #     return DataLoader(
    #         dataset=subset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #         sampler=torch.utils.data.DistributedSampler(subset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None,
    #     )
    
    def train_dataloader(self):
        num_tasks = get_world_size()
        global_rank = get_rank()
        
        if num_tasks > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(self.data_train)
            
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        num_tasks = get_world_size()
        global_rank = get_rank()
        
        if num_tasks > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False
            )
        else:
            val_sampler = torch.utils.data.SequentialSampler(self.data_val)
            
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=val_sampler,
        )

    def test_dataloader(self):
        num_tasks = get_world_size()
        global_rank = get_rank()
        
        if num_tasks > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_test,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False
            )
        else:
            test_sampler = torch.utils.data.SequentialSampler(self.data_test)
            
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=test_sampler,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass