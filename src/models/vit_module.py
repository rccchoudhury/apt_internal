import os
import timm
import warnings
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from typing import Any, Dict, Tuple
import ipdb
import torch
import wandb
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from fvcore.nn import FlopCountAnalysis
from torchmetrics.classification.accuracy import Accuracy

from src.models.optim_utils import *
from src.models.patch_tokenizer import PatchTokenizer
#from thop import profile
import copy
import math

class ViTLitModule(LightningModule):
    """LightningModule for Vision Transformer (ViT) ImageNet classification.
patch
    A `LightningModule` that handles training, validation, and testing for ViT on ImageNet.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer: PatchTokenizer,
        scheduler_cfg: CosineSchedulerConfig = None,
        compile: bool = False,
        checkpoint_path: str = '',
        match_head_shape: bool = True,  # Add new parameter to control head shape matching
        train_mix_only: bool = False,
        mixup: Any = None,
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialize a `ViTLitModule`.

        Args:
            net: The ViT model to train
            optimizer: The optimizer to use for training
            scheduler: The learning rate scheduler to use for training
            compile: Whether to compile the model using torch.compile()
            checkpoint_path: Path to checkpoint to load
            match_head_shape: Whether to enforce matching head shape when loading checkpoint
            train_mix_only: Whether to only train on mixed patches
            mixup: Mixup configuration
            label_smoothing: Label smoothing factor (0.0 means no smoothing)
        """
        super().__init__()

        # Save hyperparameters for checkpointing
        # Wnoder whether we wnat to save the opt 
        self.save_hyperparameters(logger=False, ignore=['net', 'tokenizer'])

        # Init the network and the tokenizer.
        self.net = net
        self.tokenizer = tokenizer

        self.checkpoint_path = checkpoint_path
        self.match_head_shape = match_head_shape
        self.train_mix_only = train_mix_only
        if self.checkpoint_path is not None:
            #ipdb.set_trace()
            self.load_checkpoint(self.checkpoint_path)
        # Test: Find a more robust way to do this ! 
        if hasattr(self.net, 'mixed_patch') and self.net.mixed_patch is not None:
            print("\nInitializing multiscale patch embed!!\n")
            self.net.init_multiscale_patch_embed()
            
        #print("\nManually setting input size to use 32x32 patches!\n")
        #self.net.set_input_size(patch_size=(32, 32))
        if compile:
            print("Compiling....")
            self.net = torch.compile(self.net)
            print("Done compiling.")
        # Loss function - CrossEntropyLoss for classification
        if mixup is not None:
            self.criterion = SoftTargetCrossEntropy()
        elif label_smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics for ImageNet (1000 classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=1000)
        self.val_acc = Accuracy(task="multiclass", num_classes=1000)
        self.test_acc = Accuracy(task="multiclass", num_classes=1000)

        # Top-5 accuracy metrics
        self.train_acc_top5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.val_acc_top5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.test_acc_top5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

        # Loss tracking metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Best validation accuracy tracker
        self.val_acc_best = MaxMetric()

        # Set specific modules to eval mode
        #self._set_eval_except_mixed_patch()
        
        self.mixup_fn = None
        if mixup is not None:
            print("Initializing Mixup!!")
            mixup_active = mixup.mixup > 0 or mixup.cutmix > 0. or mixup.cutmix_minmax is not None
            
            if mixup_active:
                self.mixup_fn = Mixup(
                    mixup_alpha=mixup.mixup, 
                    cutmix_alpha=mixup.cutmix, cutmix_minmax=mixup.cutmix_minmax,
                    prob=mixup.mixup_prob, switch_prob=mixup.mixup_switch_prob, mode=mixup.mixup_mode,
                    label_smoothing=mixup.smoothing, num_classes=mixup.nb_classes)

    def _set_eval_except_mixed_patch(self):
        """Set blocks, patch_embed, and head to eval mode while keeping mixed_patch in train mode."""
        # Set specific modules to eval mode
        if hasattr(self.net, 'blocks'):
            self.net.blocks.eval()
        if hasattr(self.net, 'patch_embed'):
            self.net.patch_embed.eval()
        if hasattr(self.net, 'head'):
            self.net.head.eval()
            
        # Keep mixed_patch in train mode if it exists
        if hasattr(self.net, 'mixed_patch'):
            self.net.mixed_patch.train()

    # List of allowed key prefixes that can be missing when loading checkpoint
    ALLOWED_MISSING_KEYS = ['mixed_patch', 'patch_embed32', 'grp_token', 'head']

    def resample_pos_embed(self, state_dict):
        """Resample position embedding if the checkpoint and model sizes don't match.
        
        Args:
            state_dict: The state dictionary containing the position embedding
            
        Returns:
            state_dict: The updated state dictionary with resampled position embedding if needed
        """
        # Check if pos_embed exists in both state_dict and model
        if 'pos_embed' in state_dict and hasattr(self.net, 'pos_embed') and self.net.pos_embed is not None:
            # Get the position embedding from the checkpoint
            pos_embed_checkpoint = state_dict['pos_embed']
            
            # Get the position embedding from the model
            pos_embed_model = self.net.pos_embed
            
            # Check if the shapes don't match
            if pos_embed_checkpoint.shape[1] != pos_embed_model.shape[1]:
                print(f"Position embedding size mismatch: checkpoint {pos_embed_checkpoint.shape} vs model {pos_embed_model.shape}")
                
                # Get grid sizes
                if hasattr(self.net, 'patch_embed') and hasattr(self.net.patch_embed, 'grid_size'):
                    new_size = self.net.patch_embed.grid_size
                    
                    # Calculate old size based on pos_embed shape
                    num_prefix_tokens = 0 if getattr(self.net, 'no_embed_class', False) else getattr(self.net, 'num_prefix_tokens', 1)
                    old_num_patches = pos_embed_checkpoint.shape[1] - num_prefix_tokens
                    old_hw = int(math.sqrt(old_num_patches))
                    old_size = (old_hw, old_hw)
                    
                    print(f"Resampling position embedding from {old_size} to {new_size}")
                    
                    # Import the resample function
                    from timm.layers import resample_abs_pos_embed
                    
                    # Resample the position embedding
                    state_dict['pos_embed'] = resample_abs_pos_embed(
                        pos_embed_checkpoint,
                        new_size=new_size,
                        old_size=old_size,
                        num_prefix_tokens=num_prefix_tokens,
                    )
                    
                    print(f"Position embedding resampled to {state_dict['pos_embed'].shape}")
        
        return state_dict

    def resample_patch_embed(self, state_dict):
        """Resample patch embedding if the checkpoint and model patch sizes don't match.
        
        Args:
            state_dict: The state dictionary containing the patch embedding
            
        Returns:
            state_dict: The updated state dictionary with resampled patch embedding if needed
        """
        # Check if patch embedding projection weight exists in both state_dict and model
        if 'patch_embed.proj.weight' in state_dict and hasattr(self.net, 'patch_embed') and hasattr(self.net.patch_embed, 'proj'):
            # Get the patch embedding from the checkpoint
            patch_embed_checkpoint = state_dict['patch_embed.proj.weight']
            
            # Get the patch embedding from the model
            patch_embed_model = self.net.patch_embed.proj.weight
            
            # Check if the kernel sizes don't match
            if patch_embed_checkpoint.shape[2:] != patch_embed_model.shape[2:]:
                print(f"Patch embedding kernel size mismatch: checkpoint {patch_embed_checkpoint.shape[2:]} vs model {patch_embed_model.shape[2:]}")
                
                # Get patch sizes
                new_patch_size = self.net.patch_embed.patch_size
                
                print(f"Resampling patch embedding from {patch_embed_checkpoint.shape[2:]} to {new_patch_size}")
                
                # Import the resample function
                from timm.layers import resample_patch_embed
                
                # Resample the patch embedding
                state_dict['patch_embed.proj.weight'] = resample_patch_embed(
                    patch_embed_checkpoint,
                    new_patch_size,
                    verbose=True
                )
                
                print(f"Patch embedding resampled to {state_dict['patch_embed.proj.weight'].shape}")
        
        return state_dict

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from a given path.
        If the path exists, load from the file.
        If not, try to load from timm using the path as model name.
        
        # TODO: something smarter for handling jax/flax ckpts, for now this is fine.
        
        Args:
            path: The path to the checkpoint file or timm model name
        """
        
        if path == '':
            return

        def check_missing_keys(missing_keys, unexpected_keys):
            # Filter out allowed missing keys based on match_head_shape
            allowed_keys = self.ALLOWED_MISSING_KEYS if not self.match_head_shape else [k for k in self.ALLOWED_MISSING_KEYS if 'head' in k]
            real_missing = [k for k in missing_keys 
                          if not any(k.startswith(prefix) for prefix in allowed_keys)]
            if real_missing or unexpected_keys:
                raise RuntimeError(
                    f'Error(s) in loading state_dict:\n\t'
                    f'Missing key(s) that are not in allowed list {allowed_keys}: {real_missing}\n\t'
                    f'Unexpected key(s) in state_dict: {unexpected_keys}')

        if os.path.exists(path):
            # Load from local file
            state_dict = torch.load(path, weights_only=True)
            
            # Resample position embedding if needed
            state_dict = self.resample_pos_embed(state_dict)
            
            # Resample patch embedding if needed
            state_dict = self.resample_patch_embed(state_dict)
            
            # if not self.match_head_shape:
            #     # Remove head-related keys from state dict if we don't want to match head shape
            #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            # Strict fals since we have additional params.
            missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
            #check_missing_keys(missing_keys, unexpected_keys)
        else:
            # Try to load from timm3
            try:
                # Get image size from the network
                img_size = self.net.img_size if hasattr(self.net, 'img_size') else 224
                print(f"Loading model from timm with img_size: {img_size}")
                
                # Load pretrained model from timm
                timm_model = timm.create_model(path, pretrained=True, img_size=img_size)
                
                # Convert timm state dict to our format if needed
                state_dict = timm_model.state_dict()
                
                # Resample position embedding if needed
                state_dict = self.resample_pos_embed(state_dict)
                
                # Resample patch embedding if needed
                state_dict = self.resample_patch_embed(state_dict)
                
                if not self.match_head_shape:
                    # Remove head-related keys from state dict if we don't want to match head shape
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
                missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
                #check_missing_keys(missing_keys, unexpected_keys)
            except Exception as e:
                raise ValueError(f"Failed to load model from path {path} or timm: {e}")

    def forward(self, x, input_dict) -> Tuple[torch.Tensor, float]:
        """Perform a forward pass through the model.

        Args:
            x: A tensor of images of shape (batch_size, channels, height, width)
            input_dict: Dictionary containing tokenizer outputs

        Returns:
            A tuple containing:
                - logits: A tensor of logits of shape (batch_size, num_classes)
                - throughput: Images processed per second
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        logits = self.net(x, input_dict)
        end_event.record()
        
        #Count GLFOPS with fvcorr flopcountanlaysis
        
        # macs, params = profile(self.net, inputs=(x, input_dict,))
        # gflops = macs * 2 / 1e9
        # print(f"GFLOPS: {gflops}")

        # Wait for GPU sync
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        
        # Get batch size
        batch_size = x.size(0)
        
        # Calculate throughput (images/second)
        throughput = batch_size / elapsed_time
        
        return logits, throughput

    def on_train_start(self) -> None:
        """Lightning hook called when training begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_top5.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A tuple containing the input images and target labels

        Returns:
            A tuple containing (loss, predictions, targets, logits, retained_frac, throughput)
        """
        entropy_maps = None
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, entropy_maps = batch
            
        if self.mixup_fn:
            assert entropy_maps is None, "Have to compute map after mixing up!"
            x, y_mixed = self.mixup_fn(x, y)
        else:
            y_mixed = y

        # TODO: add back in the grad for the pos-embed, i think it might be 
        # a bit helpful.
        with torch.no_grad():
            # if entropy maps is None, compute in tokenizer [needed for mix up]
            input_dict = self.tokenizer(x, entropy_maps, self.net.pos_embed)
            retained_frac = input_dict["retained_frac"]
            assert "output_mask" in input_dict
    
        logits, throughput = self.forward(x, input_dict)
        loss = self.criterion(logits, y_mixed)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits, retained_frac, throughput

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for the model.

        Args:
            batch: The input batch containing (images, labels)
            batch_idx: The index of the current batch

        Returns:
            The training loss
        """
        loss, preds, targets, logits, retained_frac, throughput = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_acc_top5(logits, targets)

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_top5", self.train_acc_top5, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/retained_frac", retained_frac, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/throughput", throughput, on_step=True, on_epoch=True, prog_bar=True)
        # Get model outputs and metrics
        # if hasattr(self.net, 'mixed_patch') and hasattr(self.net.mixed_patch, 'merge_ratio'):
        #     merge_ratio = self.net.mixed_patch.merge_ratio_curr.mean().item()
        #     edge_thres = self.net.mixed_patch.edge_thres
            
        #     self.log("train/merge_ratio", merge_ratio, on_step=True, on_epoch=True)
        #     self.log("train/edge_thres", edge_thres, on_step=True, on_epoch=True)
            
        #     # Save last batch for epoch-end visualization
        #     self.last_batch = batch[0]
         # profile somtimes harms model; copy.deepcopy is needed
        # cannot use torchinfo.summary since our model's gflops differs by the entropy_map value, but they only consider the shape of input
        # model_copy = copy.deepcopy(self)
        # flops, _ = profile(model_copy, (img_slice, entropy_map), verbose=False)
        # self.log("train/gflops", flops / 1e9, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step for the model.

        Args:
            batch: The input batch containing (images, labels)
            batch_idx: The index of the current batch
        """
        loss, preds, targets, logits, retained_frac, throughput = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_acc_top5(logits, targets)

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top5", self.val_acc_top5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/retained_frac", retained_frac, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/throughput", throughput, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        """Lightning hook called when a training epoch starts."""
        if self.net.mixed_patch is not None:
            alpha = self.current_epoch / self.trainer.max_epochs
            self.net.mixed_patch.alpha = alpha

    def on_train_epoch_end(self) -> None:
        """Lightning hook called when a training epoch ends."""
        if hasattr(self, 'last_batch'):
            img = self.net.visualize_tokenizer(self.last_batch)
            self.logger.experiment.log({"visualize": wandb.Image(img)})

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called when a validation epoch ends."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """Test step for the model.

        Args:
            batch: The input batch containing (images, labels)
            batch_idx: The index of the current batch
        """
        loss, preds, targets, logits, retained_frac, throughput = self.model_step(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        #self.test_acc_top5(logits, targets)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/acc_top5", self.test_acc_top5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/retained_frac", retained_frac, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log performance metrics
        self.log("test/throughput", throughput, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/time_per_batch", batch[0].size(0) / throughput, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate FLOPs
        img = batch[0]
        img_slice = img[0][0].unsqueeze(0) if isinstance(img[0], (tuple, list)) else img[0].unsqueeze(0)
        if len(batch) == 2:
            entropy_map = None
        else:
            entropy_map = batch[2]
            for key in entropy_map.keys():
                entropy_map[key] = entropy_map[key][0].unsqueeze(0)
        
        # # Calculate parameters if first batch
        # if batch_idx == 0:
        #     # Calculate total parameters
        #     total_params = sum(p.numel() for p in self.parameters())
        #     self.log("test/total_params_M", total_params / 1e6, on_step=False, on_epoch=True, prog_bar=True)

    # Uncomment to find un-unsed params.
    # def on_after_backward(self):
    #     """Detect unused parameters after backward pass."""
    #     unused_params = [name for name, param in self.named_parameters() if param.grad is None]
    #     if unused_params:
    #         print(f"Unused parameters in DDP: {unused_params}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            A dict containing the configured optimizer and scheduler
        """
        # HARD CODED LAYER DECAY!
        LD = 0.99

        # TEST THE FREEZING
        if self.train_mix_only and self.net.mixed_patch is not None:
            for name, module in self.named_modules():
                if "mix" not in name:
                    module.eval()
                else:
                    module.train()

            for name, param in self.net.named_parameters():
                if "mix" not in name:
                    param.requires_grad = False

        num_layers = self.net.depth
        values = list(LD** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(values)

        opt_params = setup_layer_decay(self.net, 
                                   ld_assigner=assigner, 
                                   weight_decay=0.05)
        optimizer = self.hparams.optimizer(params=opt_params)
        
        config = self.hparams.scheduler_cfg
        num_training_steps = self.trainer.estimated_stepping_batches
        batch_size = self.trainer.train_dataloader.batch_size
        config.total_steps = num_training_steps
        config.batch_size = batch_size
        scheduler = cosine_scheduler(optimizer=optimizer, config=config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    _ = ViTLitModule(None, None, None, None)
