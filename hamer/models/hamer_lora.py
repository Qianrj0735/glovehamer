import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple, List, Optional

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import MANO
from .hamer import HAMER
from .components.lora import (
    apply_lora_to_linear,
    apply_lora_to_conv,
    get_lora_parameters,
    save_lora_state_dict,
    load_lora_state_dict,
)

log = get_pylogger(__name__)


class HAMERLoRA(HAMER):
    """
    HAMER model with LoRA (Low-Rank Adaptation) for efficient fine-tuning
    """

    def __init__(
        self,
        cfg: CfgNode,
        init_renderer: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        """
        Setup HAMER model with LoRA
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
            init_renderer (bool): Whether to initialize renderers
            lora_config (Dict): LoRA configuration containing:
                - rank: LoRA rank (default: 4)
                - alpha: LoRA alpha (default: 1.0)
                - dropout: LoRA dropout (default: 0.0)
                - target_modules: dict with keys 'backbone', 'mano_head', 'discriminator'
                  each containing list of module names to apply LoRA to
        """
        # Initialize parent HAMER model first
        super().__init__(cfg, init_renderer)

        # Set default LoRA config
        self.lora_config = lora_config or {
            "rank": 4,
            "alpha": 1.0,
            "dropout": 0.0,
            "target_modules": {
                "backbone": [
                    "qkv",
                    "proj",
                    "fc1",
                    "fc2",
                ],  # ViT attention and MLP layers
                "mano_head": [
                    "to_qkv",
                    "to_out",
                    "to_kv",
                    "to_q",
                    "decpose",
                    "decshape",
                    "deccam",
                ],  # Transformer layers
                "discriminator": [
                    "D_conv1",
                    "D_conv2",
                    "betas_fc1",
                    "betas_fc2",
                    "D_alljoints_fc1",
                    "D_alljoints_fc2",
                ],  # Discriminator layers
            },
        }

        # Store LoRA layers for tracking
        self.lora_layers = {}

        # Apply LoRA to different components
        self._apply_lora()

        # Log LoRA info
        total_params, lora_params = self._count_parameters()
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"LoRA parameters: {lora_params:,}")
        log.info(f"LoRA ratio: {lora_params/total_params*100:.2f}%")

    def _apply_lora(self):
        """Apply LoRA to specified modules"""
        rank = self.lora_config["rank"]
        alpha = self.lora_config["alpha"]
        dropout = self.lora_config["dropout"]
        target_modules = self.lora_config["target_modules"]

        # Apply LoRA to backbone (ViT)
        if "backbone" in target_modules and len(target_modules["backbone"]) > 0:
            log.info("Applying LoRA to backbone...")
            backbone_lora = apply_lora_to_linear(
                self.backbone,
                target_modules["backbone"],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.lora_layers["backbone"] = backbone_lora

            # Also apply to conv layers in backbone (patch embedding)
            backbone_conv_lora = apply_lora_to_conv(
                self.backbone,
                ["proj"],  # patch embedding projection
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.lora_layers["backbone_conv"] = backbone_conv_lora

        # Apply LoRA to MANO head (Transformer decoder)
        if "mano_head" in target_modules and len(target_modules["mano_head"]) > 0:
            log.info("Applying LoRA to MANO head...")
            mano_head_lora = apply_lora_to_linear(
                self.mano_head,
                target_modules["mano_head"],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.lora_layers["mano_head"] = mano_head_lora

        # Apply LoRA to discriminator
        if (
            hasattr(self, "discriminator")
            and "discriminator" in target_modules
            and len(target_modules["discriminator"]) > 0
        ):
            log.info("Applying LoRA to discriminator...")
            # Apply to linear layers
            disc_linear_lora = apply_lora_to_linear(
                self.discriminator,
                [
                    name
                    for name in target_modules["discriminator"]
                    if "conv" not in name
                ],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            # Apply to conv layers
            disc_conv_lora = apply_lora_to_conv(
                self.discriminator,
                [name for name in target_modules["discriminator"] if "conv" in name],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.lora_layers["discriminator"] = {**disc_linear_lora, **disc_conv_lora}

    def _count_parameters(self) -> Tuple[int, int]:
        """Count total and LoRA parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        lora_params = sum(p.numel() for p in get_lora_parameters(self))
        return total_params, lora_params

    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get all LoRA parameters for optimization"""
        return get_lora_parameters(self)

    def get_non_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get all non-LoRA parameters (should be frozen)"""
        non_lora_params = []
        for name, param in self.named_parameters():
            if "lora_" not in name:
                non_lora_params.append(param)
        return non_lora_params

    def freeze_non_lora_parameters(self):
        """Freeze all non-LoRA parameters"""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def unfreeze_non_lora_parameters(self):
        """Unfreeze all non-LoRA parameters"""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = True

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup optimizers for LoRA training - only optimize LoRA parameters
        """
        # Get only LoRA parameters for optimization
        lora_params = self.get_lora_parameters()

        if len(lora_params) == 0:
            log.warning("No LoRA parameters found! Using all parameters.")
            lora_params = list(self.parameters())

        # Ensure non-LoRA parameters are frozen
        self.freeze_non_lora_parameters()

        # Create optimizer for LoRA parameters only
        param_groups = [{"params": lora_params, "lr": self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(
            params=param_groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )

        # Discriminator optimizer (if discriminator exists and has LoRA)
        if hasattr(self, "discriminator"):
            disc_lora_params = get_lora_parameters(self.discriminator)
            if len(disc_lora_params) == 0:
                # If no LoRA in discriminator, use all discriminator params
                disc_lora_params = list(self.discriminator.parameters())

            optimizer_disc = torch.optim.AdamW(
                params=disc_lora_params,
                lr=self.cfg.TRAIN.LR,
                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            )
            return optimizer, optimizer_disc
        else:
            return optimizer

    def save_lora_weights(self, path: str):
        """Save only LoRA weights"""
        log.info(f"Saving LoRA weights to {path}")
        save_lora_state_dict(self, path)

    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        log.info(f"Loading LoRA weights from {path}")
        load_lora_state_dict(self, path)

    def merge_lora_weights(self):
        """Merge LoRA weights into base model for inference"""
        log.info("Merging LoRA weights into base model...")
        for module in self.modules():
            if hasattr(module, "merge_weights"):
                module.merge_weights()

    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base model"""
        log.info("Unmerging LoRA weights from base model...")
        for module in self.modules():
            if hasattr(module, "unmerge_weights"):
                module.unmerge_weights()

    def print_lora_info(self):
        """Print information about LoRA layers"""
        log.info("LoRA Configuration:")
        log.info(f"  Rank: {self.lora_config['rank']}")
        log.info(f"  Alpha: {self.lora_config['alpha']}")
        log.info(f"  Dropout: {self.lora_config['dropout']}")

        log.info("LoRA Layers:")
        for component, layers in self.lora_layers.items():
            log.info(f"  {component}: {len(layers)} layers")
            for layer_name in layers.keys():
                log.info(f"    {layer_name}")

        total_params, lora_params = self._count_parameters()
        log.info(f"Parameter Efficiency:")
        log.info(f"  Total parameters: {total_params:,}")
        log.info(f"  LoRA parameters: {lora_params:,}")
        log.info(f"  Trainable ratio: {lora_params/total_params*100:.2f}%")


def create_lora_hamer(
    cfg: CfgNode, lora_config: Optional[Dict] = None, init_renderer: bool = True
) -> HAMERLoRA:
    """
    Factory function to create HAMER model with LoRA

    Args:
        cfg: HAMER configuration
        lora_config: LoRA configuration
        init_renderer: Whether to initialize renderers

    Returns:
        HAMERLoRA model
    """
    model = HAMERLoRA(cfg, init_renderer=init_renderer, lora_config=lora_config)
    model.print_lora_info()
    return model
