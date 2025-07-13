import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Freeze the original layer (will be set by parent)
        self.original_layer = None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional bias
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.lora_bias = None

        # Dropout layer
        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
        if self.lora_bias is not None:
            nn.init.zeros_(self.lora_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        # Get original output
        if self.original_layer is not None:
            original_out = self.original_layer(x)
        else:
            original_out = 0

        # Compute LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_out = lora_out * (self.alpha / self.rank)

        # Add bias if present
        if self.lora_bias is not None:
            lora_out = lora_out + self.lora_bias

        return original_out + lora_out


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer

        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=original_layer.bias is not None,
        )
        self.lora.original_layer = self.original_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)

    def merge_weights(self):
        """Merge LoRA weights into original layer (for inference)"""
        if self.lora.rank > 0:
            delta_weight = (self.lora.lora_B @ self.lora.lora_A) * (
                self.lora.alpha / self.lora.rank
            )
            self.original_layer.weight.data += delta_weight
            if self.lora.lora_bias is not None and self.original_layer.bias is not None:
                self.original_layer.bias.data += self.lora.lora_bias

    def unmerge_weights(self):
        """Unmerge LoRA weights from original layer"""
        if self.lora.rank > 0:
            delta_weight = (self.lora.lora_B @ self.lora.lora_A) * (
                self.lora.alpha / self.lora.rank
            )
            self.original_layer.weight.data -= delta_weight
            if self.lora.lora_bias is not None and self.original_layer.bias is not None:
                self.original_layer.bias.data -= self.lora.lora_bias


class LoRAConv2d(nn.Module):
    """
    Conv2d layer with LoRA adaptation
    """

    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer

        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Create LoRA matrices for conv
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices for convolution
        self.lora_A = nn.Parameter(
            torch.zeros(
                rank,
                original_layer.in_channels,
                original_layer.kernel_size[0],
                original_layer.kernel_size[1],
            )
        )
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_channels, rank, 1, 1))

        # Dropout
        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convolution
        original_out = self.original_layer(x)

        # LoRA adaptation
        x_dropped = self.lora_dropout(x)
        lora_out = F.conv2d(
            x_dropped,
            self.lora_A,
            stride=self.original_layer.stride,
            padding=self.original_layer.padding,
            dilation=self.original_layer.dilation,
            groups=self.original_layer.groups,
        )
        lora_out = F.conv2d(lora_out, self.lora_B)
        lora_out = lora_out * (self.alpha / self.rank)

        return original_out + lora_out


def apply_lora_to_linear(
    module: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to specified linear layers in a module

    Args:
        module: The module to apply LoRA to
        target_modules: List of module names to apply LoRA to (e.g., ['qkv', 'proj', 'fc1', 'fc2'])
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate

    Returns:
        Dictionary mapping module names to LoRA layers
    """
    lora_layers = {}

    def apply_lora_recursive(parent_module, parent_name=""):
        for name, child_module in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Check if this module should have LoRA applied
            if any(target in name for target in target_modules) and isinstance(
                child_module, nn.Linear
            ):
                # Replace with LoRA version
                lora_layer = LoRALinear(
                    child_module, rank=rank, alpha=alpha, dropout=dropout
                )
                setattr(parent_module, name, lora_layer)
                lora_layers[full_name] = lora_layer
                print(
                    f"Applied LoRA to {full_name}: {child_module.in_features} -> {child_module.out_features}"
                )
            else:
                # Recurse into child modules
                apply_lora_recursive(child_module, full_name)

    apply_lora_recursive(module)
    return lora_layers


def apply_lora_to_conv(
    module: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> Dict[str, LoRAConv2d]:
    """
    Apply LoRA to specified conv2d layers in a module
    """
    lora_layers = {}

    def apply_lora_recursive(parent_module, parent_name=""):
        for name, child_module in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Check if this module should have LoRA applied
            if any(target in name for target in target_modules) and isinstance(
                child_module, nn.Conv2d
            ):
                # Replace with LoRA version
                lora_layer = LoRAConv2d(
                    child_module, rank=rank, alpha=alpha, dropout=dropout
                )
                setattr(parent_module, name, lora_layer)
                lora_layers[full_name] = lora_layer
                print(
                    f"Applied LoRA to {full_name}: {child_module.in_channels} -> {child_module.out_channels}"
                )
            else:
                # Recurse into child modules
                apply_lora_recursive(child_module, full_name)

    apply_lora_recursive(module)
    return lora_layers


def get_lora_parameters(module: nn.Module) -> List[torch.nn.Parameter]:
    """Get all LoRA parameters from a module"""
    lora_params = []
    for name, param in module.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params


def save_lora_state_dict(module: nn.Module, path: str):
    """Save only LoRA parameters"""
    lora_state_dict = {}
    for name, param in module.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.detach().cpu()
    torch.save(lora_state_dict, path)


def load_lora_state_dict(module: nn.Module, path: str):
    """Load LoRA parameters"""
    lora_state_dict = torch.load(path, map_location="cpu")

    # Filter to only LoRA parameters
    current_state = module.state_dict()
    filtered_state = {}
    for name, param in lora_state_dict.items():
        if name in current_state and "lora_" in name:
            filtered_state[name] = param

    module.load_state_dict(filtered_state, strict=False)
