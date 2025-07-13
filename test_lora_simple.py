#!/usr/bin/env python3

"""
Simple test for LoRA components without requiring full HAMER setup
"""

import sys
import os

# Add the current directory to Python path to import our modules
sys.path.insert(0, "/home/workspace/glovehamer")

try:
    import torch

    print("PyTorch found!")
except ImportError:
    print("PyTorch not found. Installing...")
    os.system("pip install torch torchvision")
    import torch

import torch.nn as nn
from hamer.models.components.lora import LoRALayer, LoRALinear, LoRAConv2d


def test_lora_layer():
    """Test basic LoRA layer"""
    print("Testing LoRA layer...")

    # Create LoRA layer
    lora = LoRALayer(in_features=64, out_features=32, rank=4, alpha=1.0)

    # Test forward pass
    x = torch.randn(2, 64)
    y = lora(x)

    assert y.shape == (2, 32), f"Wrong output shape: {y.shape}"
    print("✓ LoRA layer test passed!")


def test_lora_linear():
    """Test LoRA linear wrapper"""
    print("Testing LoRA linear...")

    # Create original linear layer
    original = nn.Linear(64, 32)

    # Wrap with LoRA
    lora_linear = LoRALinear(original, rank=4, alpha=1.0)

    # Test forward pass
    x = torch.randn(2, 64)
    y = lora_linear(x)

    assert y.shape == (2, 32), f"Wrong output shape: {y.shape}"

    # Check that original parameters are frozen
    for param in original.parameters():
        assert not param.requires_grad, "Original parameters should be frozen"

    # Check that LoRA parameters are trainable
    lora_param_count = sum(
        p.numel() for p in lora_linear.parameters() if p.requires_grad
    )
    assert lora_param_count > 0, "Should have trainable LoRA parameters"

    print(f"✓ LoRA linear test passed! Trainable params: {lora_param_count}")


def test_lora_conv2d():
    """Test LoRA conv2d wrapper"""
    print("Testing LoRA conv2d...")

    # Create original conv layer
    original = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    # Wrap with LoRA
    lora_conv = LoRAConv2d(original, rank=4, alpha=1.0)

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = lora_conv(x)

    assert y.shape == (2, 16, 32, 32), f"Wrong output shape: {y.shape}"

    # Check that original parameters are frozen
    for param in original.parameters():
        assert not param.requires_grad, "Original parameters should be frozen"

    # Check that LoRA parameters are trainable
    lora_param_count = sum(p.numel() for p in lora_conv.parameters() if p.requires_grad)
    assert lora_param_count > 0, "Should have trainable LoRA parameters"

    print(f"✓ LoRA conv2d test passed! Trainable params: {lora_param_count}")


def test_parameter_efficiency():
    """Test parameter efficiency of LoRA"""
    print("Testing parameter efficiency...")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 64)
            self.linear2 = nn.Linear(64, 32)
            self.conv = nn.Conv2d(3, 16, kernel_size=3)

    model = SimpleModel()
    original_params = sum(p.numel() for p in model.parameters())

    # Apply LoRA
    model.linear1 = LoRALinear(model.linear1, rank=4, alpha=1.0)
    model.linear2 = LoRALinear(model.linear2, rank=4, alpha=1.0)
    model.conv = LoRAConv2d(model.conv, rank=4, alpha=1.0)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    efficiency = trainable_params / total_params * 100

    print(f"Original parameters: {original_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter efficiency: {efficiency:.2f}%")

    assert efficiency < 20, f"LoRA should be parameter efficient, got {efficiency:.2f}%"
    print("✓ Parameter efficiency test passed!")


def test_gradient_flow():
    """Test that gradients only flow through LoRA parameters"""
    print("Testing gradient flow...")

    # Create model with LoRA
    original = nn.Linear(10, 5)
    lora_model = LoRALinear(original, rank=2, alpha=1.0)

    # Forward pass
    x = torch.randn(3, 10)
    y = lora_model(x)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    original_has_grad = any(p.grad is not None for p in original.parameters())
    lora_has_grad = any(p.grad is not None for p in lora_model.lora.parameters())

    assert not original_has_grad, "Original parameters should not have gradients"
    assert lora_has_grad, "LoRA parameters should have gradients"

    print("✓ Gradient flow test passed!")


def main():
    """Run all tests"""
    print("=" * 40)
    print("LoRA Components Test Suite")
    print("=" * 40)

    try:
        test_lora_layer()
        test_lora_linear()
        test_lora_conv2d()
        test_parameter_efficiency()
        test_gradient_flow()

        print("\n" + "=" * 40)
        print("All LoRA tests passed! ✓")
        print("LoRA implementation is working correctly.")
        print("=" * 40)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
