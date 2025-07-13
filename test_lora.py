#!/usr/bin/env python3

"""
Test script for HAMER LoRA implementation
This script verifies that LoRA is working correctly
"""

import torch
import numpy as np
from hamer.configs import get_config
from hamer.models.hamer_lora import create_lora_hamer
from hamer.models.components.lora import get_lora_parameters


def test_lora_creation():
    """Test LoRA model creation"""
    print("Testing LoRA model creation...")

    # Get config
    cfg = get_config()

    # Create LoRA config
    lora_config = {
        "rank": 4,
        "alpha": 1.0,
        "dropout": 0.0,
        "target_modules": {
            "backbone": ["qkv", "proj", "fc1", "fc2"],
            "mano_head": ["to_qkv", "to_out", "decpose", "decshape", "deccam"],
            "discriminator": ["betas_fc1", "betas_fc2"],
        },
    }

    # Create model
    model = create_lora_hamer(cfg, lora_config=lora_config, init_renderer=False)

    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA ratio: {lora_params/total_params*100:.2f}%")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

    # Verify only LoRA parameters are trainable
    assert trainable_params == lora_params, "Only LoRA parameters should be trainable"
    assert lora_params > 0, "Should have some LoRA parameters"
    assert lora_params < total_params * 0.1, "LoRA parameters should be < 10% of total"

    print("✓ LoRA model creation test passed!")
    return model


def test_forward_pass(model):
    """Test forward pass with LoRA model"""
    print("\nTesting forward pass...")

    # Create dummy batch
    batch_size = 2
    batch = {
        "img": torch.randn(batch_size, 3, 256, 192),
        "keypoints_2d": torch.randn(batch_size, 21, 2),
        "keypoints_3d": torch.randn(batch_size, 21, 3),
        "mano_params": {
            "global_orient": torch.randn(batch_size, 1, 3, 3),
            "hand_pose": torch.randn(batch_size, 15, 3, 3),
            "betas": torch.randn(batch_size, 10),
        },
        "has_mano_params": {
            "global_orient": torch.ones(batch_size, dtype=torch.bool),
            "hand_pose": torch.ones(batch_size, dtype=torch.bool),
            "betas": torch.ones(batch_size, dtype=torch.bool),
        },
        "mano_params_is_axis_angle": {
            "global_orient": torch.zeros(batch_size, dtype=torch.bool),
            "hand_pose": torch.zeros(batch_size, dtype=torch.bool),
            "betas": torch.zeros(batch_size, dtype=torch.bool),
        },
    }

    # Run forward pass
    model.eval()
    with torch.no_grad():
        output = model.forward_step(batch, train=False)

    # Check outputs
    assert "pred_vertices" in output, "Should output vertices"
    assert "pred_keypoints_3d" in output, "Should output 3D keypoints"
    assert "pred_keypoints_2d" in output, "Should output 2D keypoints"

    # Check shapes
    assert output["pred_vertices"].shape == (
        batch_size,
        778,
        3,
    ), f"Wrong vertices shape: {output['pred_vertices'].shape}"
    assert output["pred_keypoints_3d"].shape == (
        batch_size,
        21,
        3,
    ), f"Wrong 3D keypoints shape: {output['pred_keypoints_3d'].shape}"
    assert output["pred_keypoints_2d"].shape == (
        batch_size,
        21,
        2,
    ), f"Wrong 2D keypoints shape: {output['pred_keypoints_2d'].shape}"

    print("✓ Forward pass test passed!")
    return output


def test_loss_computation(model, batch, output):
    """Test loss computation"""
    print("\nTesting loss computation...")

    # Compute loss
    loss = model.compute_loss(batch, output, train=True)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.numel() == 1, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"

    print(f"Loss value: {loss.item():.4f}")
    print("✓ Loss computation test passed!")


def test_gradient_flow(model, batch):
    """Test that gradients flow only through LoRA parameters"""
    print("\nTesting gradient flow...")

    model.train()
    output = model.forward_step(batch, train=True)
    loss = model.compute_loss(batch, output, train=True)

    # Backward pass
    loss.backward()

    # Check gradients
    lora_params_with_grad = 0
    non_lora_params_with_grad = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            if "lora_" in name:
                lora_params_with_grad += 1
            else:
                non_lora_params_with_grad += 1

    print(f"LoRA parameters with gradients: {lora_params_with_grad}")
    print(f"Non-LoRA parameters with gradients: {non_lora_params_with_grad}")

    assert lora_params_with_grad > 0, "LoRA parameters should have gradients"
    assert (
        non_lora_params_with_grad == 0
    ), "Non-LoRA parameters should not have gradients"

    print("✓ Gradient flow test passed!")


def test_save_load_lora():
    """Test LoRA weight saving and loading"""
    print("\nTesting LoRA save/load...")

    # Create two identical models
    cfg = get_config()
    lora_config = {
        "rank": 4,
        "alpha": 1.0,
        "dropout": 0.0,
        "target_modules": {
            "backbone": ["qkv", "proj"],
            "mano_head": ["decpose", "decshape"],
        },
    }

    model1 = create_lora_hamer(cfg, lora_config=lora_config, init_renderer=False)
    model2 = create_lora_hamer(cfg, lora_config=lora_config, init_renderer=False)

    # Modify LoRA parameters in model1
    for param in get_lora_parameters(model1):
        param.data.fill_(1.0)

    # Save LoRA weights
    temp_path = "/tmp/test_lora_weights.pth"
    model1.save_lora_weights(temp_path)

    # Load into model2
    model2.load_lora_weights(temp_path)

    # Compare LoRA parameters
    params1 = {
        name: param.data.clone()
        for name, param in model1.named_parameters()
        if "lora_" in name
    }
    params2 = {
        name: param.data.clone()
        for name, param in model2.named_parameters()
        if "lora_" in name
    }

    for name in params1:
        assert torch.allclose(
            params1[name], params2[name]
        ), f"LoRA parameter {name} not loaded correctly"

    # Clean up
    import os

    os.remove(temp_path)

    print("✓ LoRA save/load test passed!")


def main():
    """Run all tests"""
    print("=" * 50)
    print("HAMER LoRA Test Suite")
    print("=" * 50)

    try:
        # Test 1: Model creation
        model = test_lora_creation()

        # Test 2: Forward pass
        batch = {
            "img": torch.randn(2, 3, 256, 192),
            "keypoints_2d": torch.randn(2, 21, 2),
            "keypoints_3d": torch.randn(2, 21, 3),
            "mano_params": {
                "global_orient": torch.randn(2, 1, 3, 3),
                "hand_pose": torch.randn(2, 15, 3, 3),
                "betas": torch.randn(2, 10),
            },
            "has_mano_params": {
                "global_orient": torch.ones(2, dtype=torch.bool),
                "hand_pose": torch.ones(2, dtype=torch.bool),
                "betas": torch.ones(2, dtype=torch.bool),
            },
            "mano_params_is_axis_angle": {
                "global_orient": torch.zeros(2, dtype=torch.bool),
                "hand_pose": torch.zeros(2, dtype=torch.bool),
                "betas": torch.zeros(2, dtype=torch.bool),
            },
        }
        output = test_forward_pass(model)

        # Test 3: Loss computation
        test_loss_computation(model, batch, output)

        # Test 4: Gradient flow
        test_gradient_flow(model, batch)

        # Test 5: Save/load
        test_save_load_lora()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("LoRA implementation is working correctly.")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
