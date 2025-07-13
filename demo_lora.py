#!/usr/bin/env python3

"""
Inference script for HAMER with LoRA weights
Usage:
    python demo_lora.py --lora_weights path/to/lora_weights.pth --img_folder example_data --out_folder demo_out_lora
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Import HAMER components
from hamer.configs import get_config
from hamer.models import create_lora_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD

import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description="HaMeR LoRA Demo")

    # LoRA specific arguments
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights file (.pth)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Path to base HAMER model (if different from default)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4, help="LoRA rank used during training"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=1.0, help="LoRA alpha used during training"
    )

    # Standard demo arguments
    parser.add_argument(
        "--img_folder",
        type=str,
        default="example_data",
        help="Folder with input images",
    )
    parser.add_argument(
        "--out_folder", type=str, default="demo_out_lora", help="Output folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument("--side_view", action="store_true", help="Render side view")
    parser.add_argument(
        "--save_mesh", action="store_true", help="Save meshes as .obj files"
    )
    parser.add_argument("--full_frame", action="store_true", help="Render full frame")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )

    return parser.parse_args()


def load_lora_model(args):
    """Load HAMER model with LoRA weights"""

    # Get default config
    model_cfg = get_config()

    # Create LoRA config based on arguments
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": 0.0,  # No dropout during inference
        "target_modules": {
            "backbone": ["qkv", "proj", "fc1", "fc2"],
            "mano_head": [
                "to_qkv",
                "to_out",
                "to_kv",
                "to_q",
                "decpose",
                "decshape",
                "deccam",
            ],
            "discriminator": [
                "D_conv1",
                "D_conv2",
                "betas_fc1",
                "betas_fc2",
                "D_alljoints_fc1",
                "D_alljoints_fc2",
            ],
        },
    }

    print("Creating HAMER model with LoRA...")
    model = create_lora_hamer(model_cfg, lora_config=lora_config, init_renderer=True)

    # Load base model weights if specified
    if args.base_model is not None:
        print(f"Loading base model weights from {args.base_model}")
        base_checkpoint = torch.load(args.base_model, map_location="cpu")
        if "state_dict" in base_checkpoint:
            base_state_dict = base_checkpoint["state_dict"]
        else:
            base_state_dict = base_checkpoint
        model.load_state_dict(base_state_dict, strict=False)

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights}")
    model.load_lora_weights(args.lora_weights)

    # Merge LoRA weights for faster inference
    print("Merging LoRA weights into base model...")
    model.merge_lora_weights()

    # Set to eval mode
    model.eval()
    model.to(args.device)

    print(f"Model loaded successfully on {args.device}")
    return model


def process_images(model, img_folder, out_folder, args):
    """Process images with the LoRA model"""

    # Create output directory
    os.makedirs(out_folder, exist_ok=True)

    # Get image files
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(Path(img_folder).glob(f"*{ext}"))
        img_files.extend(Path(img_folder).glob(f"*{ext.upper()}"))

    if not img_files:
        print(f"No images found in {img_folder}")
        return

    print(f"Found {len(img_files)} images")

    # Process images
    dataset = ViTDetDataset(model.cfg, img_folder, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing images")):
            batch = recursive_to(batch, args.device)

            # Run inference
            out = model(batch)

            # Process outputs
            batch_size = batch["img"].shape[0]
            for j in range(batch_size):
                img_idx = i * args.batch_size + j
                if img_idx >= len(img_files):
                    break

                img_file = img_files[img_idx]
                img_name = img_file.stem

                # Extract single item from batch
                single_out = {}
                for key, value in out.items():
                    if isinstance(value, torch.Tensor):
                        single_out[key] = value[j : j + 1]
                    elif isinstance(value, dict):
                        single_out[key] = {k: v[j : j + 1] for k, v in value.items()}
                    else:
                        single_out[key] = value

                single_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        single_batch[key] = value[j : j + 1]
                    else:
                        single_batch[key] = value

                # Render and save results
                save_results(
                    model, single_batch, single_out, img_name, out_folder, args
                )


def save_results(model, batch, out, img_name, out_folder, args):
    """Save inference results"""

    # Get predictions
    pred_vertices = out["pred_vertices"].cpu().numpy()[0]  # (778, 3)
    pred_cam_t = out["pred_cam_t"].cpu().numpy()[0]  # (3,)
    pred_keypoints_2d = out["pred_keypoints_2d"].cpu().numpy()[0]  # (21, 2)

    # Get image
    img = batch["img"].cpu().numpy()[0]  # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    img = (img * np.array(DEFAULT_STD) + np.array(DEFAULT_MEAN)) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Save mesh if requested
    if args.save_mesh:
        mesh_path = os.path.join(out_folder, f"{img_name}_mesh.obj")
        save_mesh_obj(pred_vertices, model.mano.faces, mesh_path)

    # Render visualization
    focal_length = out["focal_length"].cpu().numpy()[0]  # (2,)

    # Render mesh overlay
    if model.mesh_renderer is not None:
        try:
            rendered_img = model.mesh_renderer.visualize_tensorboard(
                pred_vertices[None],  # Add batch dim
                pred_cam_t[None],
                img[None],
                pred_keypoints_2d[None],
                None,  # No GT keypoints
                focal_length=focal_length[None],
            )

            # Save rendered image
            rendered_path = os.path.join(out_folder, f"{img_name}_rendered.jpg")
            cv2.imwrite(rendered_path, rendered_img[:, :, ::-1])  # RGB to BGR

        except Exception as e:
            print(f"Warning: Could not render image {img_name}: {e}")

    # Save original image with keypoints
    img_with_kpts = img.copy()
    for kpt in pred_keypoints_2d:
        x, y = int(kpt[0] * img.shape[1]), int(kpt[1] * img.shape[0])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_with_kpts, (x, y), 3, (0, 255, 0), -1)

    kpts_path = os.path.join(out_folder, f"{img_name}_keypoints.jpg")
    cv2.imwrite(kpts_path, img_with_kpts[:, :, ::-1])  # RGB to BGR


def save_mesh_obj(vertices, faces, path):
    """Save mesh as OBJ file"""
    with open(path, "w") as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    args = parse_args()

    print("=" * 50)
    print("HAMER LoRA Inference Demo")
    print("=" * 50)
    print(f"LoRA weights: {args.lora_weights}")
    print(f"Input folder: {args.img_folder}")
    print(f"Output folder: {args.out_folder}")
    print(f"Device: {args.device}")
    print("=" * 50)

    # Check if LoRA weights exist
    if not os.path.exists(args.lora_weights):
        print(f"Error: LoRA weights file not found: {args.lora_weights}")
        return

    # Check if input folder exists
    if not os.path.exists(args.img_folder):
        print(f"Error: Input folder not found: {args.img_folder}")
        return

    # Load model
    try:
        model = load_lora_model(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process images
    try:
        process_images(model, args.img_folder, args.out_folder, args)
        print(f"\nInference completed! Results saved to {args.out_folder}")
    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == "__main__":
    main()
