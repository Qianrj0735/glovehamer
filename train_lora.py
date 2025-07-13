from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import signal
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from yacs.config import CfgNode
from hamer.configs import dataset_config
from hamer.datasets import HAMERDataModule, PthDataModule
from hamer.models.hamer_lora import HAMERLoRA, create_lora_hamer
from hamer.utils.pylogger import get_pylogger
from hamer.utils.misc import task_wrapper, log_hyperparameters

# Reset signal handling for lightning
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(
    model_cfg: CfgNode, dataset_cfg: CfgNode, lora_cfg: DictConfig, rootdir: str
):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, "model_config.yaml"))
    OmegaConf.save(config=lora_cfg, f=os.path.join(rootdir, "lora_config.yaml"))
    with open(os.path.join(rootdir, "dataset_config.yaml"), "w") as f:
        f.write(dataset_cfg.dump())


def load_pretrained_hamer(cfg: DictConfig, model_path: str) -> HAMERLoRA:
    """Load a pre-trained HAMER model and apply LoRA"""
    import torch

    log.info(f"Loading pre-trained HAMER model from {model_path}")

    # Create LoRA config from hydra config
    lora_config = {
        "rank": cfg.lora.rank,
        "alpha": cfg.lora.alpha,
        "dropout": cfg.lora.dropout,
        "target_modules": dict(cfg.lora.target_modules),
    }

    # Create model with LoRA
    model = create_lora_hamer(cfg, lora_config=lora_config, init_renderer=True)

    # Load pre-trained weights (this will load the base model weights)
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    # Load state dict, but only for non-LoRA parameters
    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict, strict=False
    )

    # Filter out LoRA-related missing keys (these are expected)
    actual_missing = [k for k in missing_keys if "lora_" not in k]
    if actual_missing:
        log.warning(f"Missing keys (non-LoRA): {actual_missing}")

    if unexpected_keys:
        log.warning(f"Unexpected keys: {unexpected_keys}")

    log.info(
        f"Loaded pre-trained model with {len(missing_keys)} total missing keys (LoRA layers)"
    )

    return model


@task_wrapper
def train_lora(cfg: DictConfig) -> Tuple[dict, dict]:
    """Train HAMER with LoRA"""

    # Load dataset config
    dataset_cfg = dataset_config(name="datasets_lora.yaml")

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.lora, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = PthDataModule(cfg, dataset_cfg)

    # Create model with LoRA
    if cfg.lora.get("pretrained_model_path") is not None:
        # Load from pre-trained checkpoint
        model = load_pretrained_hamer(cfg, cfg.lora.pretrained_model_path)
    else:
        # Create new model with LoRA
        lora_config = {
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": dict(cfg.lora.target_modules),
        }
        model = create_lora_hamer(cfg, lora_config=lora_config, init_renderer=True)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(
        os.path.join(cfg.paths.output_dir, "tensorboard"),
        name="",
        version="",
        default_hp_metric=False,
    )
    loggers = [logger]

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        filename="lora_hamer_{epoch:03d}_{step:06d}",
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor=None,  # Save based on steps, not metric
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Rich progress bar (optional)
    # rich_callback = pl.callbacks.RichProgressBar()
    # callbacks.append(rich_callback)

    # Custom callback to save LoRA weights separately
    class LoRASaveCallback(pl.callbacks.Callback):
        def __init__(self, save_interval: int, save_dir: str):
            self.save_interval = save_interval
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if (trainer.global_step + 1) % self.save_interval == 0:
                lora_path = os.path.join(
                    self.save_dir, f"lora_weights_step_{trainer.global_step:06d}.pth"
                )
                pl_module.save_lora_weights(lora_path)
                log.info(f"Saved LoRA weights at step {trainer.global_step}")

    if cfg.lora.get("save_lora_separately", True):
        lora_save_callback = LoRASaveCallback(
            save_interval=cfg.lora.get("save_interval", 1000),
            save_dir=os.path.join(cfg.paths.output_dir, "lora_weights"),
        )
        callbacks.append(lora_save_callback)

    # Setup trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        plugins=(
            SLURMEnvironment(requeue_signal=signal.SIGUSR2)
            if (cfg.get("launcher", None) is not None)
            else None
        ),
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Print LoRA information
    model.print_lora_info()

    # Verify only LoRA parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params/total_params*100:.2f}%)"
    )

    # Train the model
    log.info("Starting LoRA training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    log.info("LoRA training completed!")

    # Save final LoRA weights
    final_lora_path = os.path.join(cfg.paths.output_dir, "final_lora_weights.pth")
    model.save_lora_weights(final_lora_path)
    log.info(f"Saved final LoRA weights to {final_lora_path}")

    return {}, {}


@hydra.main(
    version_base="1.2",
    config_path=str(root / "hamer/configs_hydra"),
    config_name="train_lora.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function"""
    # Import torch here to avoid issues with multiprocessing
    import torch

    # Set random seed if specified
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Train the model
    train_lora(cfg)


if __name__ == "__main__":
    main()
