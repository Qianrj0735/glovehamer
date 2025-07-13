import os, glob
import sys

# os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
import numpy as np

# from manoannotator.mano_renderer import MANORenderer
from skvideo.io import vwrite
import utils
import json
from tqdm import tqdm


class PthDataset(Dataset):
    """
    A simple PyTorch Dataset skeleton. Fill in data loading logic as needed.

    Args:
        data_dir (str): Path to the directory containing data.
        transform (Callable, optional): Optional transform to be applied on a sample.
        target_transform (Callable, optional): Optional transform to be applied on the target.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # TODO: Populate a list of data items (e.g., file paths) in self.samples
        # Example: self.samples = sorted(os.listdir(data_dir))
        samples = glob.glob(f"{data_dir}/*.pth") * 100
        self.samples = np.array(samples)
        self.previewing = False

    def __len__(self) -> int:
        """
        Return the total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = torch.load(self.samples[idx], weights_only=False)
        # os.makedirs("preview", exist_ok=True)
        # # fn = f"preview/{idx}.png"
        # # if hasattr(self, previewing)
        # if self.previewing:
        #     rendered_image = self.preview_sample(sample)
        #     sample["rendered_image"] = rendered_image
        return sample


def calc_stats(path):
    dataset = Image2ManoDataset(data_dir=path)

    # Print dataset size
    print(f"Number of samples in dataset: {len(dataset)}")
    vids = []
    # Iterate through samples
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    manos = []
    for idx, (sample) in tqdm(enumerate(dataloader)):
        manos.append(utils.reassembly_mano(sample))
        # if idx == 5:
        #     break
    mano = torch.cat(manos, dim=0).flatten(1)
    min = mano.min(0)
    max = mano.max(0)
    mean = mano.mean(0)
    std = mano.std(0)
    with open(f"{path}/stats.json", "w") as f:
        json.dump(
            {
                "min": min.values.tolist(),
                "max": max.values.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
        )


if __name__ == "__main__":
    dataset = Image2ManoDataset(
        data_dir="/nfs1/factorworld_dataset/glovedvids_pth_dataset_train"
    )
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for sample in tqdm(dl):
        print(sample)
