#!/usr/bin/env python
import argparse
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DBM as DBM
from utils import binarize


def save_mnist_test_images(output_dir: str, binarize_images: bool):
    """
    Saves MNIST test images as PNGs in a given directory.

    Args:
        output_dir (str): Directory to save MNIST test images.
        num_images (int): Number of images to save.
    """
    if os.path.isdir(output_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST test dataset
    transform = [transforms.ToTensor()]
    if binarize_images:
        transform.append(torchvision.transforms.Lambda(binarize))
    dataset = datasets.MNIST(
        root="./dataset",
        train=False,
        download=True,
        transform=transforms.Compose(transform),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Save images
    for i, (img, _) in enumerate(dataloader):
        torchvision.utils.save_image(img, os.path.join(output_dir, f"{i}.png"))

    print(f"Saved {i} MNIST test images to {output_dir}")


def save_mnist_images(tensor, output_dir):
    """
    Save a binarized tensor (N x 28 x 28) as MNIST-style images.

    Parameters:
        tensor (numpy.ndarray): Input tensor of shape (N, 28, 28) with binary values (0 or 1).
        output_dir (str): Directory to save the images.
    """

    def save_image_worker(args):
        i, image_data, output_dir = args
        # Scale binary values (0 or 1) to grayscale range (0 to 255)
        image_data = (image_data * 255).astype(np.uint8)
        # Create a PIL image
        image = Image.fromarray(image_data, mode="L")
        # Save the image
        image.save(os.path.join(output_dir, f"image_{i:04d}.png"))

    if tensor.ndim != 3 or tensor.shape[1:] != (28, 28):
        raise ValueError("Input tensor must have shape (N, 28, 28).")

    with ThreadPoolExecutor(max_workers=32) as p:
        futures = [
            p.submit(save_image_worker, (i, tensor[i], output_dir))
            for i in range(len(tensor))
        ]
        # Wait for all threads to complete
        for future in futures:
            future.result()


def compute_fid_mnist(path1: str, path2: str, dev: int) -> float:
    """
    Computes the Fréchet Inception Distance (FID) between two directories of images
    using the pytorch-fid command-line tool.

    Args:
        path1 (str): Path to the first directory of images (e.g., real images).
        path2 (str): Path to the second directory of images (e.g., generated images).

    Returns:
        float: The computed FID score.
    """
    try:
        # Run the pytorch-fid CLI command
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytorch_fid",
                path1,
                path2,
                f"--device=cuda:{dev}",
                "--batch-size=64",
                "--num-workers=32",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract FID score from the output
        output_lines = result.stdout.strip().split("\n")
        # FID score is the last number
        fid_score = float(output_lines[-1].split()[-1])

        return fid_score

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error computing FID: {e.stderr}")


def worker(args) -> tuple[int, float]:
    model_wt_path, N, name, device, test_path = args
    full_desc = name.parent.name.split("-")
    bits = int(full_desc[1].split(":")[-1])
    L = int(full_desc[2].split(":")[-1])
    nh = int(full_desc[3].split(":")[-1])
    epoch = int(model_wt_path.stem.split("-")[1])
    s1 = time.perf_counter()
    model = DBM(28, 1, bits=bits, nh=nh, L=L)
    model = model.to(f"cuda:{device}")
    model_wt = torch.load(model_wt_path, weights_only=True)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in model_wt.items()},
        strict=True,
    )
    s2 = time.perf_counter()
    v_mode, _ = model.sample(N)
    s3 = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="/tmp/imgs_",
        suffix=f"_E{epoch}",
        # delete=False, # Uncomment if you want to see generated images
    ) as tmpdirname:
        s4 = time.perf_counter()
        save_mnist_images(v_mode.squeeze(1).numpy(force=True), tmpdirname)
        s5 = time.perf_counter()
        fid = compute_fid_mnist(test_path, tmpdirname, device)
        s6 = time.perf_counter()
        print(
            f"Load model: {s2-s1}. Sample {s3-s2}. Setup dir {s4-s3}. Save imgs {s5-s4}. Fid: {s6-s5}"
        )
        return epoch, fid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", required=True, type=Path)
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--gen-num-samples", default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = 0
    mp.set_start_method("spawn", force=True)
    bits = "bits:1" in args.exp_path.name
    print(bits, args.exp_path)
    args.exp_path = args.exp_path / "models"
    test_path = "dataset/DBM_MNIST_test"
    if bits:
        test_path += "_binary"
    save_mnist_test_images(test_path, bits)
    fid_epoch: dict[int, float] = {}
    fn_args = [
        (model_wt_path, args.gen_num_samples, args.exp_path, device, test_path)
        for model_wt_path in sorted(args.exp_path.glob("*.pt"))
    ]
    with mp.Pool(4) as p:
        for epoch, fid in tqdm.tqdm(
            p.imap(worker, fn_args), desc=f"{args.exp_path}", total=len(fn_args)
        ):
            fid_epoch[epoch] = fid
    df = pd.DataFrame(list(fid_epoch.items()), columns=["Epoch", "FID"])
    df.to_csv(args.save_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
