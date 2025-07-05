import argparse
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import uuid

def parse_args():
    """Parse command-line arguments for training."""
    p = argparse.ArgumentParser()
    
    # parameters
    p.add_argument(
        "--NUM_EPOCHS", type=int, default=50
    )
    
    p.add_argument(
        "--NOISE_DIMENSION", type=int, default=50
    )
    
    p.add_argument(
        "--BATCH_SIZE", type=int, default=128
    )
    
    p.add_argument(
        "--TRAIN_ON_GPU", type=bool, default=False
    )
    
    p.add_argument(
        "--UNIQUE_RUN_ID", type=str, default=str(uuid.uuid4())
    )
    
    p.add_argument(
        "--PRINT_STATS_AFTER_BATCH", type=int, default=50
    )
    
    p.add_argument(
        "--OPTIMIZER_LR", type=float, default=2e-4
    )
    
    p.add_argument(
        "--OPTIMIZER_BETAS", type=float, nargs=2, default=(0.5, 0.999)
    )
    
    p.add_argument(
        "--GENERATOR_OUTPUT_IMAGE_SHAPE", type=int, default=28 * 28 * 1
    )
    
    return p.parse_args()


def main():
    cfg = parse_args()

    # ---------- Initialization ----------
    # Speed ups
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    
    # ---------- Data Preprocessing ----------
    
    # ---------- Initialize Models ----------
    
    # ---------- Train ----------
    
if __name__ == "__main__":
    main()