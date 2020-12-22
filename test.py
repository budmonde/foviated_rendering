#!/usr/bin/env python3
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, get_net_paths, NUM_POPPING_VECTORS
from dataset import FoviatedLODDataset
from network import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument(
        "--lrs",
        nargs=NUM_POPPING_VECTORS,
        type=float,
        default=[0.001] * NUM_POPPING_VECTORS,
    )
    opt = parser.parse_args()

    dataset = FoviatedLODDataset(DATA_PATH, mode="test")
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    nets = [Net() for _ in range(NUM_POPPING_VECTORS)]
    net_paths = get_net_paths(opt.lrs, opt.batch_size, opt.num_epochs)
    criterions = [nn.MSELoss() for _ in range(NUM_POPPING_VECTORS)]
    print(f"Loading weights in {net_paths}")
    for pi in range(NUM_POPPING_VECTORS):
        nets[pi].load_state_dict(torch.load(net_paths[pi]))

    with torch.no_grad():
        losses = [0] * NUM_POPPING_VECTORS
        for data in dataloader:
            inps = data["input"]
            labels = data["output"]["popping_score"]
            for pi in range(NUM_POPPING_VECTORS):
                outputs = nets[pi](inps)
                losses[pi] += criterions[pi](outputs, labels[pi])
        print(losses)
