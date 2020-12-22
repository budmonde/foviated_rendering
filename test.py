#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, NUM_POPPING_VECTORS, get_net_paths
from dataset import FoviatedLODDataset
from network import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=DATA_PATH)
    parser.add_argument("--weightspath", type=str, default="./weights/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument(
        "--lrs",
        nargs=NUM_POPPING_VECTORS,
        type=float,
        default=[0.001] * NUM_POPPING_VECTORS,
    )
    opt = parser.parse_args()

    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    dataset = FoviatedLODDataset(opt.datapath, mode="test")
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    nets = [Net().to(opt.device) for _ in range(NUM_POPPING_VECTORS)]
    net_paths = get_net_paths(opt, opt.num_epochs)
    criterions = [
        nn.MSELoss().to(opt.device) for _ in range(NUM_POPPING_VECTORS)
    ]
    print(f"Loading weights in {net_paths}")
    for pi in range(NUM_POPPING_VECTORS):
        nets[pi].load_state_dict(
            torch.load(net_paths[pi], map_location=opt.device)
        )

    with torch.no_grad():
        losses = [0] * NUM_POPPING_VECTORS
        for data in dataloader:
            inps = data["input"].to(opt.device)
            labels = data["output"]["popping_score"]
            for pi in range(NUM_POPPING_VECTORS):
                outputs = nets[pi](inps)
                losses[pi] += criterions[pi](
                    outputs, labels[pi].to(opt.device)
                )
        print(losses)
