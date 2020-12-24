#!/usr/bin/env python3
import argparse

import numpy as np
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
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1
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

    out_0_202 = None
    with torch.no_grad():
        rhos = []
        for data in dataloader:
            losses = [0] * NUM_POPPING_VECTORS
            inps = data["input"].to(opt.device)
            labels = data["output"]["popping_scores"]
            losses = []
            for pi in range(NUM_POPPING_VECTORS):
                outputs = nets[pi](inps)
                popping_score = (
                    outputs
                    * 100
                    * (
                        data["output"]["count"][pi]
                        + data["output"]["count"][pi + 1]
                    ).to(opt.device)
                )
                if pi == 0:
                    out_0_202 = popping_score[:, 202].cpu().numpy()
                loss = criterions[pi](popping_score, labels[pi].to(opt.device))
                losses.append(loss.item() ** 0.5)
                rhos.append(torch.flatten(outputs).cpu().numpy())

            print(losses)
        rhos = np.concatenate(rhos)

        print("Max", np.max(rhos))
        print("Median", np.median(rhos))
        print("Min", np.min(rhos))

        from matplotlib import pyplot as plt

        plt.hist(rhos, bins=[x / 100 for x in range(-25, 25)])
        plt.title("OUTPUT HISTOGRAM")
        plt.show()

    target_0_202 = []
    for data in dataset:
        target_0_202.append(data["output"]["popping_scores"][0][202])

    plt.plot(range(len(target_0_202)), target_0_202)
    plt.plot(range(len(target_0_202)), out_0_202)
    plt.show()
