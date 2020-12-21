#!/usr/bin/env python3
import argparse
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from constants import DATA_PATH, get_net_path
from dataset import FoviatedLODDataset
from network import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    opt = parser.parse_args()

    dataset = FoviatedLODDataset(
        DATA_PATH,
        mode="test",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    net = Net()
    net_path = get_net_path(opt.lr, opt.batch_size, opt.num_epochs)
    print(f"Loading weights in {net_path}")
    net.load_state_dict(torch.load(net_path))

    def extract_labels(outputs):
        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        return predicted

    correct = 0
    total = 0
    num_runs = 0
    timer = defaultdict(float)
    with torch.no_grad():
        for data in dataloader:
            num_runs += 1
            time_start = time.perf_counter()
            inps = data["input"]
            labels = data["output"]["updated_lod"]
            time_load = time.perf_counter()
            outputs = net(inps)
            time_execute = time.perf_counter()
            _, predicted = torch.max(outputs.data, 1)
            time_output = time.perf_counter()

            timer["load"] += time_load - time_start
            timer["execute"] += time_execute - time_load
            timer["output"] += time_output - time_execute

            total += torch.flatten(labels).size(0)
            correct += (predicted == labels).sum().item()

    for k, v in timer.items():
        print(f"Time to {k} is {v / num_runs}")

    print("Accuracy %d %%" % (100 * correct / total))
