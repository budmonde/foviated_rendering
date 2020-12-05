#!/usr/bin/env python3
import argparse

import torch
from torch.utils.data import DataLoader

from constants import DATA_PATH, get_net_path
from dataset import FoviatedLODDataset
from network import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangle_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    opt = parser.parse_args()

    dataset = FoviatedLODDataset(
        DATA_PATH,
        mode="test",
        triangle_idx=opt.triangle_idx,
    )

    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    dataiter = iter(dataloader)
    data = dataiter.next()
    inps = data["input"]
    labels = data["output"]["updated_lod"]

    # print(
    #    "GroundTruth: : ",
    #    " ".join("%5s" % labels[j] for j in range(opt.batch_size)),
    # )

    net = Net()
    net_path = get_net_path(
        opt.triangle_idx, opt.lr, opt.batch_size, opt.num_epochs
    )
    net.load_state_dict(torch.load(net_path))

    outputs = net(inps)
    _, predicted = torch.max(outputs, 1)
    # print(
    #    "Predicted: ",
    #    " ".join("%5s" % predicted[j] for j in range(opt.batch_size)),
    # )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inps = data["input"]
            labels = data["output"]["updated_lod"]
            outputs = net(inps)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Triangle %d accuracy %d %%"
        % (opt.triangle_idx, 100 * correct / total)
    )
