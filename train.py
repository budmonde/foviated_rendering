#!/usr/bin/env python3
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, NUM_TRIANGLES, get_net_path
from dataset import FoviatedLODDataset
from network import Net
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    opt = parser.parse_args()

    visualizer = Visualizer(opt)

    train = FoviatedLODDataset(DATA_PATH, mode="train")
    train_loader = DataLoader(
        train, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    val = FoviatedLODDataset(DATA_PATH, mode="validation")
    val_loader = DataLoader(
        val, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    test = FoviatedLODDataset(DATA_PATH, mode="test")
    test_loader = DataLoader(
        test, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    net = Net()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

    def calculate_acc(inps, labels, outs, acc):
        inps = inps.type(torch.LongTensor)
        acc["total"] += torch.flatten(labels).size(0)
        acc["correct"] += (outs == labels).sum().item()
        acc["p"] += (inps != labels).sum().item()
        acc["n"] += (inps == labels).sum().item()
        acc["tp"] += (
            torch.logical_and(inps != labels, inps != outs).sum().item()
        )
        acc["tn"] += (
            torch.logical_and(inps == labels, inps == outs).sum().item()
        )
        acc["fp"] += (
            torch.logical_and(inps == labels, inps != outs).sum().item()
        )
        acc["fn"] += (
            torch.logical_and(inps != labels, inps == outs).sum().item()
        )

    total_iters = 0
    for epoch in range(1, opt.num_epochs + 1):
        epoch_iters = 0

        # Validation & Test
        with torch.no_grad():
            val_acc = defaultdict(int)
            for i, data in enumerate(val_loader):
                inps = data["input"]
                labels = data["output"]["updated_lod"]
                outputs = net(inps)
                _, predicted = torch.max(outputs.data, 1)
                calculate_acc(
                    inps[..., :NUM_TRIANGLES], labels, predicted, val_acc
                )
            test_acc = defaultdict(int)
            for i, data in enumerate(test_loader):
                inps = data["input"]
                labels = data["output"]["updated_lod"]
                outputs = net(inps)
                _, predicted = torch.max(outputs.data, 1, keepdim=True)
                calculate_acc(
                    inps[..., :NUM_TRIANGLES], labels, predicted, test_acc
                )
            visualizer.plot_current_accuracy(
                epoch,
                {
                    "Val: True Positive": val_acc["tp"] / val_acc["total"],
                    "Val: True Negative": val_acc["tn"] / val_acc["total"],
                    "Val: False Positive": val_acc["fp"] / val_acc["total"],
                    "Val: False Negative": val_acc["fn"] / val_acc["total"],
                    "Target Positive": val_acc["p"] / val_acc["total"],
                    "Target Negative": val_acc["n"] / val_acc["total"],
                    "Overall Accuracy": val_acc["correct"] / val_acc["total"],
                },
            )

        # Train
        for i, data in enumerate(train_loader):
            total_iters += opt.batch_size
            epoch_iters += opt.batch_size

            inps = data["input"]
            labels = data["output"]["updated_lod"]

            optimizer.zero_grad()
            outputs = net(inps)
            loss_array = criterion(outputs, labels)

            weights = torch.zeros(labels.shape, dtype=torch.float32)
            weights[inps[..., :NUM_TRIANGLES] != labels] = 1.0

            loss = (loss_array * weights).mean()
            loss.backward()
            optimizer.step()

            if total_iters % 128 == 0:
                visualizer.print_progress_bar(
                    epoch - 1,
                    float(epoch_iters) / len(train),
                )
                visualizer.plot_current_losses(
                    epoch,
                    float(epoch_iters) / len(train),
                    {"loss1": loss.item(), "loss2": loss.item()},
                )

        if epoch % 20 == 0:
            net_path = get_net_path(opt.lr, opt.batch_size, epoch)
            torch.save(net.state_dict(), net_path)
            print("Saved net at %s" % net_path)
