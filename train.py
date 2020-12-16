#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, get_net_path
from dataset import FoviatedLODDataset
from network import Net
from torch_utils import extract_labels
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
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
    criterions = [nn.CrossEntropyLoss() for _ in range(722)]
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

    total_iters = 0
    for epoch in range(1, opt.num_epochs + 1):
        epoch_iters = 0

        # Validation & Test
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for i, data in enumerate(val_loader):
                inps = data["input"]
                labels = data["output"]["updated_lod"]
                outputs = net(inps)
                predicted = torch.cat(
                    [extract_labels(outs) for outs in outputs], dim=1
                )
                val_total += torch.flatten(labels).size(0)
                val_correct += (predicted == labels).sum().item()
            test_correct = 0
            test_total = 0
            for i, data in enumerate(test_loader):
                inps = data["input"]
                labels = data["output"]["updated_lod"]
                outputs = net(inps)
                predicted = torch.cat(
                    [extract_labels(outs) for outs in outputs], dim=1
                )
                test_total += torch.flatten(labels).size(0)
                test_correct += (predicted == labels).sum().item()
            visualizer.plot_current_accuracy(
                epoch,
                {
                    "validation": val_correct / val_total,
                    "test": test_correct / test_total,
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
            loss = 0
            for i in range(len(outputs)):
                loss += criterions[i](outputs[i], labels[..., i])
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
