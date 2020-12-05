#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from console_progressbar import ProgressBar
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
        DATA_PATH, mode="train", triangle_idx=opt.triangle_idx
    )
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

    pb = ProgressBar(total=opt.num_epochs)
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inps = data["input"]
            labels = data["output"]["updated_lod"]

            optimizer.zero_grad()
            outputs = net(inps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # if i % 100 == 99:
            #     print(
            #         f"[%d, %5d] loss: %.3f"
            #         % (epoch + 1, i + 1, running_loss / 100)
            #     )
            #     running_loss = 0.0

        pb.print_progress_bar(epoch + 1)
        if epoch % 20 == 19:
            net_path = get_net_path(
                opt.triangle_idx, opt.lr, opt.batch_size, epoch + 1
            )
            torch.save(net.state_dict(), net_path)
            print("Saved net at %s" % net_path)
