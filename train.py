#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, NUM_POPPING_VECTORS, get_net_paths
from dataset import FoviatedLODDataset
from network import Net
from visualizer import Visualizer

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

    visualizer = Visualizer(opt)

    train = FoviatedLODDataset(opt.datapath, mode="train")
    train_loader = DataLoader(
        train, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    val = FoviatedLODDataset(opt.datapath, mode="validation")
    val_loader = DataLoader(
        val, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    test = FoviatedLODDataset(opt.datapath, mode="test")
    test_loader = DataLoader(
        test, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    nets = [Net().to(device=opt.device) for _ in range(NUM_POPPING_VECTORS)]
    criterions = [
        nn.MSELoss().to(device=opt.device) for _ in range(NUM_POPPING_VECTORS)
    ]
    optimizers = [
        torch.optim.SGD(nets[pi].parameters(), lr=opt.lrs[pi], momentum=0.9)
        for pi in range(NUM_POPPING_VECTORS)
    ]

    total_iters = 0
    for epoch in range(1, opt.num_epochs + 1):
        epoch_iters = 0

        # Validation & Test
        with torch.no_grad():
            val_losses = [0] * NUM_POPPING_VECTORS
            val_total = 0
            for i, data in enumerate(val_loader):
                inps = data["input"].to(opt.device)
                labels = data["output"]["popping_score"]
                for pi in range(NUM_POPPING_VECTORS):
                    outputs = nets[pi](inps)
                    val_losses[pi] += criterions[pi](
                        outputs, labels[pi].to(opt.device)
                    )
                val_total += 1

            visualizer.plot_series(
                "validation_losses",
                1,
                epoch,
                {
                    f"Validation Loss {pi}": (
                        val_losses[pi].item() / val_total
                    )
                    ** 0.5
                    for pi in range(NUM_POPPING_VECTORS)
                },
            )

            test_losses = [0] * NUM_POPPING_VECTORS
            test_total = 0
            for i, data in enumerate(test_loader):
                inps = data["input"].to(opt.device)
                labels = data["output"]["popping_score"]
                for pi in range(NUM_POPPING_VECTORS):
                    outputs = nets[pi](inps)
                    test_losses[pi] += criterions[pi](
                        outputs, labels[pi].to(opt.device)
                    )
                test_total += 1

            visualizer.plot_series(
                "test_losses",
                2,
                epoch,
                {
                    f"Test Loss {pi}": (test_losses[pi].item() / test_total)
                    ** 0.5
                    for pi in range(NUM_POPPING_VECTORS)
                },
            )

        # Train
        for i, data in enumerate(train_loader):
            total_iters += opt.batch_size
            epoch_iters += opt.batch_size

            inps = data["input"].to(opt.device)
            labels = data["output"]["popping_score"]

            losses = []
            for pi in range(NUM_POPPING_VECTORS):
                optimizers[pi].zero_grad()
                outputs = nets[pi](inps)
                loss = criterions[pi](outputs, labels[pi].to(opt.device))
                loss.backward()
                optimizers[pi].step()
                losses.append(loss.item())

            if total_iters % 128 == 0:
                visualizer.print_progress_bar(
                    epoch - 1,
                    float(epoch_iters) / len(train),
                )
                visualizer.plot_series(
                    "training_loss",
                    0,
                    epoch + float(epoch_iters) / len(train),
                    {
                        f"Popping Loss {pi}": losses[pi] ** 0.5
                        for pi in range(NUM_POPPING_VECTORS)
                    },
                )

        if epoch % 100 == 0:
            net_paths = get_net_paths(opt, epoch)
            for pi in range(NUM_POPPING_VECTORS):
                torch.save(nets[pi].state_dict(), net_paths[pi])
            print("Saved nets at %s" % net_paths)
