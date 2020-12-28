#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, NUM_POPPING_VECTORS, get_net_path
from dataset import FoviatedLODDataset
from network import FixationNet
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O Config
    parser.add_argument("--datapath", type=str, default=DATA_PATH)
    parser.add_argument("--weightspath", type=str, default="./weights/")
    # Training Config
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    # GPU Config
    parser.add_argument("--disable_cuda", action="store_true")
    # Visualization Config
    parser.add_argument("--env", type=str, default="test_fixation")
    parser.add_argument("--triangle_id", type=int, default=0)
    opt = parser.parse_args()

    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    visualizer = Visualizer(opt)

    dataset = FoviatedLODDataset(opt.datapath, mode="test")
    dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=1
    )

    net = FixationNet().to(device=opt.device)
    popping_criterions = [
        nn.MSELoss().to(device=opt.device) for _ in range(NUM_POPPING_VECTORS)
    ]
    eccentricity_criterion = nn.MSELoss().to(device=opt.device)

    net_path = get_net_path(opt, "fixation_net", opt.num_epochs)
    print(f"Loading weights in {net_path}")
    net.load_state_dict(torch.load(net_path, map_location=opt.device))

    with torch.no_grad():
        for data in dataloader:
            inps = torch.cat(
                [data["input"]["camera"], data["input"]["gaze"]], dim=1
            ).to(opt.device)
            popping_labels = [
                data["output"]["popping_density_list"][pi].to(opt.device)
                for pi in range(NUM_POPPING_VECTORS)
            ]
            eccentricity_labels = data["output"]["eccentricity_density"].to(
                opt.device
            )

            outputs = net(inps)

            loss = 0
            for pi in range(NUM_POPPING_VECTORS):
                loss += popping_criterions[pi](
                    outputs["popping_density_list"][pi],
                    popping_labels[pi],
                )
            loss += 1.0 * eccentricity_criterion(
                outputs["eccentricity_density"], eccentricity_labels
            )
            print(loss.item())

            sample_target_output = {
                "popping_score_list": [
                    data["output"]["popping_score_list"][pi][
                        :, opt.triangle_id
                    ]
                    .cpu()
                    .numpy()
                    for pi in range(NUM_POPPING_VECTORS)
                ],
                "eccentricity_score": (
                    data["output"]["eccentricity_score"][:, opt.triangle_id]
                )
                .cpu()
                .numpy(),
            }
            area = data["output"]["area"].to(opt.device)
            sample_gen_output = {
                "popping_score_list": [
                    (outputs["popping_density_list"][pi] * 40 * area)[
                        :, opt.triangle_id
                    ]
                    .cpu()
                    .numpy()
                    for pi in range(NUM_POPPING_VECTORS)
                ],
                "eccentricity_score": (
                    (outputs["eccentricity_density"] * 20 * area)[
                        :, opt.triangle_id
                    ]
                )
                .cpu()
                .numpy(),
            }

    for pi in range(NUM_POPPING_VECTORS):
        visualizer.plot_whole(
            f"Popping Scores [{pi}] for seq9",
            pi,
            list(range(len(sample_gen_output["popping_score_list"][pi]))),
            {
                "Target": sample_target_output["popping_score_list"][pi],
                "Gen": sample_gen_output["popping_score_list"][pi],
            },
        )
    visualizer.plot_whole(
        "Eccentricity Score for seq9",
        NUM_POPPING_VECTORS,
        list(range(len(sample_gen_output["eccentricity_score"]))),
        {
            "Target": sample_target_output["eccentricity_score"],
            "Gen": sample_gen_output["eccentricity_score"],
        },
    )
