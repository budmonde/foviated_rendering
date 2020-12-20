#!/usr/bin/env python3
import torch
import torch.nn as nn

from constants import CAMERA_VECTOR_DIM, NUM_LOD_LEVELS


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(CAMERA_VECTOR_DIM + 1, 10),
                        nn.ReLU(),
                        nn.Linear(10, 10),
                        nn.ReLU(),
                        nn.Linear(10, NUM_LOD_LEVELS),
                    ]
                )
                for _ in range(722)
            ]
        )

    def forward(self, inp):
        out = [
            self.model[i](
                torch.cat(
                    (inp["prior_lod"][..., i : i + 1], inp["camera"]), dim=1
                )
            )
            for i in range(722)
        ]
        return out
