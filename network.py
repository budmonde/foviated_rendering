#!/usr/bin/env python3
import torch.nn as nn

from constants import CAMERA_VECTOR_DIM, NUM_TRIANGLES


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            *[
                nn.Linear(CAMERA_VECTOR_DIM, 100),
                nn.ReLU(),
                nn.Linear(100, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, NUM_TRIANGLES),
                nn.ReLU(),
            ]
        )

    def forward(self, inp):
        return self.model(inp)
