#!/usr/bin/env python3
import torch.nn as nn

from constants import CAMERA_VECTOR_DIM, NUM_LOD_LEVELS, NUM_TRIANGLES


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            *[
                nn.Linear(CAMERA_VECTOR_DIM + NUM_TRIANGLES, 5000),
                nn.ReLU(),
                nn.Linear(5000, 5000),
                nn.ReLU(),
                nn.Linear(5000, NUM_LOD_LEVELS * NUM_TRIANGLES),
            ]
        )

    def forward(self, inp):
        out = self.model(inp)
        return out.view(*out.shape[:-1], NUM_LOD_LEVELS, NUM_TRIANGLES)
