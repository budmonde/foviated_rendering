#!/usr/bin/env python3
import os

DATA_PATH = "./Data3/"

CAMERA_VECTOR_DIM = 12
NUM_LOD_LEVELS = 5
NUM_POPPING_VECTORS = NUM_LOD_LEVELS - 1
NUM_TRIANGLES = 722


def get_net_paths(opt, epoch):
    return [
        os.path.join(
            opt.weightspath,
            f"popping_weights_{idx}_{opt.lrs[idx]}_"
            f"{opt.batch_size}_{epoch}.pth",
        )
        for idx in range(NUM_POPPING_VECTORS)
    ]
