#!/usr/bin/env python3
DATA_PATH = "./Data2/"

CAMERA_VECTOR_DIM = 12
NUM_LOD_LEVELS = 5
NUM_POPPING_VECTORS = NUM_LOD_LEVELS - 1
NUM_TRIANGLES = 722


def get_net_paths(lrs, batch_sz, epoch):
    return [
        f"./weights/popping_weights_{idx}_{lrs[idx]}_{batch_sz}_{epoch}.pth"
        for idx in range(NUM_POPPING_VECTORS)
    ]
