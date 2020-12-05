#!/usr/bin/env python3
DATA_PATH = "./Data/"


def get_net_path(triangle_idx, lr, batch_sz, epoch):
    return f"./weights/updated_lod_{triangle_idx}_{lr}_{batch_sz}_{epoch}.pth"


NUM_INPUT_DIM = 734
NUM_LOD_LEVELS = 6
NUM_POPPING_VECTORS = 5
NUM_TRIANGLES = 722
