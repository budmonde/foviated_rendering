#!/usr/bin/env python3
DATA_PATH = "./Data/"


def get_net_path(lr, batch_sz, epoch):
    return f"./weights_v3/whole_scene_{lr}_{batch_sz}_{epoch}.pth"


CAMERA_VECTOR_DIM = 12
NUM_LOD_LEVELS = 6
NUM_POPPING_VECTORS = 5
NUM_TRIANGLES = 722
