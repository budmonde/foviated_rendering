#!/usr/bin/env python3
import json
import os

DATA_PATH = "./Data1227/"

data = json.load(open(os.path.join(DATA_PATH, "seq0/0.json")))

NUM_TRIANGLES = len(data["triangleLOD"])
NUM_POPPING_VECTORS = len(data["poppingScore"])
NUM_LOD_LEVELS = NUM_POPPING_VECTORS + 1
CAMERA_VECTOR_DIM = len(data["eye"]) + len(data["lookat"]) + len(data["up"])
GAZE_VECTOR_DIM = len(data["gaze"])


def get_net_path(opt, net_name, epoch):
    root = os.path.join(opt.weightspath, net_name)
    if not os.path.isdir(root):
        os.makedirs(root)
    return os.path.join(root, f"{opt.lr}_{opt.batch_size}_{epoch}.pth")
