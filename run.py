#!/usr/bin/env python3
import subprocess

triangle_idxs = range(0, 722, 20)
batch_size = 16
num_epochs = 40
lr = 0.01
for triangle_idx in triangle_idxs:
    cmd = [
        "python3",
        "./test.py",
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        "--lr",
        str(lr),
    ]
    # print(" ".join(cmd))
    subprocess.run(cmd)
