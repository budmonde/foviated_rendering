#!/usr/bin/env python3
import json
import os

import numpy as np
from torch.utils.data import Dataset

from constants import DATA_PATH, NUM_POPPING_VECTORS, NUM_TRIANGLES

# with open("./bad_data.txt") as f:
#     exclude_set = set(f.read().strip().split("\n"))


def convertDictToArray(dictionary):
    output = np.zeros(NUM_TRIANGLES, dtype=np.float32)
    keys, values = zip(*dictionary.items())
    np.put(output, keys, values)
    return output


class FoviatedLODDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        child_dirs = filter(
            lambda d: d.startswith("seq"),
            os.listdir(root_dir),
        )

        def get_datapoint_paths(dirname):
            dirpath = os.path.join(root_dir, dirname)
            return [
                os.path.join(dirpath, fname)
                for fname in filter(
                    lambda d: d.endswith(".json"), os.listdir(dirpath)
                )
            ]

        self.paths = [
            item
            for sublist in [get_datapoint_paths(d) for d in child_dirs]
            for item in sublist
        ]
        self.paths.sort()

        # Excluding bad samples
        # self.paths = list(filter(lambda p: p not in exclude_set, self.paths))

        # Every 10th sample will be a test sample
        if mode == "train":
            self.paths = [
                item
                for idx, item in enumerate(self.paths)
                if (idx + 1) % 10 != 0 and "seq9" not in item
            ]
        elif mode == "validation":
            self.paths = [
                item
                for idx, item in enumerate(self.paths)
                if (idx + 1) % 10 == 0 and "seq9" not in item
            ]
        elif mode == "test":
            self.paths = [
                item for idx, item in enumerate(self.paths) if "seq9" in item
            ]
        else:
            raise Exception("Invalid dataset mode. Got: %s", mode)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        datapoint = json.load(open(self.paths[idx]))
        inp = np.concatenate(
            (
                np.array(datapoint["triangleLOD"], dtype=np.float32),
                np.array(datapoint["eye"], dtype=np.float32),
                np.array(datapoint["lookat"], dtype=np.float32),
                np.array(datapoint["up"], dtype=np.float32),
                np.array(datapoint["gaze"], dtype=np.float32),
            )
        )
        out = {
            "eccentricity_score": convertDictToArray(
                datapoint["eccentricityScore"]
            ),
            "popping_score": [
                convertDictToArray(datapoint["poppingScore"][i])
                for i in range(NUM_POPPING_VECTORS)
            ],
            "updated_lod": np.array(datapoint["updatedLOD"]),
        }
        return {
            "path": self.paths[idx],
            "input": inp,
            "output": out,
        }


def debug():
    train = FoviatedLODDataset(DATA_PATH, mode="train")
    print("Training set size:", len(train))
    val = FoviatedLODDataset(DATA_PATH, mode="validation")
    print("Validation set size:", len(val))
    test = FoviatedLODDataset(DATA_PATH, mode="test")
    print("Test set size:", len(test))

    scores = []
    for data in train:
        scores.append(data["output"]["eccentricity_score"])
    scores = np.concatenate(scores)
    print("Max", np.max(scores))
    print("Median", np.median(scores))
    print("Min", np.min(scores))


if __name__ == "__main__":
    debug()
