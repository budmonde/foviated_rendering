#!/usr/bin/env python3
import json
import os

import numpy as np
from torch.utils.data import Dataset

from constants import NUM_POPPING_VECTORS, NUM_TRIANGLES

with open("./bad_data.txt") as f:
    exclude_set = set(f.read().strip().split("\n"))


def convertDictToArray(dictionary):
    # NOTE: we use a magic number of 722 here because the scene has this many
    #       triangles
    output = np.zeros(NUM_TRIANGLES, dtype=np.float32)
    keys, values = zip(*dictionary.items())
    np.put(output, keys, values)
    return output


class FoviatedLODDataset(Dataset):
    def __init__(self, root_dir, mode="train", triangle_idx=0):
        self.triangle_idx = triangle_idx

        def get_datapoint_paths(dirname):
            dirpath = os.path.join(root_dir, dirname)
            return [
                os.path.join(dirpath, fname)
                for fname in filter(
                    lambda d: d.endswith(".json"), os.listdir(dirpath)
                )
            ]

        child_dirs = filter(
            lambda d: d.startswith("seq"),
            os.listdir("./Data/"),
        )
        self.paths = [
            item
            for sublist in [get_datapoint_paths(d) for d in child_dirs]
            for item in sublist
        ]
        self.paths.sort()

        # Excluding bad samples
        self.paths = list(filter(lambda p: p not in exclude_set, self.paths))

        # Every 10th sample will be a test sample
        if mode == "train":
            self.paths = [
                item
                for idx, item in enumerate(self.paths)
                if (idx + 1) % 10 != 0
            ]
        if mode == "test":
            self.paths = [
                item
                for idx, item in enumerate(self.paths)
                if (idx + 1) % 10 == 0
            ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        datapoint = json.load(open(self.paths[idx]))
        inp = np.concatenate(
            [
                np.array(datapoint["triangleLOD"], dtype=np.float32),
                np.array(datapoint["eye"], dtype=np.float32),
                np.array(datapoint["lookat"], dtype=np.float32),
                np.array(datapoint["up"], dtype=np.float32),
                np.array(datapoint["gaze"], dtype=np.float32),
            ]
        )
        out = {
            "eccentricity_score": convertDictToArray(
                datapoint["eccentricityScore"]
            ),
            "popping_score": [
                convertDictToArray(datapoint["poppingScore"][i])
                for i in range(NUM_POPPING_VECTORS)
            ],
            "final_score": convertDictToArray(datapoint["finalScore"]),
            # TODO: For now just testing with the first triangle
            "updated_lod": np.array(
                datapoint["updatedLOD"][self.triangle_idx]
            ),
        }
        return {
            "path": self.paths[idx],
            "input": inp,
            "output": out,
        }


def load():
    dataset = FoviatedLODDataset("./Data/", mode="test")
    print(len(dataset))
    print(dataset[0]["input"][282])
    print(dataset[1]["input"][282])
    print(dataset[2]["input"][282])
    print(dataset[3]["input"][282])
    scores = []
    for i in range(len(dataset)):
        scores.append(dataset[i]["output"]["final_score"])
    scores = np.concatenate(scores)
    print(scores.shape)
    scores = scores[scores != 0.0]
    print(scores.shape)
    print("Max", np.max(scores))
    print("Mean", np.mean(scores))
    print("Median", np.median(scores))
    print("Min", np.min(scores))

    import matplotlib.pyplot as plt

    plt.hist(scores, bins=[(x - 9.5) * 10 for x in range(20)])
    plt.title("Histogram")
    plt.show()


if __name__ == "__main__":
    load()
