#!/usr/bin/env python3
import json
import os

import numpy as np
from torch.utils.data import Dataset

from constants import DATA_PATH, NUM_POPPING_VECTORS, NUM_TRIANGLES


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

        # Every 10th sample will be a validation sample
        # All samples from seq9 will be a test sample
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

        inp = {
            "camera": np.concatenate(
                (
                    np.array(datapoint["eye"], dtype=np.float32),
                    np.array(datapoint["lookat"], dtype=np.float32),
                    np.array(datapoint["up"], dtype=np.float32),
                )
            ),
            "gaze": np.array(datapoint["gaze"], dtype=np.float32),
        }

        area = np.array(datapoint["projectedAreas"], dtype=np.float32)

        def calculate_density(score):
            return np.divide(
                score, area, out=np.zeros_like(score), where=area != 0.0
            )

        popping_scores = [
            np.array(datapoint["poppingScore"][pi], dtype=np.float32)
            for pi in range(NUM_POPPING_VECTORS)
        ]
        popping_density_list = [
            calculate_density(popping_scores[pi] / 40)
            for pi in range(NUM_POPPING_VECTORS)
        ]
        no_mask_popping_scores = [
            np.array(datapoint["poppingScoreNoMask"][pi], dtype=np.float32)
            for pi in range(NUM_POPPING_VECTORS)
        ]
        no_mask_popping_density_list = [
            calculate_density(no_mask_popping_scores[pi] / 50)
            for pi in range(NUM_POPPING_VECTORS)
        ]
        eccentricity_score = np.array(
            datapoint["eccentricityScore"], dtype=np.float32
        )
        eccentricity_density = calculate_density(eccentricity_score / 20)

        out = {
            "popping_density_list": popping_density_list,
            "popping_score_list": popping_scores,
            "no_mask_popping_density_list": no_mask_popping_density_list,
            "no_mask_popping_score_list": no_mask_popping_scores,
            "eccentricity_density": eccentricity_density,
            "eccentricity_score": eccentricity_score,
            "area": area,
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

    pop_scores_0 = [[] for _ in range(NUM_TRIANGLES)]
    pop_scores_processed = [[] for _ in range(NUM_TRIANGLES)]
    for data in test:
        for ti in range(NUM_TRIANGLES):
            pop_scores_0[ti].append(
                data["output"]["popping_score_list"][0][ti]
            )
            pop_scores_processed[ti].append(
                data["output"]["popping_density_list"][0][ti]
                * 40
                * data["output"]["area"][ti]
            )

    from matplotlib import pyplot as plt

    for ti in range(NUM_TRIANGLES):
        plt.plot(range(len(test)), pop_scores_0[ti])

    plt.show()
    densities = []
    for data in train:
        densities.extend(data["output"]["no_mask_popping_density_list"])
    for data in val:
        densities.extend(data["output"]["no_mask_popping_density_list"])
    for data in test:
        densities.extend(data["output"]["no_mask_popping_density_list"])
    densities = np.array(densities)

    print("Max", np.max(densities))
    print("Median", np.median(densities))
    print("Min", np.min(densities))

    # counts, bins = np.histogram(densities)
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.title("TARGET HISTOGRAM")
    # plt.show()


if __name__ == "__main__":
    debug()
