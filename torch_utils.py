#!/usr/bin/env python3
import torch


def extract_labels(outputs):
    _, predicted = torch.max(outputs.data, 1, keepdim=True)
    return predicted
