import torch
from torch.utils.data import WeightedRandomSampler


def build_ohem_sampler(dataset, sample_weights):

    weights = []

    for i in range(len(dataset)):
        weights.append(sample_weights.get(i, 0.2))

    return WeightedRandomSampler(
        torch.DoubleTensor(weights), len(dataset), replacement=True
    )
