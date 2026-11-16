__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import numpy as np
import torch


def dice_stitch(pred, target):
    pred_sum = pred.view(-1).sum()
    target_sum = target.view(-1).sum()
    intersection = pred.view(-1).float() @ target.view(-1).float()
    dice = (2 * intersection) / (pred_sum + target_sum)
    return dice


def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den


def pooled_average_precision(anomaly_map, labels):
    # Only positive-score thresholds change recall in non-interpolated AP.
    # Keep ties together and count every voxel without sorting the entire split.
    positive_scores = np.concatenate([
        scores[truth].numpy() for scores, truth in zip(anomaly_map, labels)
    ])
    thresholds, positive_counts = np.unique(positive_scores, return_counts=True)
    if not len(thresholds):
        return 0.0

    voxel_counts = np.zeros(len(thresholds) + 1, dtype=np.int64)
    for volume in anomaly_map:
        bins = np.searchsorted(thresholds, volume.numpy().ravel(), side="right")
        voxel_counts += np.bincount(bins, minlength=len(voxel_counts))
    true_positives = np.cumsum(positive_counts[::-1])
    predicted_positives = np.cumsum(voxel_counts[:0:-1])
    precision = true_positives / predicted_positives
    recall_increments = positive_counts[::-1] / len(positive_scores)
    return float(np.dot(precision, recall_increments))
