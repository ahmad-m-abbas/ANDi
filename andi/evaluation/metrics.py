__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

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
