__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import torch
from scipy.ndimage import binary_dilation, grey_dilation, median_filter
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from tqdm import tqdm


def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[1]):
            volume[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def connected_components_3d(volume):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop["filled_area"] <= 20:
                volume[i, cc_volume == prop["label"]] = 0
    return torch.Tensor(volume)


def gray_dilation(volume, kernelsize=3):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Gray Dilation")
    for i in pbar:
        volume[i] = grey_dilation(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def bin_dilation(volume, structure):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Binray Dilation")
    for i in pbar:
        volume[i] = binary_dilation(volume[i], structure=structure)
    return torch.Tensor(volume)


def norm_tensor(tensor):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    my_tensor = (tensor - my_min) / (my_max - my_min)
    return my_tensor


def gmean(input_x, dim, keepdim=False):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim, keepdim=keepdim))
