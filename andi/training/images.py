__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image


def plot_images(images, mode="RGB"):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu(),
            cmap="gray",
        )
        plt.show()
    else:
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu()
        )
        plt.show()


def save_images(images, path, mode="RGB", **kwargs):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


def upload_images(images, mode="RGB", **kwargs):
    """Creates a numpy array to upload the images to wandb.

    Parameters
    ----------
    images : tensor
        The tensor containing the images
    mode : str, optional
        flag that decides if the images are meant to be RBG or gray scale, by default "RGB"

    Returns
    -------
    numpy.array
        The array that can be uploaded to wandb
    """
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr
