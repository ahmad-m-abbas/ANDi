__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import random

import torch
import torchvision.transforms.functional as tf
from torch.nn import functional as F
from torchvision import transforms


def coarse_noise(n, channels, device, noise_size=16, noise_std=0.2, image_size=128):
    noise = torch.normal(
        mean=torch.zeros(n, channels, noise_size, noise_size), std=noise_std
    ).to(device)
    noise = F.interpolate(
        noise,
        size=(image_size, image_size),
        mode="bilinear",
        antialias=False,
        align_corners=True,
    )
    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(image_size))
    roll_y = random.choice(range(image_size))
    noise = torch.roll(noise, shifts=[roll_x, roll_y], dims=[2, 3])
    return noise


def pyramid_noise_like(n, channels, image_size, discount, device):
    u = transforms.Resize(image_size, antialias=True)
    noise = torch.randn((n, channels, image_size, image_size)).to(device)
    w = image_size
    h = image_size
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(n, channels, w, h).to(device)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def random_transform_vectorized(tensor):
    # Random rotations for the entire batch
    angles = torch.randint(0, 24, (tensor.size(0),)) * 15
    sliced_tensor = torch.stack(
        [tf.rotate(tensor[i, :, :, :], angles[i].item()) for i in range(tensor.size(0))]
    )

    # Random horizontal flip for the entire batch
    flip_h = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_h[i]:
            sliced_tensor[i, :, :, :] = tf.hflip(sliced_tensor[i, :, :, :])

    # Random vertical flip for the entire batch
    flip_v = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_v[i]:
            sliced_tensor[i, :, :, :] = tf.vflip(sliced_tensor[i, :, :, :])

    return sliced_tensor
