__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os
import pickle

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LMDBDataset(Dataset):
    """To load slices saved in a Lightning Memory-Mapped Database (LMDB).
    The numpy files in the LMDB are created by the split_healthy.py script.
    These are all preprocessed slices, i.e. they are normalized. Used for training.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(self, directory: str, my_transforms: transforms):
        super().__init__()
        self.directory = directory
        self.transforms = my_transforms
        env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not hasattr(self, "txn"):
            self.open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        unpacked = pickle.loads(byteflow)
        tensor = torch.from_numpy(unpacked).float()
        tensor = self.transforms(tensor)
        return tensor


class MRIDataVolume(Dataset):
    """The data set class to load and normalize complete volumes. Used for evaluation.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        image_size: int,
        hist: bool,
        shift: bool,
    ):
        self.df = df
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.shift = shift
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if self.shift == True:
            id_ = self.df.loc[idx, self.df.columns[0]]
            patient_number = id_.split("_")[-1]
        else:
            id_ = self.df.loc[idx, self.df.columns[0]]
            patient_number = id_
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, patient_number + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, patient_number + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=float)
        mask[mask > 0.5] = (
            1  # for data sets where the gt masks need to be registered a thr needs to be decided
        )
        mask[mask < 1] = 0
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask[None, None], [128, 128, 155], mode="nearest-exact")
        mask = mask[0, 0].type(torch.bool)
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())

        volume = torch.zeros(
            img.shape[0], self.image_size, self.image_size, mask.shape[2]
        )
        my_transform = transforms.Resize(self.image_size, antialias=True)
        for i in range(mask.shape[2]):
            volume[:, :, :, i] = my_transform(img[None, :, :, :, i])

        return volume, mask


def Brats21(conf):
    if conf["horizontal_flip"] != None:
        my_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(conf["horizontal_flip"])]
        )
    else:
        my_transforms = transforms.Compose([])
    dataset = LMDBDataset(conf["dataset_path"], my_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )
    return dataloader


def MRI_Volume(conf, hist=False, shift=False):
    df = pd.read_csv(conf["path_to_csv"])
    dataset = MRIDataVolume(
        df, conf["dataset_path"], conf["size"], hist=hist, shift=shift
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=False,
    )
    return dataloader


def normalize_volume(images):
    """Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor containing the normalized volumes.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = torch.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99

    return images


def hist_norm(images):
    """Applies histogram normalization to the volumes.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor with the histogram normalized.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality, :, :, :] = i_
    return torch.Tensor(images)


class BratsDataset(Dataset):
    """DO NOT USE DEPRECATED CODE.
    The data set class to load individual slices from the files containing the volumes.
    A .csv is requirred that specifies the slices to load from the volume.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self,
        df: pd.DataFrame,
        my_transforms: transforms,
        dataset_path: str,
        image_size: int,
        hist: bool,
    ):
        self.df = df
        self.transforms = my_transforms
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, self.df.columns[0]]
        images = []
        slice = self.df.loc[idx, "Slice"]
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, id_ + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj[:, :, slice], dtype=int)
        mask[mask >= 1] = 1  # mask contains the labels 1, 2, and 4 for BraTS
        mask = torch.from_numpy(mask)
        mask = mask[None, :, :]
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())
        img = img[:, :, :, slice]
        img = self.transforms(img)
        my_transform = transforms.Resize(self.image_size, antialias=True)
        mask = my_transform(mask)
        return img, mask
