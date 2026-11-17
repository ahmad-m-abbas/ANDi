__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os
import pickle

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm


def normalize_volume(images):
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = np.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99
    return images


def main(args):
    split_and_save(args)


def healthy_slice_batch(root, subject, resize, limit=None):
    """Return normalized, resized slices with nonempty FLAIR and no lesion labels."""
    directory = os.path.join(root, subject)
    mask = np.asarray(
        nib.load(os.path.join(directory, subject + "_seg.nii.gz")).dataobj, dtype=int
    )
    images = []
    for modality in ("flair", "t1", "t1ce", "t2"):
        path = os.path.join(directory, f"{subject}_{modality}.nii.gz")
        images.append(torch.from_numpy(np.asarray(nib.load(path).dataobj, dtype=float)))
    volume = normalize_volume(torch.stack(images).float())
    indices = [
        index for index in range(volume.shape[-1])
        if torch.count_nonzero(volume[0, :, :, index])
        and not (mask[:, :, index] >= 1).any()
    ][:limit]
    if not indices:
        return indices, None
    slices = resize(volume.permute(3, 0, 1, 2)[indices]).numpy()
    return indices, slices


def split_and_save(args):
    df = pd.read_csv(args.input_file)
    root_path = args.data_set
    os.makedirs(f"{root_path}/healthy_slices", exist_ok=True)
    ids = df.iloc[:, 0]
    resize = transforms.Resize(args.resolution, antialias=True)
    global_ids = []
    global_slices = []
    map_size = 2 * sum(
        entry.stat().st_size
        for subject in ids
        for entry in os.scandir(os.path.join(root_path, subject))
        if entry.is_file()
    )
    env = lmdb.open(f"{root_path}/healthy_slices", map_size=map_size)
    with env.begin(write=True) as txn:
        num = 0
        for subject in tqdm(ids):
            indices, slices = healthy_slice_batch(root_path, subject, resize)
            if slices is None:
                continue
            for image in slices:
                txn.put(f"{num:08}".encode("ascii"), pickle.dumps(image))
                num += 1
            global_ids.extend([subject] * len(indices))
            global_slices.extend(indices)
    env.close()
    healthy_dict = {df.columns[0]: global_ids, "Slice": global_slices}
    df_healthy = pd.DataFrame(healthy_dict)
    df_healthy.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data into healthy slices.")
    parser.add_argument(
        "-d",
        "--data_set",
        type=str,
        required=True,
        metavar="",
        help="The folder that contains the MRI-Volumes containing anomalies.",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        metavar="",
        help="The .csv that specifies the volumes to split into healthy slices.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        metavar="",
        help="The .csv path for the file that specifies the healthy slices.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        metavar="",
        help="The resolution of the stored slices, if save flag is set to true.",
    )
    args = parser.parse_args()
    main(args)
