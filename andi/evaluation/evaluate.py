__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
from pathlib import Path

import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_yen

from andi.diffusion.process import *
from andi.models.unet import *
from andi.utils import *
from andi.evaluation.metrics import pooled_average_precision


def infer(conf, cache_path):
    torch.manual_seed(conf["seed"])
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError(f"BraTS21 reproduction requires MPS, got {device}")
    print(f"Inference device: {device}", flush=True)
    model = UNet(device=device).to(device)
    ckpt = torch.load(conf["model"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(
        noise_steps=conf["noise_steps"],
        img_size=conf["size"],
        beta_start=conf["beta_start"],
        beta_end=conf["beta_end"],
        device=device,
    )
    dataloader = MRI_Volume(
        conf, hist=False, shift="shifts" in conf["dataset_path"]
    )
    model, dataloader = accelerator.prepare(model, dataloader)
    cache_path.mkdir(parents=True, exist_ok=False)

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(dataloader, desc="Volumes")):
            image = (image * 2) - 1
            num_volumes = image.shape[0]
            num_slices = image.shape[4]
            image = torch.permute(image, (0, 4, 1, 2, 3))
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            aggregations = []
            for image_split in torch.split(image, conf["size_splits"]):
                dts = diffusion.normative_diffusion(
                    model,
                    image_split,
                    conf["start"],
                    conf["stop"],
                    conf["pyramid"],
                    conf["discount"],
                ).cpu()
                aggregation = gmean(dts, dim=1) if conf["gmean"] else torch.mean(dts, dim=1)
                aggregations.append(aggregation)
                del dts

            aggregation = torch.cat(aggregations, dim=0)
            aggregation = aggregation.view(num_volumes, num_slices, *aggregation.shape[1:])
            aggregation = torch.permute(aggregation, (0, 2, 3, 4, 1))
            if conf["max"]:
                anomaly_map = torch.max(aggregation, dim=1)[0]
            else:
                anomaly_map = torch.mean(aggregation, dim=1)
            if not torch.isfinite(anomaly_map).all():
                raise RuntimeError(f"Non-finite anomaly map in batch {i}")
            pending_path = cache_path / f"{i:04d}.tmp"
            torch.save((anomaly_map.contiguous(), label.bool().cpu()), pending_path)
            pending_path.rename(cache_path / f"{i:04d}.pt")

    (cache_path / "complete").touch()


def score_maps(conf, anomaly_map, labels, output):
    anomaly_map = norm_tensor(anomaly_map).contiguous()
    thresholds = [
        round(x, 3)
        for x in np.arange(conf["thr_start"], conf["thr_end"], conf["thr_step"])
    ]
    dice_scores = {}
    # Test-label sweep is the paper's descriptive ceiling Dice only.
    for threshold in tqdm(thresholds, desc="Ceiling Dice"):
        dice_scores[threshold] = np.mean(
            dice(anomaly_map > threshold, labels).numpy(), dtype=np.float64
        )

    struc = generate_binary_structure(conf["rank"], conf["connectivity"])
    yen_segmentation = torch.empty_like(labels)
    for j, volume in enumerate(anomaly_map):
        yen_segmentation[j] = volume > threshold_yen(volume.numpy())
    yen_segmentation = bin_dilation(yen_segmentation, struc)
    dice_scores["yen"] = np.mean(dice(yen_segmentation, labels).numpy(), dtype=np.float64)
    dice_scores["AUPRC"] = pooled_average_precision(anomaly_map, labels)
    df = pd.DataFrame.from_dict(dice_scores, orient="index", columns=["value"])
    df.index.name = "thr"
    df.to_csv(output)
    print(
        f"{output}: AUPRC {dice_scores['AUPRC']:.6f}, "
        f"ceiling Dice {max(dice_scores[t] for t in thresholds):.6f}, "
        f"Yen Dice {dice_scores['yen']:.6f}",
        flush=True,
    )


def score(conf, cache_path):
    if not (cache_path / "complete").is_file():
        raise RuntimeError(f"Inference cache is incomplete: {cache_path}")
    batches = sorted(cache_path.glob("*.pt"))
    maps, labels = zip(*(
        torch.load(batch, map_location="cpu", weights_only=True) for batch in batches
    ))
    anomaly_map = torch.cat(maps, dim=0)
    labels = torch.cat(labels, dim=0).contiguous()
    del maps
    if len(anomaly_map) != len(pd.read_csv(conf["path_to_csv"])):
        raise RuntimeError("Cached volume count does not match the split")

    score_maps(conf, anomaly_map, labels, conf["output"])
    filter_sizes = {3: str(Path(conf["output_mf"]).with_name("ANDi_mf3.csv"))}
    filter_sizes[conf["kernel_size"]] = conf["output_mf"]
    for kernel_size, output in filter_sizes.items():
        filtered = median_filter_3D(anomaly_map.clone(), kernelsize=kernel_size)
        score_maps(conf, filtered, labels, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./conf/eval.yml")
    parser.add_argument("--stage", choices=("infer", "score", "both"), default="both")
    args = parser.parse_args()
    with open(args.config) as config_file:
        conf = yaml.safe_load(config_file)
    cache_path = Path(conf["output"]).parent / "maps"
    if args.stage in ("infer", "both"):
        infer(conf, cache_path)
    if args.stage in ("score", "both"):
        score(conf, cache_path)


if __name__ == "__main__":
    main()
