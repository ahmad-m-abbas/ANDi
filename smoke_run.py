"""One validation volume and a tiny real training run; no production configs changed."""

import argparse
import json
import os
from pathlib import Path
import pickle
import random
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('both', 'infer', 'train'), default='both')
    parser.add_argument('--data', type=Path, default=Path('data/BraTS21'))
    parser.add_argument('--checkpoint', type=Path, default=Path('pyramid/232_ema_ckpt.pt'))
    parser.add_argument('--train-lmdb', type=Path, help='Existing healthy-slice LMDB; otherwise prepare one training subject')
    parser.add_argument('--train-steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--inference-steps', type=int, default=3, help='Use 125 for the complete baseline window')
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are refused')
    parser.add_argument('--save-checkpoint', action='store_true', help='Exercise training save path, including its expensive 1000-step image sampling')
    args = parser.parse_args()
    if args.train_steps < 2 or args.batch_size < 1 or not 1 <= args.inference_steps <= 125:
        parser.error('Use train-steps >= 2, batch-size >= 1 and inference-steps in 1..125')

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from accelerate import Accelerator
    from andi.data.loading import MRI_Volume
    from andi.diffusion.process import Diffusion
    from andi.evaluation.evaluate import score
    from andi.evaluation.postprocessing import gmean
    from andi.models.unet import UNet

    root = Path(__file__).resolve().parent
    # Resolve user paths before changing the working directory.
    data = args.data.resolve()
    checkpoint = args.checkpoint.resolve()
    lmdb_path = args.train_lmdb.resolve() if args.train_lmdb else None
    output = (args.output or root / 'results' / f'smoke-{uuid4().hex[:10]}').resolve()
    if args.stage in ('both', 'train'):
        try:
            import wandb
            from andi.training.train import train
        except ModuleNotFoundError as exc:
            parser.error(f'Training dependency missing: {exc}. Use the project environment, or --stage infer.')
    if not data.is_dir() or (args.stage != 'train' and not checkpoint.is_file()):
        parser.error('Data directory or inference checkpoint does not exist')
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    torch.set_num_threads(4)
    torch.manual_seed(73)
    random.seed(73)
    np.random.seed(73)
    device = Accelerator().device
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run this smoke script with one process')
    summary = {'device': str(device), 'output': str(output), 'training_completed': False}
    print(f'Device: {device}; outputs: {output}', flush=True)

    if args.stage in ('both', 'infer'):
        conf = yaml.safe_load((root / 'conf/eval.yml').read_text())
        split = pd.read_csv(root / 'splits/BraTS21/scans_val.csv').iloc[:1]
        split.to_csv(output / 'validation.csv', index=False)
        conf.update(dataset_path=str(data), model=str(checkpoint),
                    path_to_csv=str(output / 'validation.csv'), workers=0, batch_size=1,
                    output=str(output / 'validation.csv.metrics.csv'),
                    output_mf=str(output / 'validation_mf5.csv'), stop=75 + args.inference_steps)
        (output / 'inference.yml').write_text(yaml.safe_dump(conf))
        model = UNet(device=device).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        diffusion = Diffusion(conf['noise_steps'], conf['size'], conf['beta_start'], conf['beta_end'], device)
        volume, label = next(iter(MRI_Volume(conf)))
        slices = (volume[0] * 2 - 1).permute(3, 0, 1, 2)
        maps = []
        for batch in slices.split(args.batch_size):
            residuals = diffusion.normative_diffusion(
                model, batch.to(device), conf['start'], conf['stop'], conf['pyramid'], conf['discount']
            ).cpu()
            aggregate = gmean(residuals, dim=1) if conf['gmean'] else residuals.mean(dim=1)
            maps.append(aggregate.amax(dim=1) if conf['max'] else aggregate.mean(dim=1))
        anomaly = torch.cat(maps).permute(1, 2, 0).unsqueeze(0).contiguous()
        if not torch.isfinite(anomaly).all():
            raise RuntimeError('Inference produced nonfinite scores')
        cache = output / 'maps'
        cache.mkdir()
        torch.save((anomaly, label.bool().cpu()), cache / '0000.pt')
        (cache / 'complete').touch()
        score(conf, cache)
        summary.update(validation_subject=str(split.iloc[0, 0]), inference_steps=args.inference_steps,
                       inference_completed=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        del model, diffusion, residuals, maps
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    if args.stage in ('both', 'train'):
        import lmdb
        if lmdb_path is None:
            import nibabel as nib
            from torchvision import transforms
            from split_healthy import normalize_volume

            split = pd.read_csv(root / 'splits/BraTS21/scans_train.csv').iloc[:1]
            subject = str(split.iloc[0, 0])
            split.to_csv(output / 'training.csv', index=False)
            directory = data / subject
            mask = np.asarray(nib.load(directory / f'{subject}_seg.nii.gz').dataobj, dtype=int)
            images = torch.stack([
                torch.from_numpy(np.asarray(nib.load(directory / f'{subject}_{modality}.nii.gz').dataobj,
                                            dtype=float))
                for modality in ('flair', 't1', 't1ce', 't2')
            ]).float()
            images = normalize_volume(images)
            selected = [i for i in range(images.shape[-1])
                        if torch.count_nonzero(images[0, :, :, i]) and not (mask[:, :, i] >= 1).any()]
            selected = selected[:args.train_steps * args.batch_size]
            if len(selected) < args.train_steps * args.batch_size:
                raise ValueError('Not enough healthy slices; use fewer steps or --train-lmdb')
            # Same preprocessing as split_healthy.py, but bounded to the smoke subset.
            slices = transforms.Resize(128, antialias=True)(images.permute(3, 0, 1, 2)[selected]).numpy()
            lmdb_path = output / 'prepared-lmdb'
            with lmdb.open(str(lmdb_path), map_size=max(64 * 1024**2, slices.nbytes * 4)) as prepared:
                with prepared.begin(write=True) as write:
                    for i, image in enumerate(slices):
                        write.put(f'{i:08}'.encode(), pickle.dumps(image))
            pd.DataFrame({'subject': subject, 'slice': selected}).to_csv(output / 'healthy.csv', index=False)
            summary['training_subject'] = subject
            del images, slices, mask
        count = args.train_steps * args.batch_size
        tiny = output / 'tiny-lmdb'
        with lmdb.open(str(lmdb_path), readonly=True, lock=False) as source:
            with source.begin() as read:
                if read.stat()['entries'] < count:
                    raise ValueError(f'Need {count} healthy slices; use fewer steps or a larger --train-lmdb')
                records = [read.get(f'{i:08}'.encode()) for i in range(count)]
        if any(record is None for record in records):
            raise ValueError('Expected contiguous eight-digit LMDB keys')
        with lmdb.open(str(tiny), map_size=max(64 * 1024**2, sum(map(len, records)) * 2)) as target:
            with target.begin(write=True) as write:
                for i, record in enumerate(records):
                    if not np.isfinite(pickle.loads(record)).all():
                        raise ValueError('Nonfinite training input')
                    write.put(f'{i:08}'.encode(), record)
        conf = yaml.safe_load((root / 'conf/train.yml').read_text())
        conf.update(run_name='smoke', dataset_path=str(tiny), epochs=1, workers=0,
                    batch_size=args.batch_size, train_continue=False, warmup_steps=1.5 / args.train_steps,
                    start_ckpt=-1 if args.save_checkpoint else 1, save_ckpt=1, num_images=1)
        (output / 'training.yml').write_text(yaml.safe_dump(conf))
        os.chdir(output)  # Keep model/image outputs out of production run directories.
        with wandb.init(project='andi-smoke', mode='offline', dir=str(output), config=conf):
            train(conf)
        if args.save_checkpoint:
            model = UNet(device='cpu')
            for name in ('0_ckpt.pt', '0_ema_ckpt.pt'):
                model.load_state_dict(torch.load(output / 'models/smoke' / name,
                                                map_location='cpu', weights_only=True), strict=True)
        summary.update(training_completed=True, training_steps=args.train_steps,
                       checkpoint_roundtrip=args.save_checkpoint, wandb_mode='offline')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Smoke run complete: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
