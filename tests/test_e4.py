"""Focused invariants for E4 objective, replay, and checkpoint safety."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from andi.diffusion.process import Diffusion
from andi.training.e4 import Trainer, intensity_transform, objective, seed_all
from andi.training.e4_checkpoints import CheckpointStore, atomic_save


class TinyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, images, timesteps):
        self.calls.append((images.detach().clone(), timesteps.detach().clone()))
        return images * self.scale


class E4Tests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "conf/train.yml").read_text())
        config.update(yaml.safe_load((root / "conf/e4.yml").read_text()))
        config.update(size=16, batch_size=2, epochs=2, warmup_steps=0.25)
        return config

    def test_gamma_background_and_seed(self):
        images = torch.rand(2, 4, 16, 16)
        images[:, :, :3] = 0
        generator = torch.Generator().manual_seed(8)
        saved = generator.get_state()
        first = intensity_transform(images, 0.15, generator)
        generator.set_state(saved)
        second = intensity_transform(images, 0.15, generator)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first == 0, images == 0))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(images, intensity_transform(images, 0, generator)))

    def test_shared_noise_and_paired_loss(self):
        config = self.config()
        diffusion = Diffusion(1000, 16, 0.0001, 0.02, "cpu")
        model = TinyModel()
        images = torch.rand(2, 4, 16, 16)
        rng = torch.Generator().manual_seed(4)
        before = rng.get_state()
        loss, terms = objective(model, diffusion, images, config, rng)
        original, t = model.calls[0]
        augmented, other_t = model.calls[1]
        self.assertTrue(torch.equal(t, other_t))
        alpha = diffusion.alpha_hat[t][:, None, None, None]
        rng.set_state(before)
        transformed = intensity_transform(images, 0.15, rng)
        noise = (original - alpha.sqrt() * (2 * images - 1)) / (1 - alpha).sqrt()
        noise_aug = (augmented - alpha.sqrt() * (2 * transformed - 1)) / (1 - alpha).sqrt()
        torch.testing.assert_close(noise, noise_aug)
        mse = torch.nn.functional.mse_loss
        expected = 0.5 * (mse(original * model.scale, noise) + mse(augmented * model.scale, noise))
        torch.testing.assert_close(terms["ddpm"], expected)
        torch.testing.assert_close(loss, expected + config["consistency_lambda"] * terms["consistency"])
        loss.backward()
        self.assertIsNotNone(model.scale.grad)

    def test_baseline_matches_legacy_draws_and_loss(self):
        config = self.config() | {"variant": "baseline", "consistency_lambda": 0.}
        diffusion = Diffusion(1000, 16, 0.0001, 0.02, "cpu")
        images = torch.rand(2, 4, 16, 16)
        model = TinyModel()
        seed_all(9)
        t = diffusion.sample_timesteps(2)
        noisy, noise = diffusion.noise_images(images * 2 - 1, t, pyramid=True, discount=0.8)
        expected = torch.nn.functional.mse_loss(model(noisy, t), noise)
        model.calls.clear()
        seed_all(9)
        actual, _ = objective(model, diffusion, images, config, torch.Generator())
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(len(model.calls), 1)

    def test_augmentation_control_has_zero_consistency_weight(self):
        config = self.config() | {"variant": "augmentation", "consistency_lambda": 0.}
        model = TinyModel()
        loss, terms = objective(
            model, Diffusion(1000, 16, 0.0001, 0.02, "cpu"),
            torch.rand(2, 4, 16, 16), config, torch.Generator()
        )
        self.assertTrue(torch.equal(loss, terms["ddpm"]))
        self.assertEqual(len(model.calls), 2)

    @patch("andi.training.e4.UNet", TinyModel)
    def test_mid_epoch_and_epoch_resume(self):
        from kaggle.e4_one_cell import assert_same

        config = self.config()
        data = torch.rand(6, 4, 16, 16)
        trainer = Trainer(config, data, data[:2], "cpu")
        trainer.step()
        snapshot = copy.deepcopy(trainer.snapshot())
        trainer.step()
        expected = copy.deepcopy(trainer.snapshot())
        resumed = Trainer(config, data, data[:2], "cpu")
        resumed.restore(snapshot)
        resumed.step()
        assert_same(expected, resumed.snapshot())
        # Validate without consuming the training RNG stream.
        rng_before = resumed.snapshot()["rng"]
        resumed.validate()
        assert_same(rng_before, resumed.snapshot()["rng"])
        resumed.step()
        resumed.finish_epoch()
        snapshot = copy.deepcopy(resumed.snapshot())
        resumed.step()
        expected = copy.deepcopy(resumed.snapshot())
        replay = Trainer(config, data, data[:2], "cpu")
        replay.restore(snapshot)
        replay.step()
        assert_same(expected, replay.snapshot())
        with self.assertRaises(ValueError):
            replay.restore(snapshot | {"config": config | {"seed": 99}})

    @patch("andi.training.e4.UNet", TinyModel)
    def test_resume_from_last_batch_before_validation(self):
        config = self.config() | {"epochs": 1}
        data = torch.rand(6, 4, 16, 16)
        original = Trainer(config, data, data[:2], "cpu")
        for _ in range(original.batches):
            original.step()
        resumed = Trainer(config, data, data[:2], "cpu")
        resumed.restore(original.snapshot())
        with TemporaryDirectory() as work:
            store = CheckpointStore(Path(work) / "checkpoints")
            with patch.object(resumed, "step", side_effect=AssertionError("unexpected step")):
                resumed.fit(store, None, save_steps=1, save_seconds=60)
            best = torch.load(store.directory / "best.pt", weights_only=False)
            self.assertEqual(best["global_step"], original.batches)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(best["batch"], 0)

    def test_atomic_failure_preserves_existing_checkpoint(self):
        with TemporaryDirectory() as work:
            path = Path(work) / "checkpoint.pt"
            atomic_save({"step": 1}, path)
            with patch("torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save({"step": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 1)
            self.assertEqual(len(list(Path(work).iterdir())), 1)

    def test_failed_upload_prevents_retention_cleanup(self):
        class Remote:
            fail = False

            def upload(self, *args):
                if self.fail:
                    raise RuntimeError("network failed")

        with TemporaryDirectory() as work:
            remote = Remote()
            store = CheckpointStore(Path(work) / "checkpoints", remote)
            first, _ = store.save({"global_step": 1})
            second, _ = store.save({"global_step": 2})
            remote.fail = True
            with self.assertRaises(RuntimeError):
                store.save({"global_step": 3})
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(len(list(store.directory.glob("*.pt"))), 3)


if __name__ == "__main__":
    unittest.main()
