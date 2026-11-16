"""Single-device E4 trainer with deterministic batch cursors and complete snapshots."""

import copy
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from andi.diffusion.process import Diffusion
from andi.models.unet import EMA, LRWarmupCosineDecay, UNet


def intensity_transform(images, magnitude, generator):
    """Per-slice/modality gamma. Preserve each modality's zero background exactly."""
    gamma = 1 + magnitude * (
        2 * torch.rand((*images.shape[:2], 1, 1), generator=generator) - 1
    )
    positive = images.clamp_min(torch.finfo(images.dtype).tiny)
    transformed = positive.pow(gamma.to(images.device))
    return torch.where(images > 0, transformed, images)


def objective(model, diffusion, images, config, augmentation_rng):
    original = images * 2 - 1
    timesteps = diffusion.sample_timesteps(images.shape[0]).to(images.device)
    noisy, noise = diffusion.noise_images(
        original, timesteps, pyramid=config["pyramid"],
        simplex=config["simplex"], discount=config["discount"],
    )
    prediction = model(noisy, timesteps)
    original_mse = F.mse_loss(prediction, noise)
    if config["variant"] == "baseline":
        zero = original_mse.new_zeros(())
        return original_mse, {"ddpm": original_mse, "consistency": zero}

    augmented = intensity_transform(
        images, config["augmentation_magnitude"], augmentation_rng
    ) * 2 - 1
    alpha = diffusion.alpha_hat[timesteps][:, None, None, None]
    noisy_augmented = alpha.sqrt() * augmented + (1 - alpha).sqrt() * noise
    prediction_augmented = model(noisy_augmented, timesteps)
    paired = 0.5 * (original_mse + F.mse_loss(prediction_augmented, noise))
    consistency = F.mse_loss(prediction, prediction_augmented)
    loss = paired + config["consistency_lambda"] * consistency
    return loss, {"ddpm": paired, "consistency": consistency}


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config, training, validation, device, initial_weights=None):
        self.config = copy.deepcopy(config)
        self.training = training
        self.validation = validation
        self.device = torch.device(device)
        seed_all(config["seed"])
        self.model = UNet(
            c_in=config["channels"], c_out=config["channels"],
            img_size=config["size"], device=self.device,
        ).to(self.device)
        if initial_weights is not None:
            self.model.load_state_dict(initial_weights, strict=True)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema = EMA(config["ema_decay"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1)
        self.batches = math.ceil(len(training) / config["batch_size"])
        self.total_steps = self.batches * config["epochs"]
        warmup = max(1, int(config["warmup_steps"] * self.total_steps))
        if warmup >= self.total_steps:
            raise ValueError("The training schedule requires at least two steps and a shorter warmup")
        self.scheduler = LRWarmupCosineDecay(
            self.optimizer, warmup, self.total_steps, config["start_lr"], config["target_lr"]
        )
        self.diffusion = Diffusion(
            config["noise_steps"], config["size"], config["beta_start"],
            config["beta_end"], self.device,
        )
        self.augmentation_rng = torch.Generator().manual_seed(config["augmentation_seed"])
        self.epoch = 0
        self.batch = 0
        self.global_step = 0
        self.best_validation = float("inf")
        self.best_step = None
        self.order_epoch = None
        self.order = None

    def snapshot(self):
        return {
            "format": "andi-e4-v1", "config": self.config,
            "model": self.model.state_dict(), "ema_model": self.ema_model.state_dict(),
            "ema_step": self.ema.step, "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "global_step": self.global_step,
            "epoch": self.epoch, "batch": self.batch,
            "best_validation": self.best_validation, "best_step": self.best_step,
            "rng": capture_rng(),
            "augmentation_rng": self.augmentation_rng.get_state(),
        }

    def restore(self, state):
        if state.get("format") != "andi-e4-v1":
            raise ValueError("Exact resume requires an E4 full-state checkpoint")
        if state["config"] != self.config:
            raise ValueError("Exact resume configuration/provenance differs from checkpoint")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema_model.load_state_dict(state["ema_model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.ema.step = state["ema_step"]
        self.epoch, self.batch = state["epoch"], state["batch"]
        self.global_step = state["global_step"]
        self.best_validation = state["best_validation"]
        self.best_step = state["best_step"]
        if not (0 <= self.epoch <= self.config["epochs"] and 0 <= self.batch <= self.batches
                and self.global_step == self.epoch * self.batches + self.batch <= self.total_steps):
            raise ValueError("Invalid checkpoint batch cursor")
        self.augmentation_rng.set_state(state["augmentation_rng"])
        restore_rng(state["rng"])

    def step(self):
        if self.order_epoch != self.epoch:
            order_rng = torch.Generator().manual_seed(self.config["shuffle_seed"] + self.epoch)
            self.order = torch.randperm(len(self.training), generator=order_rng)
            self.order_epoch = self.epoch
        start = self.batch * self.config["batch_size"]
        indices = self.order[start:start + self.config["batch_size"]]
        images = torch.stack([self.training[int(index)] for index in indices]).to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, components = objective(
            self.model, self.diffusion, images, self.config, self.augmentation_rng
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema.step_ema(self.ema_model, self.model)
        self.global_step += 1
        self.batch += 1
        return {"loss": loss.item(), **{key: value.item() for key, value in components.items()},
                "lr": self.scheduler.get_last_lr()[0]}

    @torch.no_grad()
    def validate(self):
        """Fixed-noise healthy validation MSE for all variants, with isolated RNG."""
        rng = capture_rng()
        try:
            seed_all(self.config["validation_seed"])
            total = 0.
            for start in range(0, len(self.validation), self.config["batch_size"]):
                images = torch.stack([
                    self.validation[index]
                    for index in range(start, min(start + self.config["batch_size"], len(self.validation)))
                ]).to(self.device) * 2 - 1
                t = self.diffusion.sample_timesteps(len(images)).to(self.device)
                noisy, noise = self.diffusion.noise_images(
                    images, t, pyramid=True, discount=self.config["discount"]
                )
                total += F.mse_loss(self.ema_model(noisy, t), noise).item() * len(images)
            score = total / len(self.validation)
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation loss")
            return score
        finally:
            restore_rng(rng)

    def finish_epoch(self):
        score = self.validate()
        best = score < self.best_validation
        if best:
            self.best_validation = score
            self.best_step = self.global_step
        self.epoch += 1
        self.batch = 0
        return score, best

    def save(self, store, *, best=False, final=False):
        state = self.snapshot()
        try:
            return store.save(state, best=best, final=final)
        finally:
            restore_rng(state["rng"])

    def report(self, run, metrics):
        print(f"step {self.global_step}/{self.total_steps} {metrics}", flush=True)
        if run:
            rng = capture_rng()
            try:
                run.log(metrics | {"global_step": self.global_step})
            finally:
                restore_rng(rng)

    def fit(self, store, run, *, save_steps, save_seconds):
        # Establish a recoverable remote checkpoint before any optimizer work.
        if self.global_step == 0:
            self.save(store)
        last_save = time.monotonic()
        while self.epoch < self.config["epochs"]:
            if self.batch < self.batches:
                start = time.monotonic()
                metrics = self.step()
                self.report(run, metrics | {"step_seconds": time.monotonic() - start})
                if (self.global_step % save_steps == 0
                        or time.monotonic() - last_save >= save_seconds):
                    self.save(store)
                    last_save = time.monotonic()
            if self.batch == self.batches:
                score, best = self.finish_epoch()
                self.report(run, {"validation_mse": score, "epoch": self.epoch})
                self.save(store, best=best, final=self.epoch == self.config["epochs"])
                last_save = time.monotonic()
