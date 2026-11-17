"""Atomic E4 snapshots and verified W&B checkpoint versions."""

import hashlib
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(state, path):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


class WandbCheckpoints:
    def __init__(self, run, collection):
        self.run = run
        self.collection = collection

    @staticmethod
    def download(ref, root):
        import wandb

        artifact = wandb.Api(timeout=60).artifact(ref, type="model")
        checkpoint = Path(artifact.download(root=root, skip_cache=True)) / "checkpoint.pt"
        if sha256(checkpoint) != artifact.metadata.get("sha256"):
            raise ValueError(f"SHA-256 verification failed for {ref}")
        return artifact, checkpoint

    def recover(self, ref):
        if not re.fullmatch(r"[^/]+/[^/]+/[^/:]+:(?:verified|v[0-9]+|best-step-[0-9]+)", ref):
            raise ValueError("Use entity/project/collection with :verified, :vN or :best-step-N")
        with TemporaryDirectory() as work:
            artifact, checkpoint = self.download(ref, work)
            if f"verified-{artifact.version}" not in artifact.aliases:
                raise ValueError(f"{ref} has no E4 verification marker")
            # E4 snapshots contain Python and NumPy RNG state. Only restore trusted runs.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return f"{ref.rsplit(':', 1)[0]}:{artifact.version}", state

    def upload(self, path, step, best, final):
        import wandb

        digest = sha256(path)
        for attempt in range(3):
            try:
                artifact = wandb.Artifact(
                    self.collection, type="model", metadata={"sha256": digest, "global_step": step}
                )
                artifact.add_file(str(path), name="checkpoint.pt", policy="immutable", skip_cache=True)
                self.run.log_artifact(artifact, aliases=["pending"]).wait(timeout=600)
                ref = artifact.qualified_name
                with TemporaryDirectory(dir=path.parent) as work:
                    remote, _ = self.download(ref, work)
                if remote.metadata["sha256"] != digest:
                    raise ValueError(f"Unexpected contents for {ref}")
                # A step-specific alias marks this version permanently. The other
                # aliases move only after the download proves this version is readable.
                marker = f"verified-{remote.version}"
                aliases = ["verified", marker]
                for alias, enabled in (("best", best), ("final", final)):
                    if enabled or alias in remote.aliases:
                        aliases.append(alias)
                selection_alias = f"best-step-{step}"
                if best or selection_alias in remote.aliases:
                    aliases.append(selection_alias)
                remote.aliases = aliases
                remote.save()
                api = wandb.Api(timeout=60)
                verified = api.artifact(f"{ref.rsplit(':', 1)[0]}:verified", type="model")
                if verified.version != remote.version or marker not in verified.aliases:
                    raise ValueError(f"Verification alias did not resolve to {ref}")
                print(f"Verified remote checkpoint at {ref} (step {step})", flush=True)
                return ref
            except Exception as error:
                print(f"Checkpoint upload attempt {attempt + 1}/3 failed: {error}", flush=True)
                if attempt == 2:
                    raise RuntimeError(
                        f"Checkpoint retained at {path}; the prior verified remote version is recoverable"
                    ) from error
                time.sleep(5 * (attempt + 1))


class CheckpointStore:
    def __init__(self, directory, remote=None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.remote = remote
        self.version = 0
        self.retained = []

    def save(self, state, best=False, final=False):
        self.version += 1
        step = state["global_step"]
        path = self.directory / f"step-{step:09d}-{self.version:06d}.pt"
        atomic_save(state, path)
        print(f"Saved local checkpoint at {path}", flush=True)
        ref = self.remote.upload(path, step, best, final) if self.remote else None
        if best:
            link = self.directory / ".best.pt"
            link.hardlink_to(path)
            link.replace(self.directory / "best.pt")
        self.retained.append(path)
        if len(self.retained) > 2:
            self.retained.pop(0).unlink()
        return path, ref
