"""
Load `.npz` rollout files (full 512×256 RGB stored as uint8) and feed the VAE / MDN-RNN
at paper-scale resolution by resizing (bilinear), without changing the simulator.

Why we pre-decompress at init: `np.savez_compressed` writes zlib-compressed `.npz` archives,
and `np.load(..., mmap_mode="r")` silently falls back to non-mmap for compressed archives.
That meant the old per-frame `__getitem__` re-decompressed an entire episode for every
single frame fetched (catastrophic with shuffled mini-batches). We now decompress each
episode exactly once at init, downsample to the VAE resolution, and store as packed
uint8 in RAM. At 128×64 RGB and ~30k frames that's well under 1 GB.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def list_rollout_files(data_dir: Union[str, Path], pattern: str = "*.npz") -> List[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob(pattern))


def numpy_obs_to_chw_float(
    obs: np.ndarray,
    scale: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """uint8 [H, W, 3] -> float32 [3, H, W]."""
    t = torch.from_numpy(obs).permute(2, 0, 1).contiguous().float() * scale
    if device is not None:
        t = t.to(device)
    return t


def downsample_obs_chw(
    obs_chw: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """
    Bilinear resize. `obs_chw` is [C, H, W] or [B, C, H, W], float32.
    """
    if obs_chw.dim() == 3:
        x = obs_chw.unsqueeze(0)
        return F.interpolate(x, size=(target_height, target_width), mode="bilinear", align_corners=False).squeeze(
            0
        )
    return F.interpolate(obs_chw, size=(target_height, target_width), mode="bilinear", align_corners=False)


def preprocess_obs_for_vae(
    obs_uint8_hwc: np.ndarray,
    target_height: int,
    target_width: int,
    scale: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Single frame uint8 [H, W, 3] -> float32 [3, h, w] at VAE resolution."""
    t = numpy_obs_to_chw_float(obs_uint8_hwc, scale, device=device)
    return downsample_obs_chw(t, target_height, target_width)


def _downsample_episode_uint8(
    obs_thwc_uint8: np.ndarray,
    target_height: int,
    target_width: int,
    chunk: int = 256,
) -> np.ndarray:
    """[T, H, W, 3] uint8 -> [T, 3, h, w] uint8 via bilinear resize.

    Chunks through the time axis on CPU so we never materialize the whole episode as float32.
    """
    t_total = obs_thwc_uint8.shape[0]
    out = np.empty((t_total, 3, target_height, target_width), dtype=np.uint8)
    for start in range(0, t_total, chunk):
        end = min(start + chunk, t_total)
        chunk_thwc = obs_thwc_uint8[start:end]
        chunk_chw = (
            torch.from_numpy(np.ascontiguousarray(chunk_thwc)).permute(0, 3, 1, 2).contiguous().float()
        )
        small = F.interpolate(chunk_chw, size=(target_height, target_width), mode="bilinear", align_corners=False)
        out[start:end] = small.round().clamp_(0, 255).to(torch.uint8).numpy()
    return out


class RolloutFrameDataset(Dataset):
    """
    Random-access frames across all episodes for VAE training.

    Decompresses every `.npz` once at init, downsamples to the VAE resolution, and keeps
    the result as packed uint8 in RAM. `__getitem__` is then a cheap index + float cast.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        vae_input_height: int,
        vae_input_width: int,
        obs_scale: float = 1.0 / 255.0,
        pattern: str = "*.npz",
    ) -> None:
        self.paths = list_rollout_files(data_dir, pattern)
        if not self.paths:
            raise FileNotFoundError(f"No rollout files matching {pattern!r} under {data_dir}")

        self.vae_input_height = vae_input_height
        self.vae_input_width = vae_input_width
        self.obs_scale = obs_scale

        small_chunks: List[np.ndarray] = []
        self._lengths: List[int] = []
        self._offsets: List[int] = []
        cumulative = 0
        for path in self.paths:
            with np.load(path) as data:
                obs = np.asarray(data["obs"], dtype=np.uint8)
            small = _downsample_episode_uint8(obs, vae_input_height, vae_input_width)
            small_chunks.append(small)
            t = int(small.shape[0])
            self._offsets.append(cumulative)
            self._lengths.append(t)
            cumulative += t
        self._total_steps = cumulative
        self._frames = np.concatenate(small_chunks, axis=0) if small_chunks else np.empty(
            (0, 3, vae_input_height, vae_input_width), dtype=np.uint8
        )

    def __len__(self) -> int:
        return self._total_steps

    def _locate(self, index: int) -> Tuple[Path, int]:
        if index < 0 or index >= self._total_steps:
            raise IndexError(index)
        ends = [o + ln for o, ln in zip(self._offsets, self._lengths)]
        ep = bisect.bisect_right(ends, index)
        local = index - self._offsets[ep]
        return self.paths[ep], local

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self._total_steps:
            raise IndexError(index)
        frame_uint8 = self._frames[index]
        return torch.from_numpy(frame_uint8).float() * self.obs_scale


class RolloutSequenceDataset(Dataset):
    """
    One item = full episode, for MDN-RNN (+ reward head) training.

    Also pre-decompresses each `.npz` once at init and stores each episode's frames as
    packed uint8 at VAE resolution. Per-item conversion to float32 only happens on access.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        vae_input_height: int,
        vae_input_width: int,
        obs_scale: float = 1.0 / 255.0,
        pattern: str = "*.npz",
    ) -> None:
        self.paths = list_rollout_files(data_dir, pattern)
        if not self.paths:
            raise FileNotFoundError(f"No rollout files matching {pattern!r} under {data_dir}")

        self.vae_input_height = vae_input_height
        self.vae_input_width = vae_input_width
        self.obs_scale = obs_scale

        self._episodes: List[Dict[str, np.ndarray]] = []
        for path in self.paths:
            with np.load(path) as data:
                obs = np.asarray(data["obs"], dtype=np.uint8)
                action = np.asarray(data["action"], dtype=np.float32)
                reward = np.asarray(data["reward"], dtype=np.float32)
                done = np.asarray(data["done"], dtype=bool)
            small = _downsample_episode_uint8(obs, vae_input_height, vae_input_width)
            self._episodes.append({"obs": small, "action": action, "reward": reward, "done": done})

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ep = self._episodes[index]
        obs_small = torch.from_numpy(ep["obs"]).float() * self.obs_scale
        action = torch.from_numpy(ep["action"])
        reward = torch.from_numpy(ep["reward"])
        done = torch.from_numpy(ep["done"])
        return {"obs": obs_small, "action": action, "reward": reward, "done": done}


def collate_padded_episodes(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length episodes to the max length in the batch.

    Adds `mask` float tensor [B, T] with 1.0 on valid timesteps, 0.0 on padding,
    for masked losses on the MDN / reward head.
    """
    batch_size = len(batch)
    max_t = max(item["obs"].shape[0] for item in batch)
    c, h, w = batch[0]["obs"].shape[1:]
    action_dim = batch[0]["action"].shape[1]

    obs = torch.zeros(batch_size, max_t, c, h, w, dtype=batch[0]["obs"].dtype)
    action = torch.zeros(batch_size, max_t, action_dim, dtype=batch[0]["action"].dtype)
    reward = torch.zeros(batch_size, max_t, dtype=batch[0]["reward"].dtype)
    done = torch.zeros(batch_size, max_t, dtype=torch.bool)
    mask = torch.zeros(batch_size, max_t, dtype=torch.float32)

    for i, item in enumerate(batch):
        t = item["obs"].shape[0]
        obs[i, :t] = item["obs"]
        action[i, :t] = item["action"]
        reward[i, :t] = item["reward"]
        done[i, :t] = item["done"]
        mask[i, :t] = 1.0

    return {"obs": obs, "action": action, "reward": reward, "done": done, "mask": mask}
