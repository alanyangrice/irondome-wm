"""
Load `.npz` rollout files (full 512×256 RGB stored as uint8) and feed the VAE / MDN-RNN
at paper-scale resolution by resizing on the fly (bilinear), without changing the simulator.
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


class RolloutFrameDataset(Dataset):
    """
    Random-access frames across all episodes for VAE training.

    Uses memory-mapped reads so episodes are not fully loaded into RAM.
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

        self._lengths: List[int] = []
        self._offsets: List[int] = []
        cumulative = 0
        for path in self.paths:
            with np.load(path, mmap_mode="r") as data:
                t = int(data["obs"].shape[0])
            self._offsets.append(cumulative)
            self._lengths.append(t)
            cumulative += t
        self._total_steps = cumulative

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
        path, t = self._locate(index)
        with np.load(path, mmap_mode="r") as data:
            obs = np.asarray(data["obs"][t], dtype=np.uint8)
        return preprocess_obs_for_vae(
            obs, self.vae_input_height, self.vae_input_width, self.obs_scale, device=None
        )


class RolloutSequenceDataset(Dataset):
    """
    One item = full episode, for MDN-RNN (+ optional reward head) training.

    Returns dict with float tensors at VAE resolution for `obs`, raw-length vectors for
    `action`, `reward`, `done`.
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.paths[index]
        with np.load(path, mmap_mode="r") as data:
            obs = np.asarray(data["obs"], dtype=np.uint8)
            action = torch.from_numpy(np.asarray(data["action"], dtype=np.float32))
            reward = torch.from_numpy(np.asarray(data["reward"], dtype=np.float32))
            done = torch.from_numpy(np.asarray(data["done"], dtype=np.bool))

        frames_chw = torch.from_numpy(obs).permute(0, 3, 1, 2).contiguous().float() * self.obs_scale
        obs_small = downsample_obs_chw(frames_chw, self.vae_input_height, self.vae_input_width)

        return {
            "obs": obs_small,
            "action": action,
            "reward": reward,
            "done": done,
        }


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
