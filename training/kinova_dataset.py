"""
KinovaImageDataset — flat UMI-style zarr dataset for diffusion policy training.

Reads the flat zarr format produced by convert_data.py:
  data/zed_front_rgb       (N, 224, 224, 3) uint8
  data/dji_wrist_rgb       (N, 224, 224, 3) uint8
  data/pose                (N, 10) float32
  data/action              (N, 10) float32
  data/piezense_pressure   (N, 2)  float32
  meta/episode_ends        (E,)    int64   cumulative end indices

Each __getitem__ returns:
  obs:
    zed_front_rgb:      (obs_horizon, 3, 224, 224)  float32  [0, 1]
    dji_wrist_rgb:      (obs_horizon, 3, 224, 224)  float32  [0, 1]
    pose:               (obs_horizon, 10)            float32
    piezense_pressure:  (obs_horizon, 2)             float32  Pa
  action:               (action_horizon, 10)         float32
"""

from __future__ import annotations

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from diffusion_policy.model.common.normalizer import LinearNormalizer


class KinovaImageDataset(Dataset):
    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        val_ratio: float = 0.05,
        seed: int = 42,
        preload: bool = False,
        _split: str = "train",
    ):
        self.zarr_path = zarr_path
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        store = zarr.open(zarr_path, mode="r")
        episode_ends   = store["meta/episode_ends"][:]                  # (E,) int64
        episode_starts = np.concatenate([[0], episode_ends[:-1]])       # (E,)

        # Gather valid episodes as (ep_idx, global_start, global_end)
        all_episodes = []
        for ep_idx, (s, e) in enumerate(zip(episode_starts.tolist(), episode_ends.tolist())):
            if int(e) - int(s) >= horizon:
                all_episodes.append((ep_idx, int(s), int(e)))

        # Reproducible train/val split by episode
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_episodes))
        n_val = max(1, int(len(all_episodes) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        if _split == "train":
            self.episodes = [ep for i, ep in enumerate(all_episodes) if i not in val_set]
        else:
            self.episodes = [ep for i, ep in enumerate(all_episodes) if i in val_set]

        # Build list of global t_start indices (one per sliding-window sample)
        self.samples: list[int] = []
        for _, ep_start, ep_end in self.episodes:
            T = ep_end - ep_start
            for t in range(T - horizon + 1):
                self.samples.append(ep_start + t)

        self._store = store

        # Optionally preload all data arrays into RAM
        self._preloaded: dict | None = None
        if preload:
            self._preloaded = {k: store["data"][k][:] for k in store["data"].array_keys()}

        self._val_ratio = val_ratio
        self._seed = seed

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        t_start = self.samples[idx]
        t_obs_end = t_start + self.obs_horizon
        t_act_end = t_start + self.action_horizon

        data = self._preloaded if self._preloaded is not None else self._store["data"]

        def load(key, s, e):
            return data[key][s:e]

        # Images: (obs_horizon, H, W, 3) uint8 -> (obs_horizon, 3, H, W) float [0,1]
        def load_img(key):
            imgs = load(key, t_start, t_obs_end)  # (obs_h, H, W, 3)
            return torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).float() / 255.0

        obs = {
            "zed_front_rgb":     load_img("zed_front_rgb"),
            "dji_wrist_rgb":     load_img("dji_wrist_rgb"),
            "pose":              torch.from_numpy(load("pose",              t_start, t_obs_end).copy()).float(),
            "piezense_pressure": torch.from_numpy(load("piezense_pressure", t_start, t_obs_end).copy()).float(),
        }

        action = torch.from_numpy(
            load("action", t_start, t_act_end).copy()
        ).float()

        return {"obs": obs, "action": action}

    # ── Diffusion policy workspace helpers ────────────────────────────────────

    def get_validation_dataset(self) -> "KinovaImageDataset":
        return KinovaImageDataset(
            shape_meta=self.shape_meta,
            zarr_path=self.zarr_path,
            horizon=self.horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            val_ratio=self._val_ratio,
            seed=self._seed,
            _split="val",
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Compute per-key linear normalizers from the split's episodes."""
        data_grp = self._store["data"]
        pose_all, action_all, piezense_all = [], [], []

        for _, ep_start, ep_end in self.episodes:
            pose_all.append(data_grp["pose"][ep_start:ep_end])
            action_all.append(data_grp["action"][ep_start:ep_end])
            piezense_all.append(data_grp["piezense_pressure"][ep_start:ep_end])

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "obs": {
                    "pose":              torch.from_numpy(np.concatenate(pose_all)),
                    "piezense_pressure": torch.from_numpy(np.concatenate(piezense_all)),
                },
                "action": torch.from_numpy(np.concatenate(action_all)),
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer
