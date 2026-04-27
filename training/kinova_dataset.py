"""
KinovaImageDataset — per-episode zarr dataset for diffusion policy training.

Reads the per-episode zarr format produced by convert_data.py and returns
sliding-window samples compatible with TrainDiffusionUnetImageWorkspace.

Each __getitem__ returns:
  obs:
    zed_front_rgb:      (obs_horizon, 3, 224, 224)  float32  [0, 1]
    rs_wrist_rgb:       (obs_horizon, 3, 224, 224)  float32  [0, 1]
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
        all_ep_names = sorted(store.group_keys())

        # Gather valid episodes (must have at least `horizon` frames)
        all_episodes = []
        for ep_name in all_ep_names:
            grp = store[ep_name]
            T = grp["pose"].shape[0]
            if T >= horizon:
                all_episodes.append((ep_name, T))

        # Reproducible train/val split
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(all_episodes))
        n_val = max(1, int(len(all_episodes) * val_ratio))
        val_set = set(idx[:n_val].tolist())

        if _split == "train":
            self.episodes = [ep for i, ep in enumerate(all_episodes) if i not in val_set]
        else:
            self.episodes = [ep for i, ep in enumerate(all_episodes) if i in val_set]

        # Build (ep_name, t_start) index for sliding windows
        self.samples: list[tuple[str, int]] = []
        for ep_name, T in self.episodes:
            for t in range(T - horizon + 1):
                self.samples.append((ep_name, t))

        self._store = store

        # Optionally preload all arrays into RAM
        self._preloaded: dict[str, dict] | None = None
        if preload:
            self._preloaded = {}
            for ep_name, _ in self.episodes:
                grp = store[ep_name]
                self._preloaded[ep_name] = {k: grp[k][:] for k in grp.array_keys()}

        # Stash for get_validation_dataset
        self._val_ratio = val_ratio
        self._seed = seed

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_name, t_start = self.samples[idx]

        if self._preloaded is not None:
            grp = self._preloaded[ep_name]
            def load(key, s, e): return grp[key][s:e]
        else:
            grp = self._store[ep_name]
            def load(key, s, e): return grp[key][s:e]

        t_obs_start = t_start
        t_obs_end   = t_start + self.obs_horizon

        # Images: (obs_horizon, H, W, 3) uint8 -> (obs_horizon, 3, H, W) float [0,1]
        def load_img(key):
            imgs = load(key, t_obs_start, t_obs_end)  # (obs_h, H, W, 3)
            return torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).float() / 255.0

        obs = {
            "zed_front_rgb":     load_img("zed_front_rgb"),
            "rs_wrist_rgb":      load_img("rs_wrist_rgb"),
            "pose":              torch.from_numpy(load("pose", t_obs_start, t_obs_end).copy()).float(),
            "piezense_pressure": torch.from_numpy(load("piezense_pressure", t_obs_start, t_obs_end).copy()).float(),
        }

        action = torch.from_numpy(
            load("action", t_start, t_start + self.action_horizon).copy()
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
        """Compute per-key linear normalizers from full dataset statistics."""
        pose_all, action_all, piezense_all = [], [], []

        for ep_name, _ in self.episodes:
            grp = self._store[ep_name]
            pose_all.append(grp["pose"][:])
            action_all.append(grp["action"][:])
            piezense_all.append(grp["piezense_pressure"][:])

        pose_cat     = np.concatenate(pose_all,     axis=0)  # (N, 10)
        action_cat   = np.concatenate(action_all,   axis=0)  # (N, 10)
        piezense_cat = np.concatenate(piezense_all, axis=0)  # (N, 2)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "obs": {
                    "pose":              torch.from_numpy(pose_cat),
                    "piezense_pressure": torch.from_numpy(piezense_cat),
                },
                "action": torch.from_numpy(action_cat),
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer
