"""
Training entry point for the UMI-migration pipeline (real charm-lab UMI repo).

Identical in shape to train.py, with two differences:
  1. sys.path points at HapticTeleopIL/Imitation Learning/universal_manipulation_interface
     (the REAL UMI repo) instead of Robomimic/dt_ag-main (Aiden's fork).
  2. config_path is training/config/umi/ so `--config-name=train_diffusion_unet_timm_kinova_umi`
     resolves its `defaults: - task: kinova_teleop` against config/umi/task/kinova_teleop.yaml
     (UMI schema, UmiDataset) — NOT our existing config/task/kinova_teleop.yaml
     (Aiden pipeline, untouched).

Run in the `umi` conda env, from training/:
    python train_umi.py --config-name=train_diffusion_unet_timm_kinova_umi \
        task.dataset_path=../data_collection/demo_data/Collection5/kinova_teleop_umi.zarr.zip

The dataset .zarr.zip is produced by convert_data_umi.py (NOT convert_data.py —
the two write incompatible schemas; see convert_data_umi.py's docstring).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# REAL UMI repo (charm-lab/HapticTeleopIL clone) — not Robomimic/dt_ag-main
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "..", "HapticTeleopIL",
                         "Imitation Learning", "universal_manipulation_interface")

if _UMI_ROOT not in sys.path:
    sys.path.insert(0, _UMI_ROOT)

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


@hydra.main(
    version_base=None,
    config_path=os.path.join(_THIS_DIR, "config", "umi"),
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
