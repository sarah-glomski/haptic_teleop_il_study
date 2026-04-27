"""
Training entry point for diffusion policy on Kinova Gen3 teleop data.

Adapted from Robomimic/training/train.py for the HoloLens + Kinova Gen3 setup.
Adds the UMI codebase (dt_ag-main) and this training directory to sys.path,
registers Hydra resolvers, then delegates to TrainDiffusionUnetImageWorkspace.

Usage:
    python train.py --config-name=train_diffusion_unet_timm_kinova
    python train.py --config-name=train_diffusion_unet_timm_kinova training.debug=True
    python train.py --config-name=train_diffusion_unet_timm_kinova logging.mode=disabled
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# UMI / diffusion_policy codebase (shared with Robomimic)
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "..", "Robomimic", "dt_ag-main")
_UMI_DP   = os.path.join(_UMI_ROOT, "universal_manipulation_interface")

for p in [_UMI_DP, _UMI_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Make kinova_dataset importable by Hydra
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


@hydra.main(
    version_base=None,
    config_path=os.path.join(_THIS_DIR, "config"),
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
