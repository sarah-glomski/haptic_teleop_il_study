"""
Training entry point (charm-lab UMI pipeline, real universal_manipulation_interface).

Points sys.path at the real UMI repo (HapticTeleopIL/.../universal_manipulation_interface)
and config_path at training/config/, so `--config-name=train_diffusion_unet_timm_kinova`
resolves its `defaults: - task: kinova_teleop` against config/task/kinova_teleop.yaml
(UmiDataset, per-signal robot0_*/camera0_* schema).

Run in the `umi` conda env, from training/:
    python train.py --config-name=train_diffusion_unet_timm_kinova \
        task.dataset_path=../data_collection/demo_data/Collection5.5/kinova_teleop_umi.zarr.zip

The dataset .zarr.zip is produced by convert_data.py (UMI ReplayBuffer schema).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Real UMI repo (charm-lab/HapticTeleopIL clone)
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
    config_path=os.path.join(_THIS_DIR, "config"),
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
