import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- your dataset imports ---
from galaxea_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import GalaxeaDatasetKeypointsJoints
from human_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import HumanDatasetKeypointsJoints

# ===================== dataset setup =====================
robot_dir1 = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"
human_dir1 = "/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox"
human_dir2 = "/iris/projects/humanoid/hamer/keypoint_human_data_red_outbox"
human_dir3 = "/iris/projects/humanoid/hamer/keypoint_human_data_wood_inbox"

robot_chunksize, robot_stride = 45, 2
human_chunksize, human_stride = 45, 1

apply_data_aug = True
normalize = True

datasets = {
    "robot": GalaxeaDatasetKeypointsJoints(
        dataset_dir=robot_dir1,
        chunk_size=robot_chunksize,
        stride=robot_stride,
        apply_data_aug=apply_data_aug,
        normalize=normalize,
        compute_keypoints=True,
        overlay_keypoints=False
    ),
    "human_inbox": HumanDatasetKeypointsJoints(
        dataset_dir=human_dir1,
        chunk_size=human_chunksize,
        stride=human_stride,
        apply_data_aug=apply_data_aug,
        normalize=normalize
    ),
    "human_outbox": HumanDatasetKeypointsJoints(
        dataset_dir=human_dir2,
        chunk_size=human_chunksize,
        stride=human_stride,
        apply_data_aug=apply_data_aug,
        normalize=normalize
    ),
    "human_wood": HumanDatasetKeypointsJoints(
        dataset_dir=human_dir3,
        chunk_size=human_chunksize,
        stride=human_stride,
        apply_data_aug=apply_data_aug,
        normalize=normalize
    ),
}

# ===================== helper =====================
def accumulate_actions(ds, max_batches=10, batch_size=8, skip_first_step=False):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    all_actions = []
    for i, batch in enumerate(tqdm(loader, desc=f"Collecting {ds.dataset_dir}")):
        # batch = (image_data, qpos_data, action_data, is_pad)
        action_data = batch[2]   # (B, T, 44)
        is_pad = batch[3]        # (B, T)

        B, T, D = action_data.shape

        # Flatten
        actions_flat = action_data.reshape(B*T, D).cpu().numpy()
        mask_flat = (~is_pad).reshape(B*T).cpu().numpy()

        # Keep only non-padded
        actions_valid = actions_flat[mask_flat]

        # Optionally skip first step of each trajectory (remove always-zero entries)
        if skip_first_step:
            actions_valid = actions_valid.reshape(-1, T-1, D) if (T > 1) else actions_valid
            # Flatten again (B*(T-1), D)
            if actions_valid.ndim == 3:
                actions_valid = actions_valid.reshape(-1, D)

        all_actions.append(actions_valid)

        if i >= max_batches:
            break

    return np.concatenate(all_actions, axis=0)  # (N, 44)
def plot_histograms(actions, out_prefix):
    os.makedirs("histograms", exist_ok=True)
    D = actions.shape[1]
    for d in range(D):
        plt.figure()
        plt.hist(actions[:, d], bins=100, color="blue", alpha=0.7)
        plt.title(f"Dimension {d}")
        plt.savefig(f"histograms/{out_prefix}_dim{d:02d}.png")
        plt.close()

# ===================== run =====================
for name, ds in datasets.items():
    print(f"Processing {name} ...")
    actions = accumulate_actions(ds, max_batches=50, batch_size=8)
    plot_histograms(actions, out_prefix=name)
    print(f"Saved histograms for {name}")
