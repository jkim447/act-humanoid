import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A

# TODO: make sure this is the correct path to norm_stats!
norm_stats = np.load("norm_stats_galaxea_delta.npz")

class GalaxeaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, chunk_size, apply_data_aug = True, normalize = True):
        super(GalaxeaDataset).__init__()
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.apply_data_aug = apply_data_aug
        self.img_height = 224
        self.img_width  = 224
        # hard-coded right-hand joint names
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]

        # collect demo folders: Demo1, Demo2, ..., DemoN
        def _demo_key(name: str):
            # handles "demo_12" or "Demo12"
            try:
                if name.startswith("demo_"):
                    return int(name.split("demo_")[1])
                if name.startswith("Demo"):
                    return int(name.replace("Demo", ""))
            except Exception:
                pass
            return float("inf")

        self.episode_dirs = sorted(
            [
                os.path.join(self.dataset_dir, d)
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
                and (d.startswith("demo_") or d.startswith("Demo"))
            ],
            key=lambda p: _demo_key(os.path.basename(p)),
        )

        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(height=self.img_height, width=self.img_width, scale=(0.95, 1.0), p=0.4),
                A.Rotate(limit=5, p=0.4),
                # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.55, hue=0.03, p=0.4),
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.9, hue=0.15, p=0.4),
                A.CoarseDropout(min_holes=18, max_holes=35, min_height=4, max_height=10,
                                min_width=4, max_width=10, p=0.4),
                A.Resize(height=self.img_height, width=self.img_width)
            ],
            additional_targets={"image_right": "image"}
        )

    def __len__(self):
        return len(self.episode_dirs)

    def _row_to_action(self, row):
        # Right wrist position (3)
        a = [row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]]

        # Right wrist orientation (quat → 6D)
        rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
        rR = R.from_quat(rq).as_matrix()
        a.extend(rR[:, :2].reshape(-1, order="F").tolist())  # 6D, col-major

        # Right hand command joints (20)
        a.extend([row[c] for c in self.right_hand_cols])

        return np.asarray(a, dtype=np.float32)

    def __getitem__(self, index):

        demo_dir = self.episode_dirs[index]
        # csv_path = os.path.join(demo_dir, "ee_pos", "ee_poses_and_hands.csv")
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)
        episode_len = len(df)
        start_ts = np.random.choice(episode_len)

        # get observation at start_ts for both cameras ("left" and "right")
        def load_img(cam_name):
            img_path = os.path.join(demo_dir, cam_name, f"{start_ts:06d}.jpg")
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))
            return img_rgb

        img_left = load_img("left")
        img_right = load_img("right")

        # after: img_left = load_img("left"); img_right = load_img("right")
        if self.apply_data_aug:
            aug = self.transforms(image=img_left, image_right=img_right)
            img_left = aug["image"]
            img_right = aug["image_right"]

        # Build left-hand-only actions from start_ts..end_ts (skip every 2)
        end_ts = min(start_ts + self.chunk_size * 2, episode_len)
        action = [self._row_to_action(df.iloc[t]) for t in range(start_ts, end_ts, 2)]
        action = np.stack(action, axis=0)

        action_len = action.shape[0]
        action_dim = action.shape[1]

        # compute qpos
        row0 = df.iloc[start_ts]
        qpos_parts = []

        # right wrist position
        qpos_parts.extend([row0["right_pos_x"], row0["right_pos_y"], row0["right_pos_z"]])

        # right wrist orientation (quat → 6D)
        rq0 = [row0["right_ori_x"], row0["right_ori_y"], row0["right_ori_z"], row0["right_ori_w"]]
        rR0 = R.from_quat(rq0).as_matrix()
        qpos_parts.extend(rR0[:, :2].reshape(-1, order="F").tolist())

        # right_actual_hand joints
        qpos_parts.extend([row0[c] for c in self.right_actual_hand_cols])

        qpos = np.asarray(qpos_parts, dtype=np.float32)

        # Now make positions relative to the first frame (left wrist only: indices 0..2)
        action[:, 0:3] -= action[0, 0:3]

        fixed_len = 400
        padded_action = np.zeros((fixed_len, action_dim), dtype=np.float32)
        padded_action[:action_len] = action

        is_pad = np.zeros(fixed_len, dtype=np.float32)
        is_pad[action_len:] = 1

        # stack both camera images: shape (2, H, W, C)
        all_cam_images = np.stack([img_left, img_right], axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)            # (k, H, W, C)
        qpos_data = torch.from_numpy(qpos).float()               # (A,)
        action_data = torch.from_numpy(padded_action).float()    # (T, A)
        is_pad = torch.from_numpy(is_pad).bool()                 # (T,)

        # channel last -> channel first per camera
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        if self.normalize:
            action_data = (action_data - norm_stats["action_mean"]) / norm_stats["action_std"]
            qpos_data   = (qpos_data   - norm_stats["qpos_mean"])   / norm_stats["qpos_std"]  # NEW

        return image_data, qpos_data, action_data, is_pad
##################################################################
##################################################################
# UNCOMMENT ME to find norm parameters through below code
# TODO: make sure to set chunk_size that you're using!
# TODO: make sure in the dataset normalize is set to False, so that we get raw action values!
##################################################################
##################################################################
# dataset_dir = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"

# # Create dataset (relative positions, no normalization yet)
# ds = GalaxeaDataset(dataset_dir=dataset_dir, chunk_size=45, apply_data_aug = True, normalize=False)

# # TODO!!!!!!! Check that normalization is implemented as you want it!!! right now it's implemented as a single arm, not bimanual
# from norm_stats import compute_delta_action_norm_stats_from_dirs
# compute_delta_action_norm_stats_from_dirs(
#     ds,
#     chunk_size=45,
#     stride=2,
#     out_path="norm_stats_galaxea_delta.npz",
#     print_literal=True
# )

#################################################################
#################################################################
# UNCOMMENT ME to visualize the dataset!
#################################################################
#################################################################

# Code to visualize the dataset
# def _to_numpy(arr):
#     if isinstance(arr, torch.Tensor):
#         arr = arr.detach().cpu().numpy()
#     return arr

# def _save_pair(image_data, out_dir, base_name):
#     """
#     image_data: (2, C, H, W) or (2, H, W, C), values in [0,1] or [0,255]
#     Saves: {base_name}_left.jpg and {base_name}_right.jpg
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     x = _to_numpy(image_data)

#     # Ensure (2, H, W, C)
#     if x.ndim != 4 or x.shape[0] != 2:
#         raise ValueError(f"Expected 4D with first dim=2, got {x.shape}")
#     if x.shape[1] in (1, 3):           # (2, C, H, W)
#         x = np.transpose(x, (0, 2, 3, 1)) # (2, H, W, C)

#     for i, cam in enumerate(["left", "right"]):
#         img = x[i]  # HWC, RGB
#         if img.dtype != np.uint8:
#             if img.max() <= 1.0:
#                 img = (img * 255.0).clip(0, 255).astype(np.uint8)
#             else:
#                 img = img.clip(0, 255).astype(np.uint8)
#         bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(out_dir, f"{base_name}_{cam}.jpg"), bgr)

# def dump_dataset_images(dataset, out_dir="viz_out", num_samples=20, start=0, prefix="run"):
#     save_dir = os.path.join(out_dir, prefix)
#     os.makedirs(save_dir, exist_ok=True)

#     # Optional: reproducibility
#     np.random.seed(0)
#     torch.manual_seed(0)

#     end = min(start + num_samples, len(dataset))
#     for idx in range(start, end):
#         image_data, qpos, action, is_pad = dataset[idx]
#         print(qpos.min().item(),  qpos.max().item(),  qpos.mean().item())
#         print(action.min().item(), action.max().item(), action.mean().item())
#         _save_pair(image_data, save_dir, f"{idx:06d}")
#     print(f"Saved {end-start} samples to {save_dir}")

# # TODO: change me
# dataset_dir = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"
# # init your dataset exactly as you do now
# ds = GalaxeaDataset(dataset_dir=dataset_dir,
#                      chunk_size=45, apply_data_aug=True, normalize=True)

# # dump a few samples with augmentation ON
# dump_dataset_images(ds, out_dir="viz_out", num_samples=20, prefix="aug")

# # (optional) compare with augmentation OFF
# # ds_noaug = GalaxeaDataset(dataset_dir="/path/to/Galaxea", chunk_size=50, apply_data_aug=False, normalize=True)
# # dump_dataset_images(ds_noaug, out_dir="viz_out", num_samples=20, prefix="orig")
