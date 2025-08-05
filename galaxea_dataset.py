import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: make sure this is the correct path to norm_stats!
norm_stats = np.load("norm_stats_galaxea_delta.npz")

class GalaxeaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, chunk_size, normalize = True):
        super(GalaxeaDataset).__init__()
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.chunk_size = chunk_size

        # collect demo folders: Demo1, Demo2, ..., DemoN
        def _demo_key(name):
            # extract integer after "Demo", fall back to name for robustness
            try:
                return int(name.replace("Demo", ""))
            except ValueError:
                return float("inf")

        self.episode_dirs = sorted(
            [
                os.path.join(self.dataset_dir, d)
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d)) and d.startswith("Demo")
            ],
            key=lambda p: _demo_key(os.path.basename(p)),
        )

    def __len__(self):
        return len(self.episode_dirs)

    def _row_to_action(self, row):
        # Left wrist position (3)
        a = [row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]]

        # Left wrist orientation (quat -> 6D)
        lq = [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]]
        lR = R.from_quat(lq).as_matrix()
        a.extend(lR[:, :2].reshape(-1).tolist())

        # Right wrist position (3)
        a.extend([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]])

        # Right wrist orientation (quat -> 6D)
        rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
        rR = R.from_quat(rq).as_matrix()
        a.extend(rR[:, :2].reshape(-1).tolist())

        # Left and right hand (6 + 6)
        a.extend([row[f"left_hand_{i}"] for i in range(6)])
        a.extend([row[f"right_hand_{i}"] for i in range(6)])
        return np.asarray(a, dtype=np.float32)

    def __getitem__(self, index):

        demo_dir = self.episode_dirs[index]
        csv_path = os.path.join(demo_dir, "ee_pos", "ee_poses_and_hands.csv")
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

        # get actions from start_ts to start_ts + chunk_size (skipping every 2 frames)
        end_ts = min(start_ts + self.chunk_size * 2, episode_len)
        action = [self._row_to_action(df.iloc[t]) for t in range(start_ts, end_ts, 2)]
        action = np.stack(action, axis=0)
        # Left wrist position indices: 0,1,2
        action[:, 0:3] -= action[0, 0:3]
        # Right wrist position indices: 9,10,11
        action[:, 9:12] -= action[0, 9:12]
        action_len = action.shape[0]
        action_dim = action.shape[1]

        # qpos is zeros, same dimension as a single action vector
        qpos = np.zeros((action_dim,), dtype=np.float32)

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

        return image_data, qpos_data, action_data, is_pad
##################################################################
# UNCOMMENT ME to find norm parameters through below code
# TODO: make sure to set chunk_size that you're using!
# TODO: make sure in the dataset normalize is set to False, so that we get raw action values!
##################################################################
dataset_dir = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20"

# Create dataset (relative positions, no normalization yet)
ds = GalaxeaDataset(dataset_dir=dataset_dir, chunk_size=45, normalize=False)

from norm_stats import compute_delta_action_norm_stats_from_dirs
compute_delta_action_norm_stats_from_dirs(
    ds,
    chunk_size=45,
    stride=2,
    out_path="norm_stats_galaxea_delta.npz",
    print_literal=False
)


# Legacy code, use below for absolute actions 
# # make sure normalize is set to False, so that we get raw action values from the dataset
# ds = GalaxeaDataset(dataset_dir= dataset_dir, chunk_size = 45, normalize=False)
# from norm_stats import compute_action_norm_stats_from_dirs
# norm_stats = compute_action_norm_stats_from_dirs(
#     ds,
#     out_path="norm_stats_galaxea.npz",
#     print_literal=True,            # optional
#     var_name="GALAXEA_NORM_STATS"  # optional
# )

# stats = np.load("norm_stats_galaxea.npz")
# print(stats["action_mean"])
# print(stats["action_std"])
# assert False

##################################################################
# sanity check on the dataset
##################################################################
# print(f"# demos: {len(ds)}")
# # print(ds.episode_dirs)
# image_data, qpos_data, action_data, is_pad = ds[0]

# print("\nShapes")
# print("  image_data:", tuple(image_data.shape))  # (1, C, H, W)
# print("  qpos_data: ", tuple(qpos_data.shape))   # (A,)
# print("  action_data:", tuple(action_data.shape))# (T, A)
# print("  is_pad:    ", tuple(is_pad.shape))      # (T,)

# # Image stats and visualize
# img = image_data[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C), [0,1], RGB
# print("\nImage stats")
# print(f"  min={img.min():.4f} max={img.max():.4f} mean={img.mean():.4f}")
# plt.imshow(img)
# plt.title(f"Sampled image (idx={0})")
# plt.axis("off")
# plt.show()

# # Action preview
# T, A = action_data.shape
# first_row = action_data[0].cpu().numpy()
# last_row = action_data[-1].cpu().numpy()
# print("\nActions")
# print("  first row (first 10 dims):", np.array2string(first_row[:10], precision=4, suppress_small=True))
# print("  last row  (first 10 dims):", np.array2string(last_row[:10], precision=4, suppress_small=True))
# print("  # padded timesteps:", int(is_pad.sum().item()))
# print("  is_pad first 20:", is_pad[:20].int().tolist())

# # qpos preview (should be zeros)
# print("\nqpos (first 12 dims):", qpos_data[:12].cpu().tolist())

