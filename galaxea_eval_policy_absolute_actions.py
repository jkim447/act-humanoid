"""
usage:
python3 galaxea_eval_policy.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpt_galaxea --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --policy_class ACT --num_epochs 1 --num_queries 45
"""

import os
import cv2
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from utils import set_seed, sample_box_pose, sample_insertion_pose
from policy import ACTPolicy
from visualize_episodes import save_videos
from sim_env import BOX_POSE
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  ̶k̶e̶e̶p̶s̶ ̶3̶D̶ ̶p̶r̶o̶j̶e̶c̶t̶i̶o̶n̶ ̶r̶e̶g̶i̶s̶t̶e̶r̶e̶d̶
    


image_path = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/left/000001.jpg"
csv_path = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/ee_pos/ee_poses_and_hands.csv"
norm_stats = np.load("norm_stats_galaxea.npz")

def plot_wrist_trajectories(pred_actions: torch.Tensor,
                            csv_path: str,
                            save_path: str = "wrist_trajectories.png"):
    """
    Args
    ----
    pred_actions: torch.Tensor, shape [1, 45, 30] – your `all_actions`
    csv_path:     str – path to ee_poses_and_hands.csv
    save_path:    str – where to store the PNG
    """

    # ---------------- predicted positions ----------------
    pred_np = pred_actions.squeeze(0).cpu().numpy()      # [45, 30]
    pred_left  = pred_np[:, 0:3]                         # x, y, z
    pred_right = pred_np[:, 9:12]

    # ---------------- ground-truth positions -------------
    df = pd.read_csv(csv_path)

    # rows 0..88 step 2  →  45 rows
    gt_rows = df.iloc[0:90:2]

    gt_left  = gt_rows[["left_pos_x",  "left_pos_y",  "left_pos_z"]].to_numpy()
    gt_right = gt_rows[["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy()

    # ---------------- 3-D plot ---------------------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Left & Right Wrist Trajectories")

    # left-wrist curves
    ax.scatter(gt_left[:,0],  gt_left[:,1],  gt_left[:,2],
           label="GT Left",  s=20)                 # ground truth
    ax.scatter(pred_left[:,0], pred_left[:,1], pred_left[:,2],
            label="Pred Left", marker='^', s=20)    # predictions

    # right-wrist points
    ax.scatter(gt_right[:,0],  gt_right[:,1],  gt_right[:,2],
            label="GT Right", s=20)
    ax.scatter(pred_right[:,0], pred_right[:,1], pred_right[:,2],
            label="Pred Right", marker='^', s=20)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()

    # save and clean up
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 3-D trajectory plot → {save_path}")


def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy

def get_image():
    """Load a single image and return it as a tensor in (k, c, h, w) format."""
    # Read and preprocess image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))  # resize to 224x224

    # new axis for different cameras (k = 1 for single camera)
    all_cam_images = np.stack([img_rgb], axis=0)  # (k, H, W, C)

    # convert to torch tensor
    image_data = torch.from_numpy(all_cam_images)  # (k, H, W, C)

    # channel-last -> channel-first
    image_data = torch.einsum('k h w c -> k c h w', image_data)
    image_data = image_data.unsqueeze(0)  # (1, k, C, H, W)


    # normalize to [0, 1]
    image_data = image_data / 255.0

    return image_data

def eval_act(config, ckpt_name, save_episode=True):
    set_seed(config['seed'])
    # Load policy
    ckpt_path = os.path.join(config['ckpt_dir'], ckpt_name)
    policy = make_policy(config['policy_config'])
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    with torch.inference_mode():
        qpos_numpy = qpos = np.zeros((30,), dtype=np.float32) # TODO: 30 is action dim, change if needed
        qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
        curr_image = get_image().float().cuda()
        all_actions = policy(qpos, curr_image)

        # unnormalize
        action_mean = torch.as_tensor(norm_stats["action_mean"], device=all_actions.device, 
                                        dtype=all_actions.dtype)# shape (30,)
        action_std  = torch.as_tensor(norm_stats["action_std"], device=all_actions.device, 
                                    dtype=all_actions.dtype)# shape (30,)

        all_actions_denorm = all_actions * action_std + action_mean
        
        # visualize output
        plot_wrist_trajectories(all_actions_denorm, csv_path, save_path="wrist_trajectories.jpg")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_last.ckpt')
    parser.add_argument('--policy_class', action='store', type=str, default = 'ACT')
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--camera_names', nargs='+', default=['top'])
    parser.add_argument('--episode_len', type=int, default=200)
    parser.add_argument('--num_rollouts', type=int, default=50)
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--state_dim', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default = 1)


    # ACT-specific params
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    policy_config = {
        'lr': args.lr,
        'num_queries': args.num_queries,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': args.camera_names,
    }

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'task_name': args.task_name,
        'camera_names': args.camera_names,
        'episode_len': args.episode_len,
        'num_rollouts': args.num_rollouts,
        'real_robot': args.real_robot,
        'onscreen_render': args.onscreen_render,
        'temporal_agg': args.temporal_agg,
        'state_dim': args.state_dim,
        'seed': args.seed,
        'policy_config': policy_config,
    }

    eval_act(config, args.ckpt_name, save_episode=True)