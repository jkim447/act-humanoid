"""
usage:
python3 galaxea_eval_policy_relative_actions_pose_traj.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpt_galaxea --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --policy_class ACT --num_epochs 1 --num_queries 45
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
from scipy.spatial.transform import Rotation as R
from pytransform3d.trajectories import plot_trajectory  # pip install pytransform3d

image_path_l = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/left/000001.jpg"
image_path_r = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/right/000001.jpg"
csv_path = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/ee_pos/ee_poses_and_hands.csv"
norm_stats = np.load("norm_stats_galaxea_delta.npz")
ckpt_name = "policy_last_relative_08052025.ckpt"
# ckpt_name = "policy_last.ckpt"
camera_names = ['left', 'right']  # Assuming both cameras are used

df = pd.read_csv(csv_path)
gt_rows = df.iloc[0:90:2]

def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Re-order quaternion columns from (x y z w) → (w x y z)."""
    return np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)

def plot_pose_trajectories(gt_rows: pd.DataFrame,
                           pred_left_pos: np.ndarray,  pred_left_q_xyzw: np.ndarray,
                           pred_right_pos: np.ndarray, pred_right_q_xyzw: np.ndarray,
                           save_path: str = "pose_trajectories.png") -> None:
    """Side-by-side view: GT (left) vs. prediction (right)."""
    # ── ground truth ────────────────────────────────────────────────────────────
    gt_left_pos  = gt_rows[["left_pos_x",  "left_pos_y",  "left_pos_z"]].to_numpy()
    gt_left_q    = gt_rows[["left_ori_x","left_ori_y","left_ori_z","left_ori_w"]].to_numpy()
    gt_right_pos = gt_rows[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy()
    gt_right_q   = gt_rows[["right_ori_x","right_ori_y","right_ori_z","right_ori_w"]].to_numpy()

    P_gt_left   = np.hstack([gt_left_pos,  _xyzw_to_wxyz(gt_left_q)])
    P_gt_right  = np.hstack([gt_right_pos, _xyzw_to_wxyz(gt_right_q)])

    # ── predictions ─────────────────────────────────────────────────────────────
    P_pred_left  = np.hstack([pred_left_pos,  _xyzw_to_wxyz(pred_left_q_xyzw)])
    P_pred_right = np.hstack([pred_right_pos, _xyzw_to_wxyz(pred_right_q_xyzw)])

    # ── common axis limits (unchanged) ──────────────────────────────────────────
    offset = (0.01, -0.01)
    xlim   = (0.34 + offset[0], 0.38 + offset[1])
    ylim   = (-0.10 + offset[0], 0.30 + offset[1])
    zlim   = (0.00, 0.12)           # no offset here
    size = 0.002  # marker size you used before

    # ── plotting ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 5))

    # left subplot: ground-truth -------------------------------------------------
    ax_gt = fig.add_subplot(121, projection="3d")
    plot_trajectory(ax=ax_gt, P=P_gt_left,  color="C0",
                    show_direction=False, s=size)
    plot_trajectory(ax=ax_gt, P=P_gt_right, color="C1",
                    show_direction=False, s=size)
    ax_gt.set_xlim(*xlim); ax_gt.set_ylim(*ylim); ax_gt.set_zlim(*zlim)
    ax_gt.set_box_aspect([1, 1, 1])
    ax_gt.set_title("Ground-truth")

    # right subplot: prediction --------------------------------------------------
    ax_pr = fig.add_subplot(122, projection="3d")
    plot_trajectory(ax=ax_pr, P=P_pred_left,  color="C0",
                    show_direction=False, s=size, alpha=0.7)
    plot_trajectory(ax=ax_pr, P=P_pred_right, color="C1",
                    show_direction=False, s=size, alpha=0.7)
    ax_pr.set_xlim(*xlim); ax_pr.set_ylim(*ylim); ax_pr.set_zlim(*zlim)
    ax_pr.set_box_aspect([1, 1, 1])
    ax_pr.set_title("Policy prediction")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Pose trajectory figure saved → {save_path}")

# def plot_pose_trajectories(gt_rows: pd.DataFrame,
#                            pred_left_pos: np.ndarray,  pred_left_q_xyzw: np.ndarray,
#                            pred_right_pos: np.ndarray, pred_right_q_xyzw: np.ndarray,
#                            save_path: str = "pose_trajectories.png") -> None:
#     """Visualise GT vs. policy poses with pytransform3d – all curves in one plot."""
#     # ── ground truth ────────────────────────────────────────────────────────────
#     gt_left_pos  = gt_rows[["left_pos_x",  "left_pos_y",  "left_pos_z"]].to_numpy()
#     gt_left_q    = gt_rows[["left_ori_x","left_ori_y","left_ori_z","left_ori_w"]].to_numpy()
#     gt_right_pos = gt_rows[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy()
#     gt_right_q   = gt_rows[["right_ori_x","right_ori_y","right_ori_z","right_ori_w"]].to_numpy()

#     P_gt_left   = np.hstack([gt_left_pos,  _xyzw_to_wxyz(gt_left_q)])
#     P_gt_right  = np.hstack([gt_right_pos, _xyzw_to_wxyz(gt_right_q)])

#     # ── predictions ─────────────────────────────────────────────────────────────
#     P_pred_left  = np.hstack([pred_left_pos,  _xyzw_to_wxyz(pred_left_q_xyzw)])
#     P_pred_right = np.hstack([pred_right_pos, _xyzw_to_wxyz(pred_right_q_xyzw)])

#     # ── plotting ───────────────────────────────────────────────────────────────
#     fig = plt.figure(figsize=(7, 6))
#     ax  = fig.add_subplot(111, projection="3d")

#     # plot_trajectory(ax=ax, P=P_gt_left,   color="C0", label="GT left",
#     #                 show_direction=False, s=0.04)
#     # plot_trajectory(ax=ax, P=P_pred_left, color="C0", label="Pred left",
#     #                 show_direction=False, s=0.04, alpha=0.6, linestyle="--")

#     size = 0.002
#     plot_trajectory(ax=ax, P=P_gt_right,   color="C1", #label="GT right",
#                     show_direction=False, s=size)
#     plot_trajectory(ax=ax, P=P_pred_right, color="C1", #label="Pred right",
#                     show_direction=False, s=size, alpha=0.6, linestyle="--")

#     # lock axes to the unit cube
#     ax.set_xlim(0.34, 0.38)
#     ax.set_ylim(-0.1, 0.3)
#     ax.set_zlim(0, 0.12)
#     ax.set_box_aspect([1, 1, 1])  # equal scale on x, y, z

#     ax.set_title("Wrist pose trajectories")
#     # ax.legend()
#     plt.tight_layout()
#     fig.savefig(save_path, dpi=300)
#     print(f"Pose trajectory figure saved → {save_path}")

# def rot6d_to_quat_xyzw(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     """
#     rot6d: [T, 6] with first two columns of rotation matrix flattened.
#     Returns quaternions in xyzw order: [T, 4].
#     """
#     assert rot6d.ndim == 2 and rot6d.shape[1] == 6, "Expected [T,6] rot6d"
#     a = rot6d[:, 0:3]
#     b = rot6d[:, 3:6]

#     # Gram–Schmidt to get orthonormal basis
#     a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
#     b = b - (np.sum(a * b, axis=1, keepdims=True) * a)
#     b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
#     c = np.cross(a, b)

#     R_mats = np.stack([a, b, c], axis=2)      # [T, 3, 3]
#     quats_xyzw = R.from_matrix(R_mats).as_quat()  # xyzw
#     return quats_xyzw


def rot6d_to_quat_xyzw(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Assumes the SECOND column (b) is already reliable and uses it as the
    primary direction. The FIRST column (a) is re-orthogonalized.

    Args
    ----
    rot6d : ndarray of shape [T, 6]   (two 3-vectors flattened)
    eps    : small number to avoid divide-by-zero

    Returns
    -------
    quats : ndarray of shape [T, 4] in xyzw order
    """
    assert rot6d.ndim == 2 and rot6d.shape[1] == 6, "Expected [T,6] rot6d"

    # b is trusted, a is adjusted
    b = rot6d[:, 3:6]                  # primary vector
    a = rot6d[:, 0:3]                  # vector to be re-orthogonalized

    # Normalize b
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)

    # Remove the component of a along b, then normalize
    a = a - (np.sum(a * b, axis=1, keepdims=True) * b)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)

    # Third column via right-hand rule
    c = np.cross(a, b)

    R_mats = np.stack([a, b, c], axis=2)  # [T, 3, 3]
    quats_xyzw = R.from_matrix(R_mats).as_quat()
    return quats_xyzw


def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy

# TODO: @Ke, make sure to read left and right images
def get_image():
    """Load a single image and return it as a tensor in (k, c, h, w) format."""
    # Read left image
    img_bgr_l = cv2.imread(image_path_l)
    img_rgb_l = cv2.cvtColor(img_bgr_l, cv2.COLOR_BGR2RGB)
    img_rgb_l = cv2.resize(img_rgb_l, (224, 224))  # resize to 224x224

    # Read right image
    img_bgr_r = cv2.imread(image_path_r)
    img_rgb_r = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB)
    img_rgb_r = cv2.resize(img_rgb_r, (224, 224))

    # new axis for different cameras (k = 1 for single camera)
    all_cam_images = np.stack([img_rgb_l, img_rgb_r], axis=0)  # (k, H, W, C)

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

        pred_actions = all_actions * action_std + action_mean
        
         # ---------------- predicted deltas -------------------
        pred_np = pred_actions.squeeze(0).cpu().numpy()      # [45, 30]
        dleft  = pred_np[:, 0:3]                             # Δx, Δy, Δz for left wrist
        dright = pred_np[:, 9:12]                            # Δx, Δy, Δz for right wrist

        # TODO @Ke, replace below lines with actual wrist positions from the robot
        # use the very first sample in gt_rows as the “current” wrist pose
        current_wrist_left = gt_rows.iloc[0][
            ["left_pos_x", "left_pos_y", "left_pos_z"]
        ].to_numpy(dtype=np.float32)

        current_wrist_right = gt_rows.iloc[0][
            ["right_pos_x", "right_pos_y", "right_pos_z"]
        ].to_numpy(dtype=np.float32)


        # TODO @KE, publish pred_left_wrist and pred_right_wrist to the robot for their wrist trajectories
        # convert deltas → absolute predictions
        pred_left_wrist  = dleft  + current_wrist_left
        pred_right_wrist = dright + current_wrist_right

        # Adjust these slices if your layout differs:
        left_rot6d  = pred_np[:,  3: 9]   # left wrist rotation 6D
        right_rot6d = pred_np[:, 12:18]   # right wrist rotation 6D

        # TODO @Ke, publish left_quat_xyzw and right_quat_xyzw to the robot for their wrist orientations
        # Convert to xyzw quaternions
        left_quat_xyzw  = rot6d_to_quat_xyzw(left_rot6d)   # [T, 4]
        right_quat_xyzw = rot6d_to_quat_xyzw(right_rot6d)  # [T, 4]
        print(left_quat_xyzw.shape, right_quat_xyzw.shape)

        plot_pose_trajectories(
            gt_rows,                     # already defined earlier
            pred_left_wrist,  left_quat_xyzw,
            pred_right_wrist, right_quat_xyzw,
            save_path="pose_trajectories.png"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    # parser.add_argument('--ckpt_name', type=str, default='policy_last.ckpt')
    parser.add_argument('--policy_class', action='store', type=str, default = 'ACT')
    parser.add_argument('--task_name', type=str, required=True)
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
        'camera_names': camera_names,
    }

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': ckpt_name,
        'task_name': args.task_name,
        'camera_names': camera_names,
        'episode_len': args.episode_len,
        'num_rollouts': args.num_rollouts,
        'real_robot': args.real_robot,
        'onscreen_render': args.onscreen_render,
        'temporal_agg': args.temporal_agg,
        'state_dim': args.state_dim,
        'seed': args.seed,
        'policy_config': policy_config,
    }

    eval_act(config, ckpt_name, save_episode=True)