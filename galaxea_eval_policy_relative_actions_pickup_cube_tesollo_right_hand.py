"""
usage:
python3 galaxea_eval_policy_relative_actions_pickup_cube_tesollo_right_hand.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpt_galaxea --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --policy_class ACT --num_epochs 1 --num_queries 45
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

stride = 2 # TODO: change as needed
demo = 3
idx = 90
DEMO_DIR = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_{demo}"
image_path_l = os.path.join(DEMO_DIR, "left",  f"{idx:06d}.jpg")
image_path_r = os.path.join(DEMO_DIR, "right", f"{idx:06d}.jpg")
csv_path     = os.path.join(DEMO_DIR, "ee_hand.csv")
TS = idx
norm_stats = np.load("norm_stats_galaxea_delta.npz")
ckpt_name = "policy_last.ckpt"
# ckpt_name = "policy_last.ckpt"
camera_names = ['left', 'right']  # Assuming both cameras are used

def build_right_qpos_from_csv(csv_path: str, ts: int) -> np.ndarray:
    df  = pd.read_csv(csv_path)
    row = df.iloc[ts]

    pos = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float32)

    rq  = np.array([row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]], dtype=np.float32)
    Rm  = R.from_quat(rq).as_matrix()
    rot6d = Rm[:, :2].reshape(-1, order="F").astype(np.float32)

    fingers_actual = np.array([row[f"right_actual_hand_{i}"] for i in range(20)], dtype=np.float32)

    return np.concatenate([pos, rot6d, fingers_actual], axis=0)  # (29,)


def get_image():
    # Read left image
    img_bgr_l = cv2.imread(image_path_l)
    img_rgb_l = cv2.cvtColor(img_bgr_l, cv2.COLOR_BGR2RGB)
    img_rgb_l = cv2.resize(img_rgb_l, (224, 224))

    # Read right image
    img_bgr_r = cv2.imread(image_path_r)
    img_rgb_r = cv2.cvtColor(img_bgr_r, cv2.COLOR_BGR2RGB)
    img_rgb_r = cv2.resize(img_rgb_r, (224, 224))

    all_cam_images = np.stack([img_rgb_l, img_rgb_r], axis=0)  # (k, H, W, C)
    image_data = torch.from_numpy(all_cam_images)              # (k, H, W, C)
    image_data = torch.einsum('k h w c -> k c h w', image_data).unsqueeze(0)  # (1, k, C, H, W)
    image_data = image_data.float() / 255.0
    return image_data

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

def plot_right_wrist_trajectory(pred_actions: torch.Tensor,
                                csv_path: str,
                                ts: int,
                                save_path: str = "wrist_trajectory_right.png"):
    """
    pred_actions encodes relative RIGHT wrist deltas (Δx,Δy,Δz) w.r.t. the base pose at 'ts'.
    """
    pred_np = pred_actions.squeeze(0).cpu().numpy()
    dright  = pred_np[:, 0:3]

    df = pd.read_csv(csv_path)
    T  = dright.shape[0]
    end = min(ts + T*stride, len(df))
    gt_rows  = df.iloc[ts:end:stride]
    gt_right = gt_rows[["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy()

    base_right = df.iloc[ts][["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy(dtype=float)

    L = min(T, gt_right.shape[0])
    dright   = dright[:L]
    gt_right = gt_right[:L]

    pred_right_abs = dright + base_right[None, :]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Right Wrist Trajectory (GT vs Pred)")
    ax.scatter(gt_right[:,0], gt_right[:,1], gt_right[:,2], label="GT Right", s=20)
    ax.scatter(pred_right_abs[:,0], pred_right_abs[:,1], pred_right_abs[:,2], label="Pred Right", marker="^", s=20)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); plt.tight_layout()
    fig.savefig(save_path, dpi=300); plt.close(fig)
    print(f"Saved 3-D right trajectory → {save_path}")

def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy

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
        # @TODO Ke: read stereo images from the Zed mini
        curr_image = get_image().float().cuda()

        # qpos from CSV at the same ts (29-D: left pos + rot6d + 20 joints)
        # @TODO Ke: This time the current state is not zeros, we need to use the actual current state
        # which is left hand wrist position + rotation (6d rot) + 20 joints of the left tesollo hand.
        # See this function (build_left_qpos_from_csv()) to see how to build a 6d rot, very simple 
        qpos_numpy = build_right_qpos_from_csv(csv_path, TS)
        qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
        # Note, qpos also requires its own normalization now
        qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], device=qpos.device, dtype=qpos.dtype)
        qpos_std  = torch.as_tensor(norm_stats["qpos_std"],  device=qpos.device, dtype=qpos.dtype)
        print(qpos.shape, qpos_mean.shape)
        qpos = (qpos - qpos_mean) / qpos_std


        all_actions = policy(qpos, curr_image)

        # unnormalize actions (29-D)
        action_mean = torch.as_tensor(norm_stats["action_mean"], device=all_actions.device, dtype=all_actions.dtype)
        action_std  = torch.as_tensor(norm_stats["action_std"],  device=all_actions.device, dtype=all_actions.dtype)
        pred_actions = all_actions * action_std + action_mean

        # visualize only LEFT GT vs Pred
        plot_right_wrist_trajectory(pred_actions, csv_path, TS, save_path="wrist_trajectory_left.jpg")

        # ---- LEFT wrist outputs for deployment (no right-hand slices) ----
        pred_np = pred_actions.squeeze(0).cpu().numpy()  # [T, 29]
        dright = pred_np[:, 0:3]                          # Δx,Δy,Δz for left wrist

        # if you have the robot's current left wrist pos (3,), add deltas:
        current_wrist_right = np.zeros((3,), dtype=np.float32)  # TODO Ke: replace with actual robot reading
        pred_right_wrist = dright + current_wrist_right

        # right wrist rotation (6D) -> quaternion
        right_rot6d = pred_np[:, 3:9]                     # [T, 6]
        right_quat_xyzw = rot6d_to_quat_xyzw(right_rot6d)  # [T, 4]
        print(right_quat_xyzw.shape)


        # ---- Quaternion error via SciPy (no custom math) ----
        df = pd.read_csv(csv_path)
        T_pred = right_quat_xyzw.shape[0]
        end = min(TS + T_pred*stride, len(df))
        gt_q = df.iloc[TS:end:stride][["right_ori_x", "right_ori_y", "right_ori_z", "right_ori_w"]].to_numpy()

        L = min(len(right_quat_xyzw), len(gt_q))
        rel = R.from_quat(right_quat_xyzw[:L]) * R.from_quat(gt_q[:L]).inv()
        err_deg = np.degrees(rel.magnitude())

        print(f"[Right wrist quat] N={L} | mean={err_deg.mean():.2f}° | "
            f"median={np.median(err_deg):.2f}° | p90={np.percentile(err_deg,90):.2f}° | "
            f"max={err_deg.max():.2f}°")

        plt.figure(figsize=(7, 3.2))
        plt.plot(err_deg)
        plt.xlabel("Timestep"); plt.ylabel("Angular error (deg)")
        plt.title("Right Wrist Quaternion Error (SciPy)")
        plt.tight_layout()
        out_path = "right_quat_error_deg.png"
        plt.savefig(out_path, dpi=300); plt.close()
        print(f"Saved quaternion error plot → {out_path}")
                

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
    parser.add_argument('--state_dim', type=int, default=29) # TODO: update the state dim
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