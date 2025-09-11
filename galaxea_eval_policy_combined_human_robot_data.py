"""
usage:
python3 galaxea_eval_policy_combined_human_robot_data.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpt_galaxea --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200  --lr 1e-5 --seed 0 --policy_class ACT --num_epochs 1 --num_queries 45
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
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"):   np.int   = int
if not hasattr(np, "bool"):  np.bool  = bool
from urdfpy import URDF

stride = 2 # TODO: change as needed
demo = 1
idx = 20
DEMO_DIR = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_{demo}"
image_path_l = os.path.join(DEMO_DIR, "left",  f"{idx:06d}.jpg")
image_path_r = os.path.join(DEMO_DIR, "right", f"{idx:06d}.jpg")
csv_path     = os.path.join(DEMO_DIR, "ee_hand.csv")
TS = idx
norm_stats = np.load("norm_stats_combined_human_robot_data.npz")
ckpt_name = "policy_last.ckpt"
# ckpt_name = "policy_last.ckpt"
camera_names = ['left', 'right']  # Assuming both cameras are used

# --- load URDF once (outside the function) ---
urdf_right_path = "/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"
robot_r = URDF.load(urdf_right_path)
right_joint_names = [j.name for j in robot_r.joints if j.joint_type != "fixed"]

# Extrinsics (base->cam)
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

# Hand model transforms EE->hand
theta_y = np.pi
theta_z = -np.pi/2
right_theta_z = np.pi

R_y = np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                [ 0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])
R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z),  np.cos(theta_z), 0],
                [0, 0, 1]])
R_right_z = np.array([[np.cos(right_theta_z), -np.sin(right_theta_z), 0],
                      [np.sin(right_theta_z),  np.cos(right_theta_z), 0],
                      [0, 0, 1]])

R_EE_TO_HAND_L = R_y @ R_z
R_EE_TO_HAND_R = R_y @ R_z @ R_right_z

T_EE_TO_HAND_L = np.eye(4); T_EE_TO_HAND_L[:3,:3] = R_EE_TO_HAND_L; T_EE_TO_HAND_L[:3,3] = np.array([0.00, -0.033, 0.0])
T_EE_TO_HAND_R = np.eye(4); T_EE_TO_HAND_R[:3,:3] = R_EE_TO_HAND_R; T_EE_TO_HAND_R[:3,3] = np.array([-0.02, 0.02, 0.025])

# reuse transforms from your dataset code
def _pose_to_T(pos_xyz, quat_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T

def _world_to_cam3(p_world3):
    ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
    pc = T_BASE_TO_CAM_LEFT @ ph
    return pc[:3]

def build_right_qpos_from_csv(csv_path: str, ts: int) -> np.ndarray:
    df  = pd.read_csv(csv_path)
    row = df.iloc[ts]

    # wrist position in CAMERA frame
    p_world = [row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]]
    p_cam   = _world_to_cam3(p_world).astype(np.float32)

    # wrist orientation → rot6d
    rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    rR = R.from_quat(rq).as_matrix()
    ori6d = rR[:, :2].reshape(-1, order="F").astype(np.float32)

    # joints (20 actual hand joints)
    joints20 = np.array([row[f"right_actual_hand_{i}"] for i in range(20)], dtype=np.float32)

    # fingertip positions via FK
    T_right_ee_world = _pose_to_T(
        [row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
        [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    )
    T_right_hand_world = T_right_ee_world @ T_EE_TO_HAND_R
    angles = [row[f"right_actual_hand_{i}"] for i in range(20)]
    right_fk = robot_r.link_fk(cfg=dict(zip(right_joint_names, angles)), use_names=True)

    tip_names = [f"rl_dg_{i}_tip" for i in range(1, 6)]
    tips = []
    for name in tip_names:
        if name in right_fk:
            T_link_world = T_right_hand_world @ np.linalg.inv(right_fk["FK_base"]) @ right_fk[name]
            tips.append(_world_to_cam3(T_link_world[:3, 3]))
    tips_cam = np.concatenate(tips, axis=0).astype(np.float32) if tips else np.zeros(15, np.float32)

    # final 44D state
    return np.concatenate([p_cam, ori6d, joints20, tips_cam], axis=0).astype(np.float32)


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

# def plot_right_wrist_trajectory(pred_actions: torch.Tensor,
#                                 csv_path: str,
#                                 ts: int,
#                                 save_path: str = "wrist_trajectory_right.png"):
#     """
#     pred_actions encodes relative RIGHT wrist deltas (Δx,Δy,Δz) w.r.t. the base pose at 'ts'.
#     """
#     pred_np = pred_actions.squeeze(0).cpu().numpy()
#     dright  = pred_np[:, 0:3]

#     df = pd.read_csv(csv_path)
#     T  = dright.shape[0]
#     end = min(ts + T*stride, len(df))
#     gt_rows  = df.iloc[ts:end:stride]
#     gt_right = gt_rows[["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy()

#     base_right = df.iloc[ts][["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy(dtype=float)

#     L = min(T, gt_right.shape[0])
#     dright   = dright[:L]
#     gt_right = gt_right[:L]

#     pred_right_abs = dright + base_right[None, :]

#     fig = plt.figure(figsize=(7, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_title("Right Wrist Trajectory (GT vs Pred)")
#     ax.scatter(gt_right[:,0], gt_right[:,1], gt_right[:,2], label="GT Right", s=20)
#     ax.scatter(pred_right_abs[:,0], pred_right_abs[:,1], pred_right_abs[:,2], label="Pred Right", marker="^", s=20)
#     ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
#     ax.legend(); plt.tight_layout()
#     fig.savefig(save_path, dpi=300); plt.close(fig)
#     print(f"Saved 3-D right trajectory → {save_path}")

def plot_right_wrist_trajectory(pred_actions: torch.Tensor,
                                csv_path: str,
                                ts: int,
                                save_path: str = "wrist_trajectory_right.png"):
    # camera->base rotation (inverse of base->camera)
    R_B2C = T_BASE_TO_CAM_LEFT[:3, :3]
    R_C2B = R_B2C.T

    pred_np = pred_actions.squeeze(0).cpu().numpy()
    d_cam   = pred_np[:, 0:3]                         # Δ in LEFT-camera frame
    d_base  = (R_C2B @ d_cam.T).T                     # rotate deltas into base frame

    df = pd.read_csv(csv_path)
    T  = d_base.shape[0]
    end = min(ts + T*stride, len(df))

    # GT right wrist in base frame
    gt_right = df.iloc[ts:end:stride][["right_pos_x","right_pos_y","right_pos_z"]].to_numpy()

    # starting pose in base frame
    base_right = df.iloc[ts][["right_pos_x","right_pos_y","right_pos_z"]].to_numpy(dtype=float)

    L = min(T, gt_right.shape[0])
    pred_right_abs = d_base[:L] + base_right[None, :]

    # plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Right Wrist Trajectory (GT vs Pred, base frame)")
    ax.scatter(gt_right[:L,0], gt_right[:L,1], gt_right[:L,2], label="GT", s=20)
    # ax.scatter(gt_right[0:3``,0], gt_right[0:3,1], gt_right[0:3,2], label="GT", s=20)
    
    ax.scatter(pred_right_abs[:,0], pred_right_abs[:,1], pred_right_abs[:,2], label="Pred", marker="^", s=20)
    # ax.scatter(pred_right_abs[0:3,0], pred_right_abs[0:3,1], pred_right_abs[0:3,2], label="Pred", marker="^", s=20)
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

        # @TODO Ke:the state inputs are: right hand wrist position (in left camera frame) + 6d rotation (in left camera frame) 
        # + 20 joints of the right tesollo hand + 5 finger tip keypoints (in left camera frame). 44 dimensions total.
        # Just provide the wrist position and orientation in the robot base frame and the 20 hand joints.
        # The existing code will convert them into the left camera frame and also compute the 5 finger tip positions in the left camera frame
        # So there is minimal work necessary here. 
        # See this function (build_left_qpos_from_csv()) to see how everything is built 
        qpos_numpy = build_right_qpos_from_csv(csv_path, TS)
        qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
        qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], device=qpos.device, dtype=qpos.dtype)
        qpos_std  = torch.as_tensor(norm_stats["qpos_std"],  device=qpos.device, dtype=qpos.dtype)
        qpos = (qpos - qpos_mean) / qpos_std
        # compute policy actions
        all_actions = policy(qpos, curr_image)

        # unnormalize actions (29-D)
        action_mean = torch.as_tensor(norm_stats["action_mean"], device=all_actions.device, dtype=all_actions.dtype)
        action_std  = torch.as_tensor(norm_stats["action_std"],  device=all_actions.device, dtype=all_actions.dtype)
        pred_actions = all_actions * action_std + action_mean

        # visualize only LEFT GT vs Pred
        plot_right_wrist_trajectory(pred_actions, csv_path, TS, save_path="wrist_trajectory_left.jpg")

        pred_np = pred_actions.squeeze(0).cpu().numpy()  # [T, 44]
        dright = pred_np[:, 0:3]                          # Δx,Δy,Δz for left wrist

        # if you have the robot's current left wrist pos (3,), add deltas:
        # TODO Ke: replace zeros with actual robot reading for the right wrist
        current_wrist_right = np.zeros((3,), dtype=np.float32)  
        pred_right_wrist = dright + current_wrist_right

        # right wrist rotation (6D) -> quaternion
        right_rot6d = pred_np[:, 3:9]                     # [T, 6]
        right_quat_xyzw = rot6d_to_quat_xyzw(right_rot6d)  # [T, 4]

        # TODO Ke: to apply the actions, use the predicted wrist position + orientation, and the 20 joint angles.
        # the action dimensions are delta position (3), 6d rotation (6), 20 joint angles (20), 5 finger tip positions (15) = 44
        # note that the wrist positions are in the left camera frame so they need to be transformed to the robot base frame.

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