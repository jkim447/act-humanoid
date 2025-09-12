# compare_robot_human_trajs.py
import os, numpy as np, torch, matplotlib
matplotlib.use("Agg")  # do not open windows
import matplotlib.pyplot as plt

# --- import your dataset classes (expects these modules in PYTHONPATH) ---
from galaxea_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import GalaxeaDatasetKeypointsJoints
from human_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import HumanDatasetKeypointsJoints

# ------------------- HARD-CODED SETTINGS -------------------
robot_dir = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"
human_dir = "/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox"
norm_path = "norm_stats_combined_human_robot_data.npz"   # same file both datasets used
save_dir  = "./traj_compare_outputs"
os.makedirs(save_dir, exist_ok=True)

# robot ~ 2x slower -> your choices
robot_chunksize, robot_stride = 60, 1.5
human_chunksize, human_stride = 60, 1

apply_data_aug = True
normalize = True

import cv2

def _set_axes_equal(ax, X, Y, Z):
    xs = np.concatenate([X[:,0], Y[:,0]])
    ys = np.concatenate([X[:,1], Y[:,1]])
    zs = np.concatenate([X[:,2], Y[:,2]])
    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    r = max(np.ptp(xs), np.ptp(ys), np.ptp(zs))
    r = 1e-6 if r == 0 else r
    ax.set_xlim(xmid - r/2, xmid + r/2)
    ax.set_ylim(ymid - r/2, ymid + r/2)
    ax.set_zlim(zmid - r/2, zmid + r/2)

def spin_3d_and_save(fig, ax, out_dir, out_video, n_frames=240, fps=30,
                     elev=25, azim_start=30, azim_end=390, save_every=10):
    os.makedirs(out_dir, exist_ok=True)
    w, h = fig.canvas.get_width_height()
    fourccs = [cv2.VideoWriter_fourcc(*c) for c in ("mp4v", "avc1", "XVID")]
    vw = None
    for code in fourccs:
        vw = cv2.VideoWriter(out_video, code, fps, (w, h))
        if vw.isOpened(): break
    if vw is None or not vw.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try installing ffmpeg or a different codec.")

    for i in range(n_frames):
        az = azim_start + (azim_end - azim_start) * i / max(1, n_frames-1)
        ax.view_init(elev=elev, azim=az)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape(h, w, 3)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if (i % save_every) == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()

# ------------------- DATASETS -------------------
ds_robot = GalaxeaDatasetKeypointsJoints(
    dataset_dir=robot_dir, chunk_size=robot_chunksize, stride=robot_stride,
    apply_data_aug=apply_data_aug, normalize=normalize,
    compute_keypoints=True, overlay_keypoints=False
)
ds_human = HumanDatasetKeypointsJoints(
    dataset_dir=human_dir, chunk_size=human_chunksize, stride=human_stride,
    apply_data_aug=apply_data_aug, normalize=normalize
)

# ------------------- HELPERS -------------------
def first_true_index_bool(b):
    # b is torch.bool or np.bool_ array
    b = np.asarray(b)
    idx = np.where(b)[0]
    return int(idx[0]) if idx.size > 0 else int(len(b))

def denorm_actions(action_t, stats):
    # action_t: (T, 44) torch or np
    a = np.asarray(action_t, dtype=np.float32)
    return a * stats["action_std"] + stats["action_mean"]

def ortho6d_to_R(a6):
    """
    a6: (T,6) with first two columns of R (Zhou et al., "On the Continuity of Rotation Representations...")
    Returns: (T,3,3)
    """
    a6 = np.asarray(a6, dtype=np.float32)
    a1 = a6[:, 0:3]
    a2 = a6[:, 3:6]
    # normalize a1 -> b1
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-9)
    # make b2 orthonormal to b1
    a2_proj = (np.sum(b1 * a2, axis=1, keepdims=True)) * b1
    b2 = a2 - a2_proj
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
    # b3 = b1 x b2
    b3 = np.cross(b1, b2)
    # stack as columns
    Rm = np.stack([b1, b2, b3], axis=-1)  # (T,3,3)
    return Rm

def geodesic_deg(Ra, Rb):
    """
    Ra, Rb: (T,3,3)
    returns per-step geodesic angle in degrees
    """
    Rt = Ra @ np.transpose(Rb, (0,2,1))
    tr = np.clip((np.trace(Rt, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(tr))
    return ang

# ------------------- LOAD ONE SAMPLE EACH -------------------
# You can change the indices if you want specific episodes
robot_img, robot_qpos, robot_action_norm, robot_is_pad = ds_robot[0]
human_img, human_qpos, human_action_norm, human_is_pad = ds_human[2]

# ------------------- DENORMALIZE (since normalize=True) -------------------
stats = np.load(norm_path)
robot_action = denorm_actions(robot_action_norm, stats)   # (400,44)
human_action = denorm_actions(human_action_norm, stats)   # (400,44)

# trim padding
r_T = first_true_index_bool(robot_is_pad)
h_T = first_true_index_bool(human_is_pad)
robot_action = robot_action[:r_T]
human_action = human_action[:h_T]

# ------------------- 3D TRAJECTORY PLOT (wrist positions) -------------------
# positions are first 3 dims, already delta-ized in your datasets
r_pos = robot_action[:, 0:3]
h_pos = human_action[:, 0:3]

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(projection='3d')
ax.plot(r_pos[:,0], r_pos[:,1], r_pos[:,2], label="Robot", linewidth=2)
ax.plot(h_pos[:,0], h_pos[:,1], h_pos[:,2], label="Human", linewidth=2)

# highlight the first few steps
ax.plot(r_pos[0:5,0], r_pos[0:5,1], r_pos[0:5,2], linewidth=2, color='r')
ax.plot(h_pos[0:5,0], h_pos[0:5,1], h_pos[0:5,2], linewidth=2, color='r')

ax.set_title("Wrist Position Trajectories (Camera Frame, delta)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.legend()
_set_axes_equal(ax, r_pos, h_pos, r_pos)  # square-ish box so the spin looks good

# save a static image
pos_path = os.path.join(save_dir, "traj_wrist_positions_3d.png")
plt.savefig(pos_path, dpi=200, bbox_inches="tight")

# spin and save video + a subset of frames
rot_frames_dir = os.path.join(save_dir, "traj_wrist_positions_rot_frames")
rot_vid_path   = os.path.join(save_dir, "traj_wrist_positions_spin.mp4")
spin_3d_and_save(fig, ax, rot_frames_dir, rot_vid_path,
                 n_frames=360, fps=30, elev=25, azim_start=30, azim_end=390, save_every=12)

plt.close(fig)