import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"):   np.int   = int
if not hasattr(np, "bool"):  np.bool  = bool
from urdfpy import URDF

# TODO: make sure this is the correct path to norm_stats!
norm_stats = np.load("norm_stats_combined_human_robot_data.npz")

# ===================== FK/Projection Constants =====================
urdf_left_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf"
urdf_right_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"

# Intrinsics
K_LEFT  = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                    [0.0, 730.2571411132812, 346.41082763671875],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

# TODO: replace with your actual right intrinsics
K_RIGHT = np.array([[730.257, 0.0, 637.259],
                    [0.0, 730.257, 346.410],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

# Extrinsics (base->cam)
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

# TODO: replace with your actual right extrinsics
T_BASE_TO_CAM_RIGHT = np.linalg.inv(np.array([
    [-0.00334115, -0.8768872 ,  0.48068458,  0.14700305],
    [-0.99996141,  0.0068351 ,  0.00551836, -0.02680847],
    [-0.00812451, -0.4806476 , -0.87687621,  0.46483729],
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

# Which links to skip when building point sets
LEFT_LINKS_TO_IGNORE  = {"ll_dg_mount", "ll_dg_base"}
RIGHT_LINKS_TO_IGNORE = {"rl_dg_mount", "rl_dg_base"}

# Optional ordered groups (for drawing nice skeleton lines)
HAND_LINK_GROUPS = {
    "tips":      ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    "mids":      ["thumb_knuckle2", "index_knuckle2", "middle_knuckle2", "ring_knuckle2", "pinky_knuckle2"],
    "knuckles":  ["thumb_knuckle1", "index_knuckle1", "middle_knuckle1", "ring_knuckle1", "pinky_knuckle1"],
}

def _pose_to_T(pos_xyz, quat_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T

def _actuated_joint_names(robot: URDF):
    return [j.name for j in robot.joints if j.joint_type != "fixed"]

def _connect_group(prefix, names):
    full = [f"{prefix}{n}" for n in names]
    return [(full[i], full[i+1]) for i in range(len(full)-1)]

def _project_points(P_world_3x, K, T_base_to_cam):
    P_h = np.concatenate([P_world_3x, np.ones((P_world_3x.shape[0], 1))], axis=1)
    Pc = (T_base_to_cam @ P_h.T).T
    z = Pc[:, 2:3]
    valid = (z[:, 0] > 1e-6)
    uv = (K @ (Pc[:, :3] / z).T).T[:, :2]
    return uv, z[:, 0], valid

def _draw_skeleton(img_bgr, uv_by_name, connections, point_color, line_color):
    h, w = img_bgr.shape[:2]
    rect = (0, 0, w, h)
    for name, (u, v) in uv_by_name.items():
        u, v = int(u), int(v)
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_bgr, (u, v), 5, point_color, -1)
    for a, b in connections:
        if a in uv_by_name and b in uv_by_name:
            p1 = tuple(map(int, uv_by_name[a]))
            p2 = tuple(map(int, uv_by_name[b]))
            is_visible, p1c, p2c = cv2.clipLine(rect, p1, p2)
            if is_visible:
                cv2.line(img_bgr, p1c, p2c, line_color, 2)

class FKProjector:
    """
    Loads URDFs once, projects 3D link positions for both hands,
    returns 2D keypoints and optionally overlays them on images.
    """
    def __init__(self, urdf_left_path, urdf_right_path):
        self.robot_l = URDF.load(urdf_left_path)
        self.robot_r = URDF.load(urdf_right_path)
        self.left_joint_names  = _actuated_joint_names(self.robot_l)
        self.right_joint_names = _actuated_joint_names(self.robot_r)

        # Precompute drawing connections
        self.left_connections  = [(j.parent, j.child) for j in self.robot_l.joints]
        self.right_connections = [(j.parent, j.child) for j in self.robot_r.joints]
        self.left_connections += _connect_group("ll_", HAND_LINK_GROUPS["tips"])
        self.left_connections += _connect_group("ll_", HAND_LINK_GROUPS["mids"])
        self.left_connections += _connect_group("ll_", HAND_LINK_GROUPS["knuckles"])
        self.right_connections += _connect_group("rl_", HAND_LINK_GROUPS["tips"])
        self.right_connections += _connect_group("rl_", HAND_LINK_GROUPS["mids"])
        self.right_connections += _connect_group("rl_", HAND_LINK_GROUPS["knuckles"])

        # Stable keypoint name order (sorted for determinism)
        self.left_kpt_names  = sorted([l.name for l in self.robot_l.links if l.name not in LEFT_LINKS_TO_IGNORE])
        self.right_kpt_names = sorted([l.name for l in self.robot_r.links if l.name not in RIGHT_LINKS_TO_IGNORE])

    def _fk_points_world(self, T_hand_world, fk_results, ignore):
        T_fkbase_inv = np.linalg.inv(fk_results.get("FK_base", np.eye(4)))
        pts = {}
        for link_name, T_link_model in fk_results.items():
            if link_name in ignore: 
                continue
            T_link_hand  = T_fkbase_inv @ T_link_model
            T_link_world = T_hand_world @ T_link_hand
            pts[link_name] = T_link_world[:3, 3]
        return pts  # dict[name] = (3,)

    def compute(self, row):
        # Build EE -> world transforms from CSV (this is in robot base frame)
        T_left_ee_world = _pose_to_T([row["left_pos_x"],  row["left_pos_y"],  row["left_pos_z"]],
                                     [row["left_ori_x"],  row["left_ori_y"],  row["left_ori_z"],  row["left_ori_w"]])
        T_right_ee_world = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                                      [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])

        T_left_hand_world  = T_left_ee_world  @ T_EE_TO_HAND_L
        T_right_hand_world = T_right_ee_world @ T_EE_TO_HAND_R

        left_angles  = [row[f"left_actual_hand_{i}"]  for i in range(20)]
        right_angles = [row[f"right_actual_hand_{i}"] for i in range(20)]

        left_fk  = self.robot_l.link_fk(cfg=dict(zip(self.left_joint_names,  left_angles)),  use_names=True)
        right_fk = self.robot_r.link_fk(cfg=dict(zip(self.right_joint_names, right_angles)), use_names=True)

        # 3D points in world
        l3d_map = self._fk_points_world(T_left_hand_world,  left_fk,  LEFT_LINKS_TO_IGNORE)
        r3d_map = self._fk_points_world(T_right_hand_world, right_fk, RIGHT_LINKS_TO_IGNORE)

        # Repack into ordered arrays
        L3D = np.stack([l3d_map[name] for name in self.left_kpt_names], axis=0)   # (N_l, 3)
        R3D = np.stack([r3d_map[name] for name in self.right_kpt_names], axis=0)  # (N_r, 3)
        return L3D, R3D

    def project_both_cams(self, L3D, R3D):
        # Left camera
        l_uv_L, l_z_L, l_valid_L = _project_points(L3D, K_LEFT,  T_BASE_TO_CAM_LEFT)
        r_uv_L, r_z_L, r_valid_L = _project_points(R3D, K_LEFT,  T_BASE_TO_CAM_LEFT)
        # Right camera
        l_uv_R, l_z_R, l_valid_R = _project_points(L3D, K_RIGHT, T_BASE_TO_CAM_RIGHT)
        r_uv_R, r_z_R, r_valid_R = _project_points(R3D, K_RIGHT, T_BASE_TO_CAM_RIGHT)

        return {
            "left_cam":  {"left_uv": l_uv_L, "left_z": l_z_L, "left_valid": l_valid_L,
                          "right_uv": r_uv_L, "right_z": r_z_L, "right_valid": r_valid_L},
            "right_cam": {"left_uv": l_uv_R, "left_z": l_z_R, "left_valid": l_valid_R,
                          "right_uv": r_uv_R, "right_z": r_z_R, "right_valid": r_valid_R},
        }

    def overlay_on_images(self, img_left_bgr, img_right_bgr, proj):
        # Build name->uv dicts for drawing
        lnames = self.left_kpt_names
        rnames = self.right_kpt_names

        # Left image
        l_uv_L = proj["left_cam"]["left_uv"];  r_uv_L = proj["left_cam"]["right_uv"]
        lmap_L = {name: l_uv_L[i] for i, name in enumerate(lnames) if proj["left_cam"]["left_valid"][i]}
        rmap_L = {name: r_uv_L[i] for i, name in enumerate(rnames) if proj["left_cam"]["right_valid"][i]}
        _draw_skeleton(img_left_bgr,  lmap_L, self.left_connections,  (0,255,255), (0,200,200))   # yellow/cyan-ish
        _draw_skeleton(img_left_bgr,  rmap_L, self.right_connections, (255,255,0), (200,200,0))

        # Right image
        l_uv_R = proj["right_cam"]["left_uv"]; r_uv_R = proj["right_cam"]["right_uv"]
        lmap_R = {name: l_uv_R[i] for i, name in enumerate(lnames) if proj["right_cam"]["left_valid"][i]}
        rmap_R = {name: r_uv_R[i] for i, name in enumerate(rnames) if proj["right_cam"]["right_valid"][i]}
        _draw_skeleton(img_right_bgr, lmap_R, self.left_connections,  (0,255,255), (0,200,200))
        _draw_skeleton(img_right_bgr, rmap_R, self.right_connections, (255,255,0), (200,200,0))

class GalaxeaDatasetKeypointsJoints(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, chunk_size, stride = 3, apply_data_aug = True, normalize = True,
                compute_keypoints=True, overlay_keypoints=False):
        super(GalaxeaDatasetKeypointsJoints).__init__()
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.stride = stride
        self.apply_data_aug = apply_data_aug
        self.img_height = 224
        self.img_width  = 224
        # hard-coded right-hand joint names
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]
        self.compute_keypoints  = compute_keypoints
        self.overlay_keypoints  = overlay_keypoints
        self.action_camera = "left"  # which camera to use for action ref frame
        self.fk = FKProjector(urdf_left_path, urdf_right_path) if compute_keypoints else None
        if self.fk is not None:
            tip_names = [f"rl_dg_{i}_tip" for i in range(1, 6)]  # matches URDF naming
            name_to_idx = {n: i for i, n in enumerate(self.fk.right_kpt_names)}
            self.right_tip_idx = [name_to_idx[n] for n in tip_names if n in name_to_idx]
        
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
                # TODO: undo me
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

    def _world_to_cam3(self, p_world3):
        """Transform a single 3D point from robot base -> left camera frame."""
        ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
        pc = T_BASE_TO_CAM_LEFT @ ph
        return pc[:3]  # (Xc, Yc, Zc)

    def __len__(self):
        return len(self.episode_dirs)

    def _row_to_action(self, row):
        # --- Right wrist position in CAMERA frame ---
        p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
        p_cam = self._world_to_cam3(p_world)  # (3,)

        # --- Right wrist orientation (quat -> 6D), unchanged ---
        rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
        rR = R.from_quat(rq).as_matrix()
        ori6d = rR[:, :2].reshape(-1, order="F")  # (6,)

        # --- Right hand command joints (20), unchanged ---
        joints20 = np.asarray([row[c] for c in self.right_hand_cols], dtype=np.float32)

        # --- Fingertip positions (5 tips) in CAMERA frame (each is 3D) ---
        # We need FK for THIS row to get right-hand link 3D in robot base frame; then transform to camera.
        tips_cam = np.zeros(15, dtype=np.float32)
        if self.fk is not None and len(self.right_tip_idx) == 5:
            _, R3D_world = self.fk.compute(row)  # R3D_world: (N_r, 3) in world/base
            tips = []
            for idx in self.right_tip_idx:
                pc = self._world_to_cam3(R3D_world[idx])  # (3,)
                tips.append(pc.astype(np.float32)) 
            tips_cam = np.concatenate(tips, axis=0)  # (15,)

        # --- Final action vector: [pos_cam(3), ori6d(6), joints20(20), tips_cam(15)] = 44 dims ---
        a = np.concatenate([p_cam.astype(np.float32), ori6d.astype(np.float32), joints20, tips_cam], axis=0)
        return a  # (44,)

    def __getitem__(self, index):
        demo_dir = self.episode_dirs[index]
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)
        episode_len = len(df)
        start_ts = np.random.choice(episode_len)

        # Load original images in BGR first (we will draw on these if requested)
        def load_img_bgr(cam_name):
            img_path = os.path.join(demo_dir, cam_name, f"{start_ts:06d}.jpg")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            return img_bgr

        img_left_bgr  = load_img_bgr("left")
        img_right_bgr = load_img_bgr("right")

        # Compute FK keypoints and project to both cameras
        keypoints_payload = None
        if self.compute_keypoints:
            row = df.iloc[start_ts]
            L3D, R3D = self.fk.compute(row)
            proj = self.fk.project_both_cams(L3D, R3D)

            # Optionally overlay on original images BEFORE any resize/Albumentations
            if self.overlay_keypoints:
                self.fk.overlay_on_images(img_left_bgr, img_right_bgr, proj)

            # Build a compact return structure with ordered names
            keypoints_payload = {
                "left_names":  self.fk.left_kpt_names,
                "right_names": self.fk.right_kpt_names,

                # For each camera, provide left-hand and right-hand 2D keypoints and depths
                "left_cam": {
                    "left_uv":  proj["left_cam"]["left_uv"].astype(np.float32),   # (N_l, 2)
                    "left_z":   proj["left_cam"]["left_z"].astype(np.float32),    # (N_l,)
                    "right_uv": proj["left_cam"]["right_uv"].astype(np.float32),  # (N_r, 2)
                    "right_z":  proj["left_cam"]["right_z"].astype(np.float32),   # (N_r,)
                },
                "right_cam": {
                    "left_uv":  proj["right_cam"]["left_uv"].astype(np.float32),
                    "left_z":   proj["right_cam"]["left_z"].astype(np.float32),
                    "right_uv": proj["right_cam"]["right_uv"].astype(np.float32),
                    "right_z":  proj["right_cam"]["right_z"].astype(np.float32),
                },
            }

        # Convert BGR to RGB for the augmentation pipeline (drawing already done if enabled)
        img_left_rgb  = cv2.cvtColor(img_left_bgr,  cv2.COLOR_BGR2RGB)
        img_right_rgb = cv2.cvtColor(img_right_bgr, cv2.COLOR_BGR2RGB)

        # Apply Albumentations jointly
        if self.apply_data_aug:
            aug = self.transforms(image=img_left_rgb, image_right=img_right_rgb)
            img_left_rgb  = aug["image"]
            img_right_rgb = aug["image_right"]
        else:
            # Even without aug, enforce size
            # TODO: undo me
            img_left_rgb  = cv2.resize(img_left_rgb,  (self.img_width, self.img_height))
            img_right_rgb = cv2.resize(img_right_rgb, (self.img_width, self.img_height))
            # pass  # already at desired size

        # Build action chunk, qpos, padding (unchanged from your code)
        end_ts = min(start_ts + self.chunk_size * self.stride, episode_len)
        action = [self._row_to_action(df.iloc[t]) for t in range(start_ts, end_ts, self.stride)]
        action = np.stack(action, axis=0)

        # --- qpos := first absolute action (matches 44 dims) ---
        a_abs0 = action[0].copy().astype(np.float32)  # (44,)
        qpos = a_abs0

        # plot finger tip action trajectory for sanity checking
        DEBUG_TRAJ = False # TODO: set to False to disable
        if DEBUG_TRAJ:
            STEP = 2  # draw every 2 timesteps; set 5 if you prefer
            tip_colors = [(0,0,255), (0,165,255), (0,255,255), (0,255,0), (255,0,0)]  # BGR: red, orange, yellow, green, blue

            def proj_cam3_to_uv(p3):
                z = max(float(p3[2]), 1e-6)
                u = K_LEFT[0,0]*(float(p3[0])/z) + K_LEFT[0,2]
                v = K_LEFT[1,1]*(float(p3[1])/z) + K_LEFT[1,2]
                return int(round(u)), int(round(v))

            img_dbg = img_left_bgr.copy()
            h, w = img_dbg.shape[:2]

            # draw tips for multiple timesteps
            for t in range(0, action.shape[0], STEP):
                tips_cam = action[t, -15:].reshape(5, 3)  # (5,3) in left-cam frame (absolute by construction)
                for i in range(5):
                    u, v = proj_cam3_to_uv(tips_cam[i])
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(img_dbg, (u, v), 5, tip_colors[i], -1)

            cv2.imwrite("debug_tips_traj.jpg", img_dbg)
            print("Saved debug_tips_traj.jpg")

        action_len = action.shape[0]
        action_dim = action.shape[1]

        # make the translation motion delta!
        action[:, 0:3] -= action[0, 0:3]

        fixed_len = 400
        padded_action = np.zeros((fixed_len, action_dim), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(fixed_len, dtype=np.float32)
        is_pad[action_len:] = 1

        # Stack images: (2, H, W, C)
        all_cam_images = np.stack([img_left_rgb, img_right_rgb], axis=0)

        # Tensors
        image_data  = torch.from_numpy(all_cam_images).permute(0,3,1,2).float() / 255.0
        qpos_data   = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_t    = torch.from_numpy(is_pad).bool()

        if self.normalize:
            action_data = (action_data - norm_stats["action_mean"]) / norm_stats["action_std"]
            qpos_data   = (qpos_data   - norm_stats["qpos_mean"])   / norm_stats["qpos_std"]

        # Return keypoints payload as an extra item
        return image_data, qpos_data, action_data, is_pad_t, #keypoints_payload

##################################################################
##################################################################
# UNCOMMENT ME to find norm parameters through below code
# TODO: make sure to set chunk_size that you're using!
# TODO: make sure in the dataset normalize is set to False, so that we get raw action values!
##################################################################
##################################################################
# dataset_dir = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"

# # Create dataset (relative positions, no normalization yet)
# ds = GalaxeaDatasetKeypointsJoints(dataset_dir=dataset_dir, chunk_size=45, apply_data_aug = True, normalize=False)

# # TODO!!!!!!! Check that normalization is implemented as you want it!!! right now it's implemented as a single arm, not bimanual
# from norm_stats import compute_delta_action_norm_stats_from_dirs
# compute_delta_action_norm_stats_from_dirs(
#     ds,
#     chunk_size=45,
#     stride=3,
#     out_path="norm_stats_galaxea_delta.npz",
#     print_literal=True
# )

#################################################################
#################################################################
# UNCOMMENT ME to visualize the dataset!
#################################################################
#################################################################

# def save_images(images, out_dir, idx):
#     """Save (2,C,H,W) tensor images as jpg."""
#     os.makedirs(out_dir, exist_ok=True)
#     imgs = images.permute(0,2,3,1).cpu().numpy()  # -> (2,H,W,C), float [0,1]
#     for cam_id in range(imgs.shape[0]):
#         img = (imgs[cam_id] * 255).astype("uint8")[:, :, ::-1]  # RGB->BGR for cv2
#         out_path = os.path.join(out_dir, f"sample{idx}_cam{cam_id}.jpg")
#         cv2.imwrite(out_path, img)
#         print(f"Saved {out_path}")


# # --- minimal fingertip overlay (left cam) ---
# ORIG_W, ORIG_H = 1280, 720  # set to your raw capture size
# Sx, Sy = 224/ORIG_W, 224/ORIG_H
# K224 = np.array([[K_LEFT[0,0]*Sx, 0, K_LEFT[0,2]*Sx],
#                  [0, K_LEFT[1,1]*Sy, K_LEFT[1,2]*Sy],
#                  [0, 0, 1]], dtype=np.float32)

# # optionally use this
# def save_with_tips_fullres(images_2chw, actions_ta, out_dir, idx):
#     os.makedirs(out_dir, exist_ok=True)
#     # images_2chw: (2,C,H,W), already full-res RGB in [0,1]
#     imgs = (images_2chw.permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")  # (2,H,W,C)

#     # Save both cams first
#     for cam in range(2):
#         out_path = os.path.join(out_dir, f"sample{idx}_cam{cam}.jpg")
#         cv2.imwrite(out_path, imgs[cam][:,:,::-1])  # RGB->BGR

#     # Get fingertip positions from action (left cam frame, 3D)
#     a0 = actions_ta[0].cpu().numpy()
#     tips = a0[-15:].reshape(5,3).astype(np.float32)  # (Xc,Yc,Zc)

#     # Project with original intrinsics K_LEFT
#     Z = np.clip(tips[:,2:3], 1e-6, None)
#     xn = tips[:,:2] / Z                     # (x/z, y/z)
#     uv = (K_LEFT[:2,:2] @ xn.T).T + K_LEFT[:2,2]  # (5,2) pixels in full res

#     # Draw dots on left cam image
#     left_path = os.path.join(out_dir, f"sample{idx}_cam0.jpg")
#     im = cv2.imread(left_path)  # full-res BGR
#     for u,v in uv:
#         u,v = int(round(u)), int(round(v))
#         if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
#             cv2.circle(im, (u,v), 8, (0,0,255), -1)  # red dots
#     cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam0_with_tips.jpg"), im)

# def save_with_tips(images_2chw, actions_ta, out_dir, idx):
#     os.makedirs(out_dir, exist_ok=True)
#     imgs = (images_2chw.permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")  # (2,H,W,C) RGB
#     # Save both cams first
#     for cam in range(2):
#         cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam{cam}.jpg"), imgs[cam][:,:,::-1])

#     # Get first timestep action and its last 15 dims -> (5,3) cam-frame points
#     a0 = actions_ta[0].cpu().numpy()
#     tips = a0[-15:].reshape(5,3).astype(np.float32)  # (Xc,Yc,Zc)
#     Z = np.clip(tips[:,2:3], 1e-6, None)
#     xn = tips[:,:2] / Z                               # (x/z, y/z)
#     uv = (K224[:2,:2] @ xn.T).T + K224[:2,2]          # (5,2) in 224x224

#     # Draw on left image (cam0) and resave
#     left_path = os.path.join(out_dir, f"sample{idx}_cam0.jpg")
#     im = cv2.imread(left_path)                        # BGR 224x224
#     for u,v in uv:
#         u,v = int(round(u)), int(round(v))
#         if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
#             cv2.circle(im, (u,v), 5, (255,0,255), -1)  # red dots
#     cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam0_with_tips.jpg"), im)


# ds = GalaxeaDatasetKeypointsJoints(
#     dataset_dir="/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox",
#     chunk_size=20,
#     apply_data_aug=False,
#     normalize=False,
#     compute_keypoints=True,
#     overlay_keypoints=True   # skeletons drawn before resizing
# )

# loader = DataLoader(ds, batch_size=1, shuffle=True)

# out_dir = "vis_samples"
# for i, batch in enumerate(loader):
#     image_data, qpos, action, is_pad, keypoints_payload = batch
#     save_images(image_data[0], out_dir, i)
#     # save_with_tips(image_data[0], action[0], out_dir, i)
#     save_with_tips_fullres(image_data[0], action[0], out_dir, i)
#     print("qpos:", qpos.shape, "action:", action.shape, "is_pad:", is_pad.shape)
#     if keypoints_payload is not None:
#         print("keypoints available")
#     if i >= 4:   # save first 5 samples only
#         break

# # (optional) compare with augmentation OFF
# # ds_noaug = GalaxeaDataset(dataset_dir="/path/to/Galaxea", chunk_size=50, apply_data_aug=False, normalize=True)
# # dump_dataset_images(ds_noaug, out_dir="viz_out", num_samples=20, prefix="orig")
