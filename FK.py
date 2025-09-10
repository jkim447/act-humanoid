

# =============================================================================
# --- IMPORTS AND INITIAL SETUP ---
# =============================================================================

# --- put shim BEFORE importing urdfpy (needed for numpy>=1.24) ---
import numpy as np
if not hasattr(np, "float"): np.float = float

import csv
import cv2
import os
from datetime import datetime
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R

# =============================================================================
# --- CONFIGURATION & CONSTANTS ---
# =============================================================================

# --- Input/Output Directories ---
demo_base_dir = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_20"
img_dir_left = os.path.join(demo_base_dir, "left")
img_dir_right = os.path.join(demo_base_dir, "right") # NEW: Path for right images
csv_path = os.path.join(demo_base_dir, "ee_hand.csv")
urdf_left_path = "/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf"
urdf_right_path = "/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir_base = os.path.join("FK_results", timestamp)
output_dir_left = os.path.join(output_dir_base, "left_view")   # NEW: Output for left view
output_dir_right = os.path.join(output_dir_base, "right_view") # NEW: Output for right view
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

# --- Camera Parameters ---
# Left Camera
fx_l, fy_l = 730.2571411132812, 730.2571411132812
cx_l, cy_l = 637.2598876953125, 346.41082763671875
K_left = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]])

T_cam_to_base_left = np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
T_base_to_camera_left = np.linalg.inv(T_cam_to_base_left)

# Right Camera (!!! IMPORTANT: REPLACE WITH YOUR CALIBRATION DATA !!!)
# TODO: Replace these placeholder values with your actual right camera intrinsics.
fx_r, fy_r = 730.257, 730.257
cx_r, cy_r = 637.259, 346.410
K_right = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])

# TODO: Replace this placeholder transform with your actual right camera extrinsics.
# This is a guess assuming the right camera is ~6.5cm to the right of the left one.
T_cam_to_base_right = np.array([
    [-0.00334115, -0.8768872 ,  0.48068458,  0.14700305],
    [-0.99996141,  0.0068351 ,  0.00551836, -0.02680847],
    [-0.00812451, -0.4806476 , -0.87687621,  0.46483729],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
T_base_to_camera_right = np.linalg.inv(T_cam_to_base_right)


# --- Hand Model Transformations (Unchanged) ---
theta_y = np.pi
theta_z = -np.pi/2
right_theta_z = np.pi
c_y, s_y = np.cos(theta_y), np.sin(theta_y)
R_y = np.array([[ c_y, 0, s_y], [ 0, 1, 0], [-s_y, 0, c_y]])
c_z, s_z = np.cos(theta_z), np.sin(theta_z)
R_z = np.array([[c_z, -s_z, 0], [s_z,  c_z, 0], [0,  0, 1]])
c_right_z, s_right_z = np.cos(right_theta_z), np.sin(right_theta_z)
R_right_z = np.array([[c_right_z, -s_right_z, 0], [s_right_z,  c_right_z, 0], [0,  0, 1]])
R_ee_to_hand_manual_left = R_y @ R_z
R_ee_to_hand_manual_right = R_y @ R_z @ R_right_z
translation_offset_left = np.array([0.00, -0.033, 0.0])
translation_offset_right = np.array([-0.02, 0.02, 0.025])
T_ee_to_hand_left = np.eye(4)
T_ee_to_hand_left[:3, :3] = R_ee_to_hand_manual_left
T_ee_to_hand_left[:3, 3] = translation_offset_left
T_ee_to_hand_right = np.eye(4)
T_ee_to_hand_right[:3, :3] = R_ee_to_hand_manual_right
T_ee_to_hand_right[:3, 3] = translation_offset_right

# --- Visualization Parameters (Unchanged) ---
left_links_to_ignore = {"ll_dg_mount", "ll_dg_base"}
right_links_to_ignore = {"rl_dg_mount", "rl_dg_base"}
HAND_LINK_GROUPS = {
    "tips": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    "mids": ["thumb_knuckle2", "index_knuckle2", "middle_knuckle2", "ring_knuckle2", "pinky_knuckle2"],
    "knuckles": ["thumb_knuckle1", "index_knuckle1", "middle_knuckle1", "ring_knuckle1", "pinky_knuckle1"]
}

# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================

def pose_to_matrix(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T

def actuated_joint_names(robot: URDF):
    return [j.name for j in robot.joints if j.joint_type != "fixed"]

def get_link_connections(robot: URDF):
    connections = []
    for joint in robot.joints:
        connections.append((joint.parent, joint.child))
    return connections

def connect_link_group(link_names):
    return [(link_names[i], link_names[i+1]) for i in range(len(link_names) - 1)]

def get_world_fk_map(T_hand_world, T_fkbase_model_inv, fk_results, links_to_ignore=set()):
    points_map = {}
    for link_name, T_link_model in fk_results.items():
        if link_name in links_to_ignore:
            continue
        T_link_in_hand = T_fkbase_model_inv @ T_link_model
        T_link_world   = T_hand_world @ T_link_in_hand
        points_map[link_name] = T_link_world[:3, 3]
    return points_map

def project_and_draw_on_image(img, all_fk_maps, all_connections, K, T_base_to_cam):
    """Projects 3D world points onto an image and draws the skeleton."""
    if img is None:
        return None
    
    h, w = img.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # --- Project all 3D points to 2D image space ---
    projected_points = {}
    for hand, fk_map in all_fk_maps.items():
        for link_name, p_world in fk_map.items():
            p_cam = T_base_to_cam @ np.append(p_world, 1.0)
            depth = p_cam[2]
            if depth > 1e-6:
                u = int(fx * p_cam[0] / depth + cx)
                v = int(fy * p_cam[1] / depth + cy)
                projected_points[link_name] = {'uv': (u, v), 'depth': depth}

    # --- Collect and prepare all primitives for drawing ---
    draw_primitives = []
    for link_name, proj_data in projected_points.items():
        u, v = proj_data['uv']
        if 0 <= u < w and 0 <= v < h:
            color = (0, 255, 255) if link_name.startswith('ll_') else (255, 255, 0) # Left=Yellow, Right=Cyan
            draw_primitives.append({'type': 'circle', 'uv': (u,v), 'depth': proj_data['depth'], 'color': color, 'radius': 5})
    
    for hand, connections in all_connections.items():
        for parent, child in connections:
            if parent in projected_points and child in projected_points:
                p1_data, p2_data = projected_points[parent], projected_points[child]
                color = (0, 200, 200) if hand == 'left' else (200, 200, 0)
                draw_primitives.append({'type': 'line', 'uv1': p1_data['uv'], 'uv2': p2_data['uv'], 'depth': (p1_data['depth'] + p2_data['depth']) / 2.0, 'color': color, 'thickness': 2})

    # --- Sort primitives by depth and draw ---
    draw_primitives.sort(key=lambda p: p['depth'], reverse=True)
    img_rect = (0, 0, w, h)
    for item in draw_primitives:
        if item['type'] == 'circle':
            cv2.circle(img, item['uv'], item['radius'], item['color'], -1)
        elif item['type'] == 'line':
            is_visible, pt1, pt2 = cv2.clipLine(img_rect, item['uv1'], item['uv2'])
            if is_visible:
                cv2.line(img, pt1, pt2, item['color'], item['thickness'])
    
    return img

# =============================================================================
# --- MAIN SCRIPT EXECUTION ---
# =============================================================================

print("Loading URDF models and extracting structure...")
robot_l = URDF.load(urdf_left_path)
robot_r = URDF.load(urdf_right_path)

left_connections = get_link_connections(robot_l)
right_connections = get_link_connections(robot_r)

for group_name, links in HAND_LINK_GROUPS.items():
    left_links = [f"ll_{link}" for link in links]
    left_connections.extend(connect_link_group(left_links))
    right_links = [f"rl_{link}" for link in links]
    right_connections.extend(connect_link_group(right_links))

left_joint_names = actuated_joint_names(robot_l)
right_joint_names = actuated_joint_names(robot_r)
all_connections = {"left": left_connections, "right": right_connections}

print(f"Reading CSV and processing frames from: {csv_path}")
with open(csv_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        frame_id = row["frame_id"]
        
        # --- 1. Calculate 3D FK poses (camera-agnostic) ---
        T_left_ee_world = pose_to_matrix([float(row[f"left_pos_{a}"]) for a in "xyz"], [float(row[f"left_ori_{a}"]) for a in "xyzw"])
        T_right_ee_world = pose_to_matrix([float(row[f"right_pos_{a}"]) for a in "xyz"], [float(row[f"right_ori_{a}"]) for a in "xyzw"])
        
        T_left_hand_world = T_left_ee_world @ T_ee_to_hand_left
        T_right_hand_world = T_right_ee_world @ T_ee_to_hand_right

        left_angles = [float(row[f"left_actual_hand_{i}"]) for i in range(20)]
        right_angles = [float(row[f"right_actual_hand_{i}"]) for i in range(20)]
        
        left_fk = robot_l.link_fk(cfg=dict(zip(left_joint_names, left_angles)), use_names=True)
        right_fk = robot_r.link_fk(cfg=dict(zip(right_joint_names, right_angles)), use_names=True)

        T_left_fkbase_model_inv = np.linalg.inv(left_fk.get("FK_base", np.eye(4)))
        T_right_fkbase_model_inv = np.linalg.inv(right_fk.get("FK_base", np.eye(4)))

        left_fk_map = get_world_fk_map(T_left_hand_world, T_left_fkbase_model_inv, left_fk, left_links_to_ignore)
        right_fk_map = get_world_fk_map(T_right_hand_world, T_right_fkbase_model_inv, right_fk, right_links_to_ignore)
        all_fk_maps = {"left": left_fk_map, "right": right_fk_map}

        # --- 2. Process LEFT image ---
        img_path_left = os.path.join(img_dir_left, f"{frame_id}.jpg")
        img_left = cv2.imread(img_path_left)
        if img_left is not None:
            output_img_left = project_and_draw_on_image(img_left, all_fk_maps, all_connections, K_left, T_base_to_camera_left)
            out_path = os.path.join(output_dir_left, f"projected_{frame_id}.jpg")
            cv2.imwrite(out_path, output_img_left)
            print(f"Saved: {out_path}")
        else:
            print(f"[Warn] Left image not found: {img_path_left}")

        # --- 3. Process RIGHT image ---
        img_path_right = os.path.join(img_dir_right, f"{frame_id}.jpg")
        img_right = cv2.imread(img_path_right)
        if img_right is not None:
            output_img_right = project_and_draw_on_image(img_right, all_fk_maps, all_connections, K_right, T_base_to_camera_right)
            out_path = os.path.join(output_dir_right, f"projected_{frame_id}.jpg")
            cv2.imwrite(out_path, output_img_right)
            print(f"Saved: {out_path}")
        else:
            print(f"[Warn] Right image not found: {img_path_right}")

print("Processing complete.")