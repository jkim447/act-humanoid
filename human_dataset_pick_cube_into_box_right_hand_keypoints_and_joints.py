import os, cv2, numpy as np, torch, pandas as pd
from scipy.spatial.transform import Rotation as R
import albumentations as A
from torch.utils.data import Dataset
if not hasattr(np,"float"): np.float=float
if not hasattr(np,"int"):   np.int=int
if not hasattr(np,"bool"):  np.bool=bool

# TODO: make sure this is the correct path to norm_stats!
norm_stats = np.load("norm_stats_combined_human_robot_data.npz")

# NOTE: the hardcoded array is cam->base; we invert to get base->cam
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))
R_B2C = T_BASE_TO_CAM_LEFT[:3,:3]

# << define K_LEFT for your raw image size >>
K_LEFT = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                   [0.0, 730.2571411132812, 346.41082763671875],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

def _pos_base_to_cam(pxyz):
    ph = np.array([pxyz[0], pxyz[1], pxyz[2], 1.0], dtype=np.float64)
    return (T_BASE_TO_CAM_LEFT @ ph)[:3].astype(np.float32)

def _rot_base_to_cam(R_base):
    return (R_B2C @ R_base).astype(np.float32)

class HumanDatasetKeypointsJoints(Dataset):
    def __init__(self, dataset_dir, chunk_size, stride=2, apply_data_aug=True, normalize=True):
        self.dataset_dir, self.chunk_size, self.stride = dataset_dir, chunk_size, stride
        self.apply_aug, self.normalize = apply_data_aug, normalize
        self.img_h, self.img_w = 224, 224

        def _demo_key(d):
            try:
                if d.startswith("demo_"): return int(d.split("demo_")[1])
                if d.startswith("Demo"):  return int(d.replace("Demo",""))
            except: pass
            return 10**9
        self.episode_dirs = sorted(
            [os.path.join(dataset_dir,d) for d in os.listdir(dataset_dir)
             if os.path.isdir(os.path.join(dataset_dir,d)) and (d.startswith("demo_") or d.startswith("Demo"))],
            key=lambda p:_demo_key(os.path.basename(p))
        )

        self.wrist_xyz  = ["right_wrist_x","right_wrist_y","right_wrist_z"]
        self.wrist_quat = ["right_wrist_qx","right_wrist_qy","right_wrist_qz","right_wrist_qw"]
        self.joint_cols = [f"right_hand_{i}" for i in range(20)]
        tip_ids = [4,8,12,16,20]  # MANO-style 0-based tips
        self.tip_cols = sum([[f"right_hand_kp{i}_x",f"right_hand_kp{i}_y",f"right_hand_kp{i}_z"] for i in tip_ids], [])

        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(height=self.img_h, width=self.img_w, scale=(0.95,1.0), p=0.4),
                A.Rotate(limit=5, p=0.4),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15, p=0.4),
                A.CoarseDropout(min_holes=18, max_holes=35, min_height=4, max_height=10,
                                min_width=4, max_width=10, p=0.4),
                A.Resize(height=self.img_h, width=self.img_w),
            ],
            additional_targets={"image_right":"image"}
        )

    def __len__(self): return len(self.episode_dirs)

    def _row_to_action(self, row):
        p_cam = _pos_base_to_cam([row[c] for c in self.wrist_xyz])
        R_base = R.from_quat([row[c] for c in self.wrist_quat]).as_matrix()
        R_cam  = _rot_base_to_cam(R_base)
        ori6d  = R_cam[:,:2].reshape(-1, order="F").astype(np.float32)
        joints = np.asarray([row[c] for c in self.joint_cols], dtype=np.float32)
        tips_cam = []
        for i in range(0, len(self.tip_cols), 3):
            tips_cam.append(_pos_base_to_cam([row[self.tip_cols[i]], row[self.tip_cols[i+1]], row[self.tip_cols[i+2]]]))

        tips_cam = np.concatenate(tips_cam, axis=0).astype(np.float32)  # (15,)
        return np.concatenate([p_cam.astype(np.float32), ori6d, joints, tips_cam], axis=0)  # (44,)

    def __getitem__(self, idx):
        demo = self.episode_dirs[idx]
        df = pd.read_csv(os.path.join(demo, "robot_commands.csv"))
        T = len(df)
        if T == 0: raise FileNotFoundError(f"Empty demo: {demo}")

        def _load_bgr(p):
            im = cv2.imread(p)
            return im  # return None if missing

        # ---------- robust frame sampling (retry up to 10 times) ----------
        MAX_TRIES = 10
        imgL_bgr = imgR_bgr = None
        s = None
        for _ in range(MAX_TRIES):
            s_try = np.random.randint(0, T)
            pL = os.path.join(demo, "left",  f"{s_try:06d}.jpg")
            pR = os.path.join(demo, "right", f"{s_try:06d}.jpg")
            imL, imR = _load_bgr(pL), _load_bgr(pR)
            if imL is not None and imR is not None:
                s = s_try
                imgL_bgr, imgR_bgr = imL, imR
                break    

        imgL_rgb = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
        imgR_rgb = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)
        if self.apply_aug:
            aug = self.transforms(image=imgL_rgb, image_right=imgR_rgb)
            imgL_rgb, imgR_rgb = aug["image"], aug["image_right"]
        else:
            imgL_rgb = cv2.resize(imgL_rgb, (self.img_w, self.img_h))
            imgR_rgb = cv2.resize(imgR_rgb, (self.img_w, self.img_h))

        end_ts = min(s + self.chunk_size * self.stride, T)
        action = np.stack([ self._row_to_action(df.iloc[t]) for t in range(s, end_ts, self.stride) ], axis=0).astype(np.float32)

        qpos = action[0].copy()                 # first absolute action (44)

        # ===== DEBUG: project wrist + 5 tips (t=0) onto RAW LEFT IMAGE =====
        DEBUG_PROJ = False # TODO: make sure this is set to false during training!
        if DEBUG_PROJ:
            def proj_cam3(p3):
                z = max(float(p3[2]), 1e-6)
                u = K_LEFT[0,0]*(float(p3[0])/z) + K_LEFT[0,2]
                v = K_LEFT[1,1]*(float(p3[1])/z) + K_LEFT[1,2]
                return int(round(u)), int(round(v))
            vis = imgL_bgr.copy()
            h, w = vis.shape[:2]
            # wrist
            u,v = proj_cam3(qpos[:3])
            if 0 <= u < w and 0 <= v < h: cv2.circle(vis,(u,v),8,(0,255,0),-1)  # green
            # tips
            tips = qpos[-15:].reshape(5,3)
            for (x,y,z) in tips:
                u,v = proj_cam3((x,y,z))
                if 0 <= u < w and 0 <= v < h: cv2.circle(vis,(u,v),6,(0,0,255),-1)  # red
            cv2.imwrite(os.path.join("/iris/projects/humanoid/act", f"debug_proj_{s:06d}.jpg"), vis)
            print(f"Wrote debug_proj_{s:06d}.jpg to /iris/projects/humanoid/act")

        # delta only on translation
        action[:,0:3] -= action[0,0:3]

        fixed_len, A = 400, action.shape[1]
        padded = np.zeros((fixed_len, A), np.float32); padded[:len(action)] = action
        is_pad = np.zeros((fixed_len,), np.bool_);     is_pad[len(action):] = True

        imgs = np.stack([imgL_rgb, imgR_rgb], axis=0)  # (2,H,W,3)
        image_t = torch.from_numpy(imgs).permute(0,3,1,2).float() / 255.0
        qpos_t  = torch.from_numpy(qpos).float()
        action_t= torch.from_numpy(padded).float()
        ispad_t = torch.from_numpy(is_pad)

        if self.normalize:
            action_t = (action_t - norm_stats["action_mean"]) / norm_stats["action_std"]
            qpos_t   = (qpos_t   - norm_stats["qpos_mean"])   / norm_stats["qpos_std"]

        return image_t, qpos_t, action_t, ispad_t


# TODO: uncomment to check the dataset!
# import os, torch, numpy as np, cv2
# from torch.utils.data import DataLoader

# # --- import your class & K_LEFT from the module where it's defined ---

# DATASET_DIR = "/iris/projects/humanoid/hamer/keypoint_human_data_wood_inbox"
# OUT_DIR     = "human_ds_vis"
# os.makedirs(OUT_DIR, exist_ok=True)

# def save_images(images_2chw, out_dir, idx):
#     """images: (2,C,H,W) float[0,1] -> write BGR jpgs"""
#     imgs = (images_2chw.permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")
#     for cam in range(imgs.shape[0]):
#         cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam{cam}.jpg"), imgs[cam][:,:,::-1])
#         print(f"Wrote {os.path.join(out_dir, f'sample{idx}_cam{cam}.jpg')}")

# def main():
#     ds = HumanDatasetKeypointsJoints(
#         dataset_dir=DATASET_DIR,
#         chunk_size=45,
#         stride=1,
#         apply_data_aug=True,   # start with no aug for repeatability
#         normalize=True         # raw for debugging
#     )
#     print(f"#episodes: {len(ds)}")
#     loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

#     for i, batch in enumerate(loader):
#         image_t, qpos_t, action_t, ispad_t = batch  # shapes: (B,2,C,H,W), (B,44), (B,400,44), (B,400)
#         image_2chw = image_t[0]                    # (2,C,H,W)
#         qpos       = qpos_t[0].numpy()             # (44,)
#         action     = action_t[0].numpy()           # (400,44)
#         is_pad     = ispad_t[0].numpy()            # (400,)

#         # save images
#         save_images(image_2chw, OUT_DIR, i)

#         # prints
#         print(f"[{i}] image {tuple(image_2chw.shape)}  qpos {qpos.shape}  action {action.shape}  is_pad {is_pad.shape}")
#         print("    wrist(cam xyz):", np.round(qpos[:3], 4))
#         tips = qpos[-15:].reshape(5,3)
#         print("    tips(cam xyz) first row:", np.round(tips[0], 4), " | min/max z:", tips[:,2].min(), tips[:,2].max())
#         print("    num valid steps:", int((~is_pad).sum()))

#         # stop after a few
#         if i >= 4:
#             break

# if __name__ == "__main__":
#     main()