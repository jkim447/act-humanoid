import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D
import os, numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless

def _set_axes_equal(ax, *pts):
    pts = [np.asarray(p) for p in pts if p is not None and len(p) > 0]
    if not pts: return
    xyz = np.concatenate(pts, axis=0)
    xmid, ymid, zmid = xyz[:,0].mean(), xyz[:,1].mean(), xyz[:,2].mean()
    rx, ry, rz = np.ptp(xyz[:,0]), np.ptp(xyz[:,1]), np.ptp(xyz[:,2])
    r = max(rx, ry, rz); r = 1e-6 if r == 0 else r
    ax.set_xlim(xmid - r/2, xmid + r/2)
    ax.set_ylim(ymid - r/2, ymid + r/2)
    ax.set_zlim(zmid - r/2, zmid + r/2)

def spin_3d_and_save(fig, ax, out_video, n_frames=360, fps=30,
                     elev=25, azim_start=30, azim_end=390):
    w, h = fig.canvas.get_width_height()
    # try a few codecs
    for fourcc in ("mp4v", "avc1", "XVID"):
        vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened(): break
    if not vw.isOpened():
        raise RuntimeError("Could not open VideoWriter. Install ffmpeg or try a different codec.")

    for i in range(n_frames):
        az = azim_start + (azim_end - azim_start) * i / max(1, n_frames-1)
        ax.view_init(elev=elev, azim=az)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape(h, w, 3)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()

# file paths
human_csv = "/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox/Demo1/robot_commands.csv"
robot_csv = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_2/ee_hand.csv"

# load
df_h = pd.read_csv(human_csv)
df_r = pd.read_csv(robot_csv)

# extract positions
h_pos = df_h[["right_wrist_x","right_wrist_y","right_wrist_z"]].to_numpy()
r_pos = df_r[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy()

# plot
# plot + save static and spinning video
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(h_pos[:,0], h_pos[:,1], h_pos[:,2], label="Human Right Wrist")
ax.plot(r_pos[:,0], r_pos[:,1], r_pos[:,2], label="Robot Right Wrist")

# highlight first few steps to show starts
ax.plot(h_pos[:5,0], h_pos[:5,1], h_pos[:5,2], linewidth=2, color="red")
ax.plot(r_pos[:5,0], r_pos[:5,1], r_pos[:5,2], linewidth=2, color="red")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend(); plt.tight_layout()
_set_axes_equal(ax, h_pos, r_pos)  # keeps bounds fixed while spinning

# static image
os.makedirs("./traj_compare_outputs", exist_ok=True)
png_path = "./traj_compare_outputs/human_robot_wrist_traj.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")

# 360-degree spin video
mp4_path = "./traj_compare_outputs/human_robot_wrist_traj_spin.mp4"
spin_3d_and_save(fig, ax, mp4_path, n_frames=360, fps=30, elev=25, azim_start=30, azim_end=390)

plt.close(fig)
print("Saved →", png_path)
print("Saved →", mp4_path)