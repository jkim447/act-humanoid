def compute_delta_action_norm_stats_from_dataset_configs(
    ds_configs,                    # list of (ds, chunk_size, stride)
    out_path=None,                 # ".npz" or ".json"
    eps=1e-8,
    print_literal=True,
    var_name="COMBINED_DELTA_NORM_STATS",
    max_episodes_per_dataset=None, # e.g., 100; if None, use all
    seed=0,
):
    import os, json
    import numpy as np, pandas as pd

    rng = np.random.default_rng(seed)

    all_actions, all_qpos = [], []

    for ds, chunk_size, stride in ds_configs:
        episode_dirs = list(getattr(ds, "episode_dirs", []))
        if max_episodes_per_dataset is not None and len(episode_dirs) > max_episodes_per_dataset:
            idx = rng.choice(len(episode_dirs), size=max_episodes_per_dataset, replace=False)
            episode_dirs = [episode_dirs[i] for i in idx]

        for demo_dir in episode_dirs:
            csv_path = os.path.join(demo_dir, "ee_hand.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            T = len(df)
            if T == 0:
                continue

            for s in range(T):
                a0_abs = ds._row_to_action(df.iloc[s]).astype(np.float32)
                all_qpos.append(a0_abs)

                wrist_anchor = a0_abs[0:3].copy()
                end_ts = min(s + chunk_size * stride, T)
                for t in range(s, end_ts, stride):
                    a_t = ds._row_to_action(df.iloc[t]).astype(np.float32)
                    a_t[0:3] -= wrist_anchor
                    all_actions.append(a_t)

    if not all_actions or not all_qpos:
        raise ValueError("No actions/qpos collected — check inputs.")

    actions_np = np.vstack(all_actions)
    qpos_np    = np.vstack(all_qpos)

    norm_stats = {
        "action_mean": actions_np.mean(0).astype(np.float32),
        "action_std":  np.clip(actions_np.std(0, ddof=1), eps, None).astype(np.float32),
        "qpos_mean":   qpos_np.mean(0).astype(np.float32),
        "qpos_std":    np.clip(qpos_np.std(0, ddof=1), eps, None).astype(np.float32),
    }

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if out_path.endswith(".npz"):
            np.savez_compressed(out_path, **norm_stats)
        elif out_path.endswith(".json"):
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, open(out_path, "w"))
        else:
            raise ValueError("out_path must end with .npz or .json")
        print(f"Saved combined delta normalization stats to: {out_path}")

    if print_literal:
        am, sd = norm_stats["action_mean"].tolist(), norm_stats["action_std"].tolist()
        qm, qs = norm_stats["qpos_mean"].tolist(),  norm_stats["qpos_std"].tolist()
        print(f"{var_name} = {{")
        print(f"    'action_mean': np.array({am}, dtype=np.float32),")
        print(f"    'action_std':  np.array({sd}, dtype=np.float32),")
        print(f"    'qpos_mean':   np.array({qm}, dtype=np.float32),")
        print(f"    'qpos_std':    np.array({qs}, dtype=np.float32),")
        print("}")

    return norm_stats

from galaxea_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import GalaxeaDatasetKeypointsJoints
from human_dataset_pick_cube_into_box_right_hand_keypoints_and_joints import HumanDatasetKeypointsJoints
##########################################################################
##########################################################################
# TODO: MAKE SURE TO SET NORMALIZE PARAMS TO FALSE! SINCE WE ARE COMPUTING STATS HERE
# TODO: VERITFY THE CHUNKSIZE AND THE STRIDE VALUES!
##########################################################################
##########################################################################
robot_dir1 = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"
human_dir1 = "/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox"
human_dir2 = "/iris/projects/humanoid/hamer/keypoint_human_data_red_outbox"
human_dir3 = "/iris/projects/humanoid/hamer/keypoint_human_data_wood_inbox"
# TODO: change accordingly
max_episodes_per_dataset = 35
# TODO: robot chunks are set assuming robot is twice as slow
robot_chunksize = 45
robot_stride = 2

human_chunksize = 45
human_stride = 1


apply_data_aug = False
normalize = False

ds_robot = GalaxeaDatasetKeypointsJoints(
    dataset_dir=robot_dir1,
    chunk_size=robot_chunksize,
    stride=robot_stride,
    apply_data_aug=apply_data_aug,
    normalize=normalize,
    compute_keypoints=True,
    overlay_keypoints=False   # skeletons drawn before resizing
)

ds_human1 = HumanDatasetKeypointsJoints(
        dataset_dir=human_dir1,
        chunk_size=human_chunksize,
        stride=human_stride,
        apply_data_aug=apply_data_aug,   # start with no aug for repeatability
        normalize=normalize         # raw for debugging
    )
    
ds_human2 = HumanDatasetKeypointsJoints(
        dataset_dir=human_dir2,
        chunk_size=human_chunksize,
        stride=human_stride,
        apply_data_aug=apply_data_aug,   # start with no aug for repeatability
        normalize=normalize         # raw for debugging
    )

ds_human3 = HumanDatasetKeypointsJoints(
    dataset_dir=human_dir3,
    chunk_size=human_chunksize,
    stride=human_stride,
    apply_data_aug=apply_data_aug,   # start with no aug for repeatability
    normalize=normalize         # raw for debugging
)

cfgs = [
    (ds_robot, robot_chunksize, robot_stride),
    (ds_human1, human_chunksize, human_stride),
    (ds_human2, human_chunksize, human_stride),
    (ds_human3, human_chunksize, human_stride),
]

stats = compute_delta_action_norm_stats_from_dataset_configs(
    cfgs,
    out_path="norm_stats_combined_human_robot_data.npz",
    max_episodes_per_dataset=max_episodes_per_dataset,  # cap per-dataset; set None to use all
    seed=42
)