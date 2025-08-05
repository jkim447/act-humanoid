import os
import json
import numpy as np
import pandas as pd

def compute_delta_action_norm_stats_from_dirs(
    ds,
    chunk_size=45,          # should match ds.chunk_size
    stride=2,               # should match the skip in your dataset
    out_path=None,          # ".npz" or ".json"
    eps=1e-8,
    print_literal=False,
    var_name="GALAXEA_DELTA_NORM_STATS",
):
    """
    Collect *all* relative wrist-translation action vectors, stack, then
    compute per-dimension mean and std (no streaming algorithm).

    For every episode and every anchor timestep s, we:
      • take frames t = s, s+stride, …, s+chunk_size*stride (clipped to len(df))
      • build the action vector with ds._row_to_action
      • subtract anchor left/right wrist positions (indices 0:3 and 9:12)

    Returns
    -------
    norm_stats : dict with float32 numpy arrays
    """
    action_list = []

    for demo_dir in ds.episode_dirs:
        csv_path = os.path.join(demo_dir, "ee_pos", "ee_poses_and_hands.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        T = len(df)
        if T == 0:
            continue

        for s in range(T):
            anchor_vec   = ds._row_to_action(df.iloc[s])           # (A,)
            left_anchor  = anchor_vec[0:3]
            right_anchor = anchor_vec[9:12]

            end_ts = min(s + chunk_size * stride, T)               # exclusive
            for t in range(s, end_ts, stride):
                row_vec = ds._row_to_action(df.iloc[t]).copy()

                # relative translation
                row_vec[0:3]  -= left_anchor
                row_vec[9:12] -= right_anchor

                action_list.append(row_vec.astype(np.float32))

    if not action_list:
        raise ValueError("No actions collected — check dataset paths.")

    actions_np   = np.vstack(action_list)          # (N_total, A)
    action_mean  = actions_np.mean(axis=0)
    action_std   = actions_np.std(axis=0, ddof=1)
    action_std   = np.clip(action_std, a_min=eps, a_max=None)

    norm_stats = {
        "action_mean": action_mean.astype(np.float32),
        "action_std":  action_std.astype(np.float32),
        "qpos_mean":   np.zeros_like(action_mean, dtype=np.float32),
        "qpos_std":    np.ones_like(action_mean,  dtype=np.float32),
    }

    # ---- optional save ----
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if out_path.endswith(".npz"):
            np.savez_compressed(out_path,
                                action_mean=norm_stats["action_mean"],
                                action_std=norm_stats["action_std"],
                                qpos_mean=norm_stats["qpos_mean"],
                                qpos_std=norm_stats["qpos_std"])
        elif out_path.endswith(".json"):
            json.dump({k: v.tolist() for k, v in norm_stats.items()},
                      open(out_path, "w"))
        else:
            raise ValueError("out_path must end with .npz or .json")
        print(f"Saved delta normalization stats to: {out_path}")

    # ---- optional literal ----
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


def compute_action_norm_stats_from_dirs(ds, out_path=None, eps=1e-8, print_literal=False, var_name="GALAXEA_NORM_STATS"):
    """
    Aggregate actions from all episodes, stack once, then compute per-dim mean/std.
    Optionally saves to .npz or .json and can print a copy-pasteable Python literal.

    Args:
        ds: GalaxeaDataset instance (uses ds.episode_dirs and ds._row_to_action)
        out_path: if given, save to .npz or .json
        eps: small value to avoid zeros in std
        print_literal: if True, print a Python variable you can paste in code
        var_name: variable name used when print_literal=True

    Returns:
        norm_stats dict with float32 numpy arrays
    """
    actions = []

    for demo_dir in ds.episode_dirs:
        csv_path = os.path.join(demo_dir, "ee_pos", "ee_poses_and_hands.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
        for _, row in df.iterrows():
            actions.append(ds._row_to_action(row))  # (A,)

    if not actions:
        raise ValueError("No actions found across episodes.")

    actions_np = np.vstack(actions).astype(np.float32)  # (N, A)

    action_mean = actions_np.mean(axis=0)                     # (A,)
    action_std  = actions_np.std(axis=0, ddof=1)              # (A,)
    action_std  = np.clip(action_std, a_min=eps, a_max=None)  # avoid zeros

    norm_stats = {
        "action_mean": action_mean.astype(np.float32),
        "action_std":  action_std.astype(np.float32),
        "qpos_mean":   np.zeros_like(action_mean, dtype=np.float32),
        "qpos_std":    np.ones_like(action_mean,  dtype=np.float32),
    }

    # Save if requested
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if out_path.endswith(".npz"):
            np.savez_compressed(out_path,
                                action_mean=norm_stats["action_mean"],
                                action_std=norm_stats["action_std"],
                                qpos_mean=norm_stats["qpos_mean"],
                                qpos_std=norm_stats["qpos_std"])
        elif out_path.endswith(".json"):
            payload = {k: v.tolist() for k, v in norm_stats.items()}
            with open(out_path, "w") as f:
                json.dump(payload, f)
        else:
            raise ValueError("out_path must end with .npz or .json")
        print(f"Saved normalization stats to: {out_path}")

    # Optional: print a copy-pasteable Python literal
    if print_literal:
        am = norm_stats["action_mean"].tolist()
        sd = norm_stats["action_std"].tolist()
        qm = norm_stats["qpos_mean"].tolist()
        qs = norm_stats["qpos_std"].tolist()
        print(f"{var_name} = {{")
        print(f"    'action_mean': np.array({am}, dtype=np.float32),")
        print(f"    'action_std':  np.array({sd}, dtype=np.float32),")
        print(f"    'qpos_mean':   np.array({qm}, dtype=np.float32),")
        print(f"    'qpos_std':    np.array({qs}, dtype=np.float32),")
        print("}")

    return norm_stats
