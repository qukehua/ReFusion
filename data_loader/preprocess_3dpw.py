"""
Preprocess 3DPW dataset to generate multimodal evaluation files.

This script generates:
1. data_candi_*.npz - Candidate trajectories for multimodal evaluation
2. t_his*_filtered.npz - Multimodal indices (same history, different future)

Usage:
    python preprocess_3dpw.py --data_path /data/user/qkh/datasets/3DPW --output_dir /data/user/qkh/datasets/3DPW/multimodal
"""

import os
import argparse
import pickle as pkl
from glob import glob

import numpy as np
import torch
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f, encoding="latin1")
    return data


def _parse_scene_filter(scene_filter):
    if scene_filter is None:
        return None
    if isinstance(scene_filter, str):
        s = scene_filter.strip().lower()
        if s in ("", "all", "none"):
            return None
        return {x.strip().lower() for x in scene_filter.split(",") if x.strip()}
    return {str(x).strip().lower() for x in scene_filter if str(x).strip()}


def load_3dpw_sequences(data_path, split="test", scene_filter=None, require_two_person=True):
    """
    Load 3DPW sequences from:
      data_path/sequenceFiles/{train|validation|test}/*.pkl

    Keep only two-person sequences if require_two_person=True.
    """
    split_map = {"train": "train", "val": "validation", "test": "test"}
    if split not in split_map:
        raise ValueError(f"Unsupported split '{split}', expected one of {list(split_map.keys())}.")

    split_dir = os.path.join(data_path, "sequenceFiles", split_map[split])
    pkl_files = sorted(glob(os.path.join(split_dir, "*.pkl")))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No 3DPW files found in {split_dir}")

    scene_set = _parse_scene_filter(scene_filter)

    all_sequences = []
    sequence_info = []
    for pkl_file in tqdm(pkl_files, desc=f"Loading 3DPW {split}"):
        stem = os.path.splitext(os.path.basename(pkl_file))[0]
        scene_name = stem.split("_")[0].lower()
        if scene_set is not None and scene_name not in scene_set:
            continue

        seq_dict = load_pkl(pkl_file)
        joints = seq_dict.get("jointPositions", None)
        if joints is None or len(joints) < 2:
            continue
        if require_two_person and len(joints) != 2:
            continue

        p1 = np.asarray(joints[0], dtype=np.float32).reshape(-1, 24, 3)
        p2 = np.asarray(joints[1], dtype=np.float32).reshape(-1, 24, 3)
        n_frames = min(p1.shape[0], p2.shape[0])
        p1 = p1[:n_frames]
        p2 = p2[:n_frames]
        seq = np.concatenate([p1, p2], axis=1)

        # Match Dataset3DPW root-relative representation.
        seq[:, 1:] -= seq[:, :1]

        all_sequences.append(seq)
        sequence_info.append(
            {
                "scene": scene_name,
                "file": os.path.basename(pkl_file),
                "length": n_frames,
            }
        )
    return all_sequences, sequence_info


def extract_windows(sequences, t_his, t_pred, skip_rate):
    t_total = t_his + t_pred
    windows = []
    window_origins = []
    for seq_idx, seq in enumerate(tqdm(sequences, desc="Extracting windows")):
        seq_len = seq.shape[0]
        for i in range(0, seq_len - t_total, skip_rate):
            windows.append(seq[i : i + t_total])
            window_origins.append((seq_idx, i))
    return np.array(windows), window_origins


def compute_multimodal_indices(windows, t_his, thre_his=0.5, thre_pred=0.1):
    n_windows = len(windows)
    if n_windows == 0:
        print("ERROR: No windows to process.")
        return {}

    history = windows[:, t_his - 1 : t_his, 1:].reshape(n_windows, -1)
    future = windows[:, t_his:, 1:].reshape(n_windows, -1)
    print(f"Computing pairwise distances for {n_windows} windows using {DEVICE}...")

    multimodal_dict = {}
    if USE_CUDA:
        history_t = torch.tensor(history, dtype=torch.float32, device=DEVICE)
        future_t = torch.tensor(future, dtype=torch.float32, device=DEVICE)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        estimated_mem_per_batch = n_windows * future.shape[1] * 4 * 2
        max_batch = max(1, int(gpu_mem * 0.3 / max(estimated_mem_per_batch, 1)))
        batch_size = min(200, max_batch)
        print(f"Using batch_size={batch_size} for {n_windows} windows")

        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CUDA)"):
            batch_end = min(i + batch_size, n_windows)
            try:
                hist_i = history_t[i:batch_end]
                fut_i = future_t[i:batch_end]
                dist_his = torch.norm(hist_i[:, None, :] - history_t[None, :, :], dim=2)
                dist_pred = torch.norm(fut_i[:, None, :] - future_t[None, :, :], dim=2)
                mask = (dist_his <= thre_his) & (dist_pred >= thre_pred)
                for j in range(batch_end - i):
                    idx = i + j
                    mask[j, idx] = False
                    idx_multi = torch.where(mask[j])[0].cpu().numpy().tolist()
                    if len(idx_multi) > 0:
                        multimodal_dict[idx] = idx_multi
                del dist_his, dist_pred, mask, hist_i, fut_i
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM at batch {i}, switching this batch to CPU.")
                    torch.cuda.empty_cache()
                    hist_i = history[i:batch_end]
                    fut_i = future[i:batch_end]
                    dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
                    dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
                    for j in range(batch_end - i):
                        idx = i + j
                        mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                        mask[idx] = False
                        idx_multi = np.where(mask)[0].tolist()
                        if len(idx_multi) > 0:
                            multimodal_dict[idx] = idx_multi
                else:
                    raise e
        del history_t, future_t
        torch.cuda.empty_cache()
    else:
        batch_size = 500
        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CPU)"):
            batch_end = min(i + batch_size, n_windows)
            hist_i = history[i:batch_end]
            fut_i = future[i:batch_end]
            dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
            dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
            for j in range(batch_end - i):
                idx = i + j
                mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                mask[idx] = False
                idx_multi = np.where(mask)[0]
                if len(idx_multi) > 0:
                    multimodal_dict[idx] = idx_multi.tolist()
    return multimodal_dict


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3DPW for multimodal evaluation")
    parser.add_argument("--data_path", type=str, default="/data/user/qkh/datasets/3DPW", help="Path to 3DPW root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user/qkh/datasets/3DPW/multimodal",
        help="Output directory for multimodal npz files",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to preprocess")
    parser.add_argument(
        "--scene_filter",
        type=str,
        default=all,
        help="Comma-separated scenes (e.g. courtyard,downtown). Use 'all' or empty string for no filtering (all scenes in that split).",
    )
    parser.add_argument("--require_two_person", action="store_true", help="Keep only exactly-2-person sequences")
    parser.add_argument("--t_his", type=int, default=25, help="History frames")
    parser.add_argument("--t_pred", type=int, default=100, help="Prediction frames")
    parser.add_argument("--skip_rate", type=int, default=20, help="Skip rate for extracting windows")
    parser.add_argument("--thre_his", type=float, default=0.5, help="History similarity threshold")
    parser.add_argument("--thre_pred", type=float, default=0.1, help="Future difference threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scene_filter = args.scene_filter
    scene_tag = "all" if scene_filter is None or scene_filter.strip() == "" else scene_filter.replace(",", "_")
    two_person = args.require_two_person

    print("=" * 60)
    print("3DPW Preprocessing for TransFusion")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"scene_filter: {scene_filter}")
    print(f"require_two_person: {two_person}")
    print(f"t_his: {args.t_his}, t_pred: {args.t_pred}, skip_rate: {args.skip_rate}")
    print(f"thre_his: {args.thre_his}, thre_pred: {args.thre_pred}")
    print("=" * 60)

    print("\n[1/4] Loading sequences...")
    sequences, seq_info = load_3dpw_sequences(
        args.data_path,
        split=args.split,
        scene_filter=scene_filter,
        require_two_person=two_person,
    )
    print(f"Loaded {len(sequences)} sequences")

    print("\n[2/4] Extracting sliding windows...")
    windows, _ = extract_windows(sequences, args.t_his, args.t_pred, args.skip_rate)
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows.shape}")

    print("\n[3/4] Saving candidate trajectories...")
    tag = "2p" if two_person else "allp"
    candi_file = os.path.join(
        args.output_dir,
        f"data_candi_3dpw_{args.split}_{scene_tag}_{tag}_t_his{args.t_his}_t_pred{args.t_pred}_skiprate{args.skip_rate}.npz",
    )
    np.savez_compressed(candi_file, **{"data_candidate.npy": windows})
    print(f"Saved: {candi_file}")

    print("\n[4/4] Computing multimodal indices...")
    multimodal_dict = compute_multimodal_indices(windows, args.t_his, args.thre_his, args.thre_pred)
    multi_file = os.path.join(
        args.output_dir,
        f"t_his{args.t_his}_3dpw_{args.split}_{scene_tag}_{tag}_thre{args.thre_his:.3f}_t_pred{args.t_pred}_thre{args.thre_pred:.3f}_filtered.npz",
    )
    np.savez_compressed(multi_file, data_multimodal=multimodal_dict)
    print(f"Saved: {multi_file}")

    n_multi = len(multimodal_dict)
    avg_multi = np.mean([len(v) for v in multimodal_dict.values()]) if n_multi > 0 else 0
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    if len(windows) > 0:
        print(f"Windows with multimodal futures: {n_multi}/{len(windows)} ({100 * n_multi / len(windows):.1f}%)")
    print(f"Average multimodal count: {avg_multi:.1f}")
    if len(seq_info) > 0:
        scenes = sorted(list(set([x["scene"] for x in seq_info])))
        print(f"Scenes used: {scenes}")
    print("=" * 60)

    print("\nUse these paths in cfg/3dpw.yml:")
    print(f"  multimodal_path: {multi_file}")
    print(f"  data_candi_path: {candi_file}")


if __name__ == "__main__":
    main()
