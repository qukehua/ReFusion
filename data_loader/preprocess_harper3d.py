"""
Preprocess Harper3D dataset to generate multimodal evaluation files.
Similar to the preprocessing done for H36M and HumanEva in GSPS.

This script generates:
1. data_candi_*.npz - Candidate trajectories for multimodal evaluation
2. t_his*_filtered.npz - Multimodal indices (same history, different future)

Usage:
    python preprocess_harper3d.py --data_path ./data/harper3d --output_dir ./data/harper3d_multi_modal
"""

import os
import argparse
import numpy as np
from glob import glob
import pickle as pkl
from tqdm import tqdm
import torch

# Check CUDA availability
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(f"Using device: {DEVICE}")


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


def load_harper3d_sequences(data_path, split, include_spot=True):
    """Load all sequences from Harper3D dataset."""
    data_folder = os.path.join(data_path, split)
    pkl_files = glob(os.path.join(data_folder, "*.pkl"))
    
    all_sequences = []
    sequence_info = []
    
    for pkl_file in tqdm(pkl_files, desc=f"Loading {split} data"):
        sequence_dict = load_pkl(pkl_file)
        
        first_frame = sequence_dict[0]
        subject = first_frame['subject']
        action = first_frame['action']
        
        frames = sorted(sequence_dict.keys())
        human_seq = np.array([sequence_dict[f]['human_joints_3d'] for f in frames])
        
        if include_spot:
            spot_seq = np.array([sequence_dict[f]['spot_joints_3d'] for f in frames])
            seq = np.concatenate([human_seq, spot_seq], axis=1)
        else:
            seq = human_seq
        
        # Make relative to root
        seq[:, 1:] -= seq[:, :1]
        
        all_sequences.append(seq)
        sequence_info.append({
            'subject': subject,
            'action': action,
            'file': os.path.basename(pkl_file),
            'length': len(frames)
        })
    
    return all_sequences, sequence_info


def extract_windows(sequences, t_his, t_pred, skip_rate):
    """Extract sliding windows from all sequences."""
    t_total = t_his + t_pred
    windows = []
    window_origins = []
    
    for seq_idx, seq in enumerate(tqdm(sequences, desc="Extracting windows")):
        seq_len = seq.shape[0]
        for i in range(0, seq_len - t_total, skip_rate):
            window = seq[i:i + t_total]
            windows.append(window)
            window_origins.append((seq_idx, i))
    
    return np.array(windows), window_origins


def compute_multimodal_indices(windows, t_his, thre_his=0.5, thre_pred=0.1):
    """
    Find multimodal pairs: same history, different future.
    Uses CUDA if available for faster computation.
    
    For each window, find other windows that have:
    - Similar history (distance < thre_his)
    - Different future (distance >= thre_pred)
    """
    n_windows = len(windows)
    
    if n_windows == 0:
        print("ERROR: No windows to process. Check your data path and directory structure.")
        print("Expected structure:")
        print("  data_path/")
        print("  ├── train/")
        print("  │   └── *.pkl")
        print("  └── test/")
        print("      └── *.pkl")
        return {}
    
    t_pred = windows.shape[1] - t_his
    
    # Extract history and future parts (excluding root joint)
    history = windows[:, t_his-1:t_his, 1:].reshape(n_windows, -1)  # Last frame of history
    future = windows[:, t_his:, 1:].reshape(n_windows, -1)
    
    print(f"Computing pairwise distances for {n_windows} windows using {DEVICE}...")
    
    multimodal_dict = {}
    
    if USE_CUDA:
        # Use PyTorch + CUDA for faster computation
        history_t = torch.tensor(history, dtype=torch.float32, device=DEVICE)
        future_t = torch.tensor(future, dtype=torch.float32, device=DEVICE)
        
        # Dynamically adjust batch size based on data size to avoid OOM
        # Estimate memory: batch * n_windows * D * 4 bytes * 2 (for two distance matrices)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        estimated_mem_per_batch = n_windows * future.shape[1] * 4 * 2  # bytes per batch item
        max_batch = max(1, int(gpu_mem * 0.3 / estimated_mem_per_batch))  # Use 30% of GPU memory
        batch_size = min(200, max_batch)  # Cap at 200 for safety
        
        print(f"Using batch_size={batch_size} for {n_windows} windows")
        
        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CUDA)"):
            batch_end = min(i + batch_size, n_windows)
            
            try:
                hist_i = history_t[i:batch_end]  # (batch, D)
                fut_i = future_t[i:batch_end]
                
                # Compute distances using torch
                dist_his = torch.norm(hist_i[:, None, :] - history_t[None, :, :], dim=2)
                dist_pred = torch.norm(fut_i[:, None, :] - future_t[None, :, :], dim=2)
                
                # Create mask and process
                mask = (dist_his <= thre_his) & (dist_pred >= thre_pred)
                
                for j in range(batch_end - i):
                    idx = i + j
                    mask[j, idx] = False  # Exclude self
                    multimodal_indices = torch.where(mask[j])[0].cpu().numpy().tolist()
                    
                    if len(multimodal_indices) > 0:
                        multimodal_dict[idx] = multimodal_indices
                
                # Clear intermediate tensors
                del dist_his, dist_pred, mask, hist_i, fut_i
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM at batch {i}, falling back to CPU for remaining...")
                    torch.cuda.empty_cache()
                    # Fall back to CPU for this batch
                    hist_i = history[i:batch_end]
                    fut_i = future[i:batch_end]
                    dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
                    dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
                    
                    for j in range(batch_end - i):
                        idx = i + j
                        mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                        mask[idx] = False
                        multimodal_indices = np.where(mask)[0].tolist()
                        if len(multimodal_indices) > 0:
                            multimodal_dict[idx] = multimodal_indices
                else:
                    raise e
        
        # Clean up GPU memory
        del history_t, future_t
        torch.cuda.empty_cache()
    else:
        # Fallback to NumPy (CPU)
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
                multimodal_indices = np.where(mask)[0]
                
                if len(multimodal_indices) > 0:
                    multimodal_dict[idx] = multimodal_indices.tolist()
    
    return multimodal_dict


def main():
    parser = argparse.ArgumentParser(description='Preprocess Harper3D for multimodal evaluation')
    parser.add_argument('--data_path', type=str, default='/data3/user/qkh/HARPER/HAPER_robot view',
                        help='Path to Harper3D dataset root')
    parser.add_argument('--fps', type=str, default='30hz', choices=['30hz', '120hz'],
                        help='Frame rate version: 30hz or 120hz')
    parser.add_argument('--output_dir', type=str, default='/data3/user/qkh/DATASET/TransFusion/HARPER',
                        help='Output directory for preprocessed files')
    parser.add_argument('--t_his', type=int, default=25, help='History frames')
    parser.add_argument('--t_pred', type=int, default=100, help='Prediction frames')
    parser.add_argument('--skip_rate', type=int, default=20, help='Skip rate for extracting windows')
    parser.add_argument('--thre_his', type=float, default=0.5, help='History similarity threshold')
    parser.add_argument('--thre_pred', type=float, default=0.1, help='Future difference threshold')
    parser.add_argument('--include_spot', action='store_true', help='Include Spot robot joints')
    args = parser.parse_args()
    
    # Construct actual data path with fps subdirectory
    actual_data_path = os.path.join(args.data_path, args.fps)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Harper3D Preprocessing for TransFusion")
    print("=" * 60)
    print(f"Data root: {args.data_path}")
    print(f"FPS version: {args.fps}")
    print(f"Actual data path: {actual_data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"t_his: {args.t_his}, t_pred: {args.t_pred}")
    print(f"skip_rate: {args.skip_rate}")
    print(f"thre_his: {args.thre_his}, thre_pred: {args.thre_pred}")
    print(f"include_spot: {args.include_spot}")
    print("=" * 60)
    
    # Check if data path exists
    if not os.path.exists(actual_data_path):
        print(f"\nERROR: Data path does not exist: {actual_data_path}")
        print(f"Available directories in {args.data_path}:")
        if os.path.exists(args.data_path):
            for item in os.listdir(args.data_path):
                print(f"  - {item}")
        exit(1)
    
    # Load test data (multimodal evaluation is done on test set)
    print("\n[1/4] Loading test sequences...")
    sequences, seq_info = load_harper3d_sequences(
        actual_data_path, 'test', include_spot=args.include_spot
    )
    print(f"Loaded {len(sequences)} sequences")
    
    # Extract windows
    print("\n[2/4] Extracting sliding windows...")
    windows, window_origins = extract_windows(
        sequences, args.t_his, args.t_pred, args.skip_rate
    )
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows.shape}")
    
    # Save candidate trajectories (include fps in filename)
    print("\n[3/4] Saving candidate trajectories...")
    candi_file = os.path.join(
        args.output_dir,
        f'data_candi_{args.fps}_t_his{args.t_his}_t_pred{args.t_pred}_skiprate{args.skip_rate}.npz'
    )
    np.savez_compressed(candi_file, **{'data_candidate.npy': windows})
    print(f"Saved: {candi_file}")
    
    # Compute multimodal indices
    print("\n[4/4] Computing multimodal indices...")
    multimodal_dict = compute_multimodal_indices(
        windows, args.t_his, args.thre_his, args.thre_pred
    )
    
    # Convert to the format expected by TransFusion
    # Original format: data_multimodal[subject][action][frame_idx] = indices
    # Simplified format for Harper3D: just use window index directly
    
    multi_file = os.path.join(
        args.output_dir,
        f't_his{args.t_his}_{args.fps}_thre{args.thre_his:.3f}_t_pred{args.t_pred}_thre{args.thre_pred:.3f}_filtered.npz'
    )
    np.savez_compressed(multi_file, data_multimodal=multimodal_dict)
    print(f"Saved: {multi_file}")
    
    # Statistics
    n_multi = len(multimodal_dict)
    avg_multi = np.mean([len(v) for v in multimodal_dict.values()]) if n_multi > 0 else 0
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print(f"Windows with multimodal futures: {n_multi}/{len(windows)} ({100*n_multi/len(windows):.1f}%)")
    print(f"Average multimodal count: {avg_multi:.1f}")
    print("=" * 60)
    
    # Update config hint
    print("\nUpdate your cfg/harper3d.yml with:")
    print(f"  multimodal_path: {multi_file}")
    print(f"  data_candi_path: {candi_file}")


if __name__ == '__main__':
    main()
