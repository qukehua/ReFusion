"""Future-only evaluation with an immutable, replayable record per run."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.experiment import _json_value
from utils.metrics import compute_all_metrics
from utils.script import sample_preprocessing
from utils.motion_latent import prepare_motion_inputs, decode_future_motion


METRIC_NAMES = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'ADE-m', 'FDE-m',
                'MMADE-m', 'MMFDE-m', 'ADE-w', 'FDE-w', 'MMADE-w', 'MMFDE-w']


def write_summary(path, values):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Metric', 'ReFusion'])
        writer.writeheader()
        writer.writerows({'Metric': name, 'ReFusion': value} for name, value in zip(METRIC_NAMES, values))


def compute_stats(diffusion, multimodal_dict, model, logger, cfg, wandb_logger=None):
    """Archive K futures per observation before reducing metrics.

    Distances use flattened xyz joint vectors. Median metrics use the lower
    median for even K (torch.median), matching the historical implementation.
    Archives can be rescored without a model, private loader, or original data.
    """
    count = multimodal_dict['num_samples']
    k = getattr(cfg, 'eval_samples', 50)
    if not isinstance(k, int) or k < 1 or count < 1:
        raise ValueError('Evaluation requires positive sample and observation counts.')
    chunk_size = max(1, min(int(getattr(cfg, 'eval_batch_size', cfg.batch_size)), count))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = Path(cfg.result_dir) / ('evaluation_' + stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'status': 'incomplete', 'num_observations': count, 'samples_per_observation': k,
        'implementation_version': getattr(cfg, 'implementation_version', 'unknown'),
        'manifest_path': getattr(cfg, 'manifest_path', None),
        'config': {name: _json_value(value) for name, value in vars(cfg).items()
                   if not name.startswith(('dct_m', 'idct_m'))},
        'metric_protocol': {
            'distance': 'L2 over flattened target joint xyz, per frame; no joint averaging',
            'APD': 'pairwise L2 over the whole future, averaged over unordered sample pairs',
            'multimodal_neighbors': 'test-window target poses at the last observed frame; includes self',
            'multimodal_threshold': getattr(cfg, 'multimodal_threshold', None),
            'skeleton_scaling': False, 'median': 'lower median for even K',
            'reduction': 'min/median/max over K for each GT, then mean over multimodal GT and windows',
        },
        'window_ids': _json_value(multimodal_dict.get('window_ids', list(range(count)))),
        'neighbor_indices': [np.asarray(x).tolist() for x in multimodal_dict.get('neighbor_indices', [])],
    }
    metadata_path = run_dir / 'metadata.json'
    manifest = getattr(cfg, 'manifest_path', None)
    if manifest and Path(manifest).is_file():
        metadata['experiment_manifest'] = json.loads(Path(manifest).read_text(encoding='utf-8'))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    was_training = model.training if model is not None else None
    if model is not None:
        model.eval()
    logger.info('Evaluation: observations=%s, K=%s, chunk=%s, archive=%s', count, k, chunk_size, run_dir)
    try:
        with torch.no_grad():
            for start in tqdm(range(0, count, chunk_size), desc='Eval chunks'):
                end = min(start + chunk_size, count)
                observed = multimodal_dict['data_group'][start:end, :cfg.t_his]
                target, condition = prepare_motion_inputs(observed, cfg)
                target = torch.as_tensor(target, device=cfg.device, dtype=torch.float32)
                condition = torch.as_tensor(condition, device=cfg.device, dtype=torch.float32)
                predictions = []
                for _ in range(k):
                    options, unused, encoded = sample_preprocessing(target, cfg, 'metrics', traj_cond=condition)
                    future_dct = diffusion.sample_ddim(model, unused, encoded, options)
                    decoded = decode_future_motion(future_dct, observed, cfg)[:, cfg.t_his:]
                    predictions.append(decoded.cpu().numpy())
                predictions = np.stack(predictions, axis=1)  # [observations,K,P,D]
                gt = multimodal_dict['gt_group'][start:end]
                gt_multi = multimodal_dict['traj_gt_arr'][start:end]
                offsets = np.cumsum([0] + [len(x) for x in gt_multi])
                np.savez_compressed(
                    run_dir / f'predictions_{start:08d}_{end:08d}.npz',
                    sample_indices=np.arange(start, end), observations=observed,
                    predictions=predictions, ground_truth=gt,
                    multimodal_ground_truth=np.concatenate(gt_multi), multimodal_offsets=offsets,
                )
                for i in range(end - start):
                    values = compute_all_metrics(torch.from_numpy(predictions[i]).to(cfg.device),
                                                 gt[i:i + 1], gt_multi[i])
                    rows.append([start + i] + [float(value) for value in values])
    finally:
        if model is not None:
            model.train(was_training)

    with (run_dir / 'per_sample.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index'] + METRIC_NAMES)
        writer.writerows(rows)
    means = np.asarray(rows)[:, 1:].mean(axis=0)
    write_summary(run_dir / 'summary.csv', means)
    # Convenience snapshots; the timestamped directory is the authoritative record.
    write_summary(Path(cfg.result_dir) / 'stats_latest.csv', means)
    write_summary(Path(cfg.result_dir) / 'stats.csv', means)
    metadata['status'] = 'complete'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, value in zip(METRIC_NAMES, means):
        logger.info('%s: ReFusion: %.6f', name, value)
    if wandb_logger is not None:
        metrics = {'eval/' + name: value for name, value in zip(METRIC_NAMES, means)}
        wandb_logger.log(metrics)
        wandb_logger.summary.update(metrics)
    return str(run_dir)
