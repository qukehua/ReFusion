"""Recompute a saved evaluation: python -m utils.rescore_evaluation RUN_DIR."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from utils.evaluation import METRIC_NAMES, write_summary
from utils.metrics import compute_all_metrics


def rescore(run_dir):
    root = Path(run_dir)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete':
        raise ValueError('Cannot report an incomplete evaluation as a complete run.')
    values, indices = [], []
    for path in sorted(root.glob('predictions_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            offsets = data['multimodal_offsets']
            for i, sample_index in enumerate(data['sample_indices']):
                pred = data['predictions'][i]
                if len(pred) != metadata['samples_per_observation']:
                    raise ValueError('Prediction count does not match the run metadata.')
                metrics = compute_all_metrics(
                    torch.from_numpy(pred), data['ground_truth'][i:i + 1],
                    data['multimodal_ground_truth'][offsets[i]:offsets[i + 1]],
                )
                values.append([float(v) for v in metrics])
                indices.append(int(sample_index))
    if indices != list(range(metadata['num_observations'])):
        raise ValueError('Prediction archives have missing, duplicated, or unordered windows.')
    means = np.asarray(values).mean(axis=0)
    with (root / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        saved = {row['Metric']: float(row['ReFusion']) for row in csv.DictReader(stream)}
    np.testing.assert_allclose(means, [saved[name] for name in METRIC_NAMES], rtol=1e-5, atol=1e-6)
    write_summary(root / 'summary_rescored.csv', means)
    return means


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    rescore(args.run_dir)
    print('Saved predictions reproduce summary.csv within numerical tolerance.')
