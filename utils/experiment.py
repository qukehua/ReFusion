"""Record the exact implementation/configuration used for each experiment."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_json_value(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return str(value)


def save_experiment_manifest(cfg, model):
    """Save a timestamped manifest, including source hashes and resolved overrides.

    DCT matrices are deterministic functions of the recorded T/P lengths and are
    excluded from JSON. Existing experiment records are never overwritten.
    """
    root = Path(__file__).resolve().parents[1]
    source_files = [root / 'main.py', root / 'config.py', root / 'cfg' / (cfg.id + '.yml')]
    for folder in ('models', 'utils', 'data_loader'):
        source_files.extend(sorted((root / folder).glob('*.py')))
    config = {
        key: _json_value(value) for key, value in vars(cfg).items()
        if not key.startswith(('dct_m', 'idct_m')) and key not in ('idx_pad', 'zero_index')
    }
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_version': cfg.implementation_version,
        'config': config,
        'model_class': type(model).__name__,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'tensor_shapes': {
            'observation_dct': ['B', cfg.t_his, 3 * cfg.cond_joint_num],
            'future_dct': ['B', cfg.n_pre, 3 * cfg.joint_num],
            'predicted_future': ['B', cfg.t_pred, 3 * cfg.joint_num],
        },
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'device': str(cfg.device)},
        'source_sha256': {str(path.relative_to(root)).replace('\\', '/'):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        'checkpoint': None,
    }
    checkpoint = getattr(cfg, 'resume', None) if cfg.mode == 'train' else getattr(cfg, 'ckpt', None)
    if checkpoint and Path(checkpoint).is_file():
        digest = hashlib.sha256()
        with open(checkpoint, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        record['checkpoint'] = {'path': str(Path(checkpoint).resolve()), 'sha256': digest.hexdigest()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(cfg.cfg_dir) / ('manifest_' + cfg.mode + '_' + stamp + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    return str(path)
