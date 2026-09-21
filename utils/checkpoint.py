"""Versioned checkpoints prevent silent reuse across different data contracts."""

import torch


CONTRACT_FIELDS = (
    'dataset', 't_his', 't_pred', 'n_pre', 'joint_num', 'cond_joint_num',
    'model_variant', 'num_layers', 'stage1_num_layers', 'latent_dims',
    'num_heads', 'dit_attn_mode', 'use_velocity_input', 'drop_root_joint',
    'predict_human_only', 'noise_steps', 'scheduler',
    'harper_spot_joint_indices', 'comad_p1_joint_indices', 'comad_robot_joint_indices',
    'use_spot_condition', 'use_robot_condition', 'use_partner_condition',
    'use_hr_robot_condition', 'use_hh_human_condition',
)


def model_contract(cfg):
    result = {name: getattr(cfg, name, None) for name in CONTRACT_FIELDS}
    result['drop_root_joint'] = getattr(cfg, 'drop_root_joint', True)
    return result


def save_checkpoint(path, model, cfg, epoch, validation_loss):
    torch.save({
        'implementation_version': cfg.implementation_version,
        'model_contract': model_contract(cfg),
        'state_dict': model.state_dict(), 'epoch': epoch,
        'validation_loss': float(validation_loss),
        'manifest_path': getattr(cfg, 'manifest_path', None),
    }, path)


def load_checkpoint(path, model, cfg):
    record = torch.load(path, map_location=cfg.device, weights_only=True)
    if not isinstance(record, dict) or record.get('implementation_version') != cfg.implementation_version:
        raise ValueError('Checkpoint has no matching implementation version. Retrain with the aligned '
                         'code; unversioned historical weights cannot certify this paper/code contract.')
    expected = model_contract(cfg)
    if record.get('model_contract') != expected:
        different = [key for key, value in expected.items()
                     if record.get('model_contract', {}).get(key) != value]
        raise ValueError(f'Checkpoint configuration mismatch: {different}.')
    model.load_state_dict(record['state_dict'], strict=True)
    return record
