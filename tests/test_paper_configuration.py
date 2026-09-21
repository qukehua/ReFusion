"""Cross-check the shipped settings against manuscript IV-C."""

from pathlib import Path

import pytest
import yaml

from config import Config, update_config


@pytest.mark.parametrize('name,observed,predicted,target,condition', [
    ('harper3d_30hz', 25, 100, 20, 41),
    ('harper3d_120hz', 100, 400, 20, 41),
    ('chico', 10, 25, 14, 23),
    ('comad', 15, 15, 9, 11),
    ('3dpw', 25, 100, 23, 47),
    ('cmu_mocap', 25, 100, 38, 77),
])
def test_shipped_paper_settings(tmp_path, monkeypatch, name, observed, predicted, target, condition):
    source = Path(__file__).resolve().parents[1] / 'cfg' / (name + '.yml')
    text = source.read_text(encoding='utf-8')
    settings = yaml.safe_load(text)
    config_dir = tmp_path / 'cfg'
    config_dir.mkdir()
    (config_dir / source.name).write_text(text, encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    cfg = update_config(Config(name), {'cfg': name, 'mode': 'train', 'device': 'cpu'})
    assert (cfg.t_his, cfg.t_pred, cfg.n_pre) == (observed, predicted, predicted)
    assert (cfg.joint_num, cfg.cond_joint_num) == (target, condition)
    assert cfg.num_layers == 9 and cfg.stage1_num_layers == 2
    assert cfg.latent_dims == 512 and cfg.dropout == 0.2
    assert cfg.noise_steps == 1000 and cfg.scheduler == 'Cosine'
    assert cfg.num_epoch == 100 and cfg.num_data_sample == 50000
    assert cfg.lr == 3e-4 and cfg.gamma == 0.8
    assert cfg.milestone == [20, 40, 60, 80, 100]
    assert cfg.aug_rotate_prob == 0.5 and cfg.aug_reverse_prob == 0.3
    assert cfg.optimizer == 'AdamW' and not cfg.use_velocity_input
    assert cfg.mod_train == cfg.mod_test == 1 and cfg.velocity_loss_weight == 0
    assert settings['model_variant'] == 'two_stage' and not cfg.Complete
    if cfg.dataset == 'harper3d':
        assert settings['harper_spot_joint_indices'] == cfg.harper_spot_joint_indices == list(range(21))
