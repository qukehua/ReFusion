"""Small CPU fixtures for paper/code alignment regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.util import get_dct_matrix


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture
def tiny_cfg(tmp_path):
    # Deliberately use T != P and more observed joints than predicted joints.
    cfg = SimpleNamespace(
        t_his=3, t_pred=5, n_pre=5, joint_num=2, cond_joint_num=3,
        implementation_version='refusion_future_dct_v2',
        dataset="chico", drop_root_joint=False, predict_human_only=True,
        output_total_joints=2, use_robot_condition=True, use_velocity_input=False,
        model_variant="two_stage", num_layers=2, stage1_num_layers=1,
        num_heads=2, latent_dims=16, dropout=0.0,
        dit_attn_mode="spatio_temporal", device="cpu", dtype=torch.float32,
        noise_steps=11, ddim_timesteps=4, scheduler="Cosine", Complete=False,
        padding="LastFrame", mod_train=1.0, mod_test=1.0, vis_col=3,
        ema=False, lr=1e-3, weight_decay=0.05, adam_betas=(0.9, 0.999), adam_eps=1e-8,
        milestone=[10], gamma=0.5,
        batch_size=2, num_data_sample=2, num_val_data_sample=2, num_epoch=1,
        velocity_loss_weight=0.1, seed=11, save_model_interval=0,
        model_path=str(tmp_path),
    )
    cfg.dct_m_obs, cfg.idct_m_obs = [m.float() for m in get_dct_matrix(cfg.t_his)]
    cfg.dct_m_pred, cfg.idct_m_pred = [m.float() for m in get_dct_matrix(cfg.t_pred)]
    return cfg


@pytest.fixture
def raw_motion(tiny_cfg):
    return torch.randn(2, tiny_cfg.t_his + tiny_cfg.t_pred, 3, 3)
