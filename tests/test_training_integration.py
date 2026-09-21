"""An actual CPU training/validation epoch with synthetic joint trajectories."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.script import create_model_and_diffusion
from utils.training import Trainer


class SyntheticDataset:
    def __init__(self, batch):
        self.batch = batch

    def sampling_generator(self, num_samples, batch_size, aug=True):
        for _ in range(num_samples // batch_size):
            yield self.batch[:batch_size].copy()


@pytest.mark.parametrize("ema", [False, True])
def test_adamw_training_and_validation_use_future_only_latents(tiny_cfg, raw_motion, monkeypatch, ema):
    tiny_cfg.ema = ema
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    dataset = SyntheticDataset(raw_motion.numpy())
    trainer = Trainer(model, diffusion, {"train": dataset, "test": dataset}, tiny_cfg,
                      logging.getLogger("alignment-test"), tb_logger=None)
    noising_inputs = []
    original_noise_motion = diffusion.noise_motion

    def record_noising(clean, t):
        noising_inputs.append(clean.detach().clone())
        return original_noise_motion(clean, t)

    monkeypatch.setattr(diffusion, "noise_motion", record_noising)
    conditions = []
    handle = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: conditions.append(kwargs["mod"].detach().clone()), with_kwargs=True)
    ema_handle = None if not ema else trainer.ema_model.register_forward_pre_hook(
        lambda _module, _args, kwargs: conditions.append(kwargs['mod'].detach().clone()), with_kwargs=True)
    before = model.final_layer.out_proj.weight.detach().clone()
    try:
        trainer.loop()
    finally:
        handle.remove()
        if ema_handle is not None:
            ema_handle.remove()

    assert type(trainer.optimizer) is torch.optim.AdamW
    assert trainer.optimizer.param_groups[0]["weight_decay"] == tiny_cfg.weight_decay
    assert not torch.equal(before, model.final_layer.out_proj.weight)
    assert np.isfinite(trainer.train_losses.avg) and trainer.train_losses.avg > 0
    assert np.isfinite(trainer.val_losses.avg) and trainer.val_losses.avg > 0
    assert np.isfinite(trainer.train_velocity_losses.avg)
    assert len(noising_inputs) == 2 and len(conditions) == 2  # train and validation
    for latent, observed in zip(noising_inputs, conditions):
        assert latent.shape == (2, tiny_cfg.t_pred, 6)
        assert observed.shape == (2, tiny_cfg.t_his, 9)
        torch.testing.assert_close(tiny_cfg.idct_m_pred @ latent, raw_motion[:, tiny_cfg.t_his:, :2].flatten(-2))
        torch.testing.assert_close(tiny_cfg.idct_m_obs @ observed, raw_motion[:, :tiny_cfg.t_his].flatten(-2))
    expected_checkpoint = "best_ema.pt" if ema else "best.pt"
    assert (Path(tiny_cfg.model_path) / expected_checkpoint).is_file()


def test_optimizer_applies_decoupled_weight_decay(tiny_cfg):
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    trainer = Trainer(model, diffusion, {}, tiny_cfg, logging.getLogger("alignment-test"), None)
    trainer.before_train()
    parameter = model.motion_joint_embed.weight
    original = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    trainer.optimizer.step()
    torch.testing.assert_close(parameter, original * (1 - tiny_cfg.lr * tiny_cfg.weight_decay))
