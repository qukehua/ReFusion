"""DDIM boundary checks and sampling without access to a future target."""

import numpy as np
import pytest
import torch
from torch import nn

from models.diffusion import Diffusion
from utils.motion_latent import prepare_motion_inputs


def small_diffusion(ddim_steps=4):
    return Diffusion(noise_steps=11, ddim_timesteps=ddim_steps, scheduler="Cosine",
                     device="cpu", motion_size=(5, 6), n_pre=5)


class OracleNoise(nn.Module):
    def __init__(self, diffusion, clean):
        super().__init__()
        self.diffusion = diffusion
        self.clean = clean
        self.visited = []

    def forward(self, x, t, mod=None):
        self.visited.append(int(t[0]))
        alpha = self.diffusion.alpha_hat[t][:, None, None]
        return (x - alpha.sqrt() * self.clean) / (1 - alpha).sqrt()


@pytest.mark.parametrize("ddim_steps", [1, 4, 11])
def test_ddim_oracle_reaches_clean_future_for_all_step_layouts(ddim_steps):
    diffusion = small_diffusion(ddim_steps)
    assert len(diffusion.alpha_hat) == 12
    assert diffusion.alpha_hat[0] == 1
    assert len(np.unique(diffusion.ddim_timestep_seq)) == ddim_steps
    assert diffusion.ddim_timestep_seq[-1] == diffusion.noise_steps
    assert diffusion.ddim_timestep_prev_seq[0] == 0
    clean = torch.randn(2, 5, 6, dtype=torch.float64)
    terminal = torch.full((2,), diffusion.noise_steps, dtype=torch.long)
    initial, _ = diffusion.noise_motion(clean, terminal)
    oracle = OracleNoise(diffusion, clean)
    stages = list(diffusion.sample_ddim_progressive(
        oracle, None, None, {"sample_num": 2}, noise=initial))

    assert len(stages) == ddim_steps
    assert oracle.visited == diffusion.ddim_timestep_seq[::-1].tolist()
    assert all(stage.shape == clean.shape and torch.isfinite(stage).all() for stage in stages)
    torch.testing.assert_close(stages[-1], clean, rtol=1e-10, atol=1e-10)


def test_forward_process_has_a_clean_origin_and_trains_on_terminal_step():
    diffusion = small_diffusion()
    clean = torch.randn(2, 5, 6)
    noised, _ = diffusion.noise_motion(clean, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(noised, clean, rtol=0, atol=0)
    indices = diffusion.sample_timesteps(4096)
    assert set(indices.tolist()) == set(range(1, diffusion.noise_steps + 1))


class ObservedConditionNoise(nn.Module):
    def forward(self, x, t, mod=None):
        assert mod.shape[1:] == (3, 9)
        return 0.01 * mod.mean(dim=1)[:, None, :6].expand_as(x)


@pytest.mark.parametrize("mode", ["metrics", "pred"])
def test_sampling_with_fixed_noise_is_independent_of_ground_truth_future(tiny_cfg, raw_motion, mode):
    from utils.script import sample_preprocessing

    if mode == "pred":
        raw_motion = raw_motion[:1]
    changed = raw_motion.clone()
    changed[:, tiny_cfg.t_his:] = float("nan")
    target, condition = prepare_motion_inputs(raw_motion, tiny_cfg)
    changed_target, changed_condition = prepare_motion_inputs(changed, tiny_cfg)
    options, inpaint_target, encoded = sample_preprocessing(target, tiny_cfg, mode, traj_cond=condition)
    other_options, other_target, other_encoded = sample_preprocessing(
        changed_target, tiny_cfg, mode, traj_cond=changed_condition)

    assert inpaint_target is None and other_target is None
    assert "mask" not in options and "mask" not in other_options
    torch.testing.assert_close(encoded, other_encoded, rtol=0, atol=0)
    noise = torch.randn(options["sample_num"], tiny_cfg.n_pre, 6)
    diffusion = small_diffusion()
    sample = diffusion.sample_ddim(ObservedConditionNoise(), None, encoded, options, noise=noise)
    other_sample = diffusion.sample_ddim(ObservedConditionNoise(), None, other_encoded, other_options, noise=noise)
    torch.testing.assert_close(sample, other_sample, rtol=0, atol=0)
    assert sample.shape == noise.shape
    assert torch.isfinite(sample).all()


def test_future_sampler_rejects_observation_inpainting():
    with pytest.raises(ValueError, match="inpaint"):
        Diffusion(EnableComplete=True)
    with pytest.raises(ValueError, match="inpaint"):
        small_diffusion().sample_ddim(nn.Identity(), torch.zeros(1, 5, 6), None, {"sample_num": 1})


@pytest.mark.parametrize("invalid_steps", [0, 12, 2.5])
def test_ddim_rejects_invalid_step_counts(invalid_steps):
    with pytest.raises(ValueError):
        small_diffusion(invalid_steps)
