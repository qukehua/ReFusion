"""Protect the separation between observations, future targets and outputs."""

import pytest
import torch

from utils.motion_latent import (
    decode_future_motion,
    encode_future,
    encode_observation,
    prepare_motion_inputs,
)


def test_future_target_is_independent_of_observed_positions(tiny_cfg, raw_motion):
    target, observed = prepare_motion_inputs(raw_motion, tiny_cfg)
    changed = raw_motion.clone()
    changed[:, :tiny_cfg.t_his] = 999 * torch.randn_like(changed[:, :tiny_cfg.t_his])
    changed_target, changed_observed = prepare_motion_inputs(changed, tiny_cfg)

    latent = encode_future(target, tiny_cfg)
    torch.testing.assert_close(latent, encode_future(changed_target, tiny_cfg), rtol=0, atol=0)
    assert latent.shape == (2, tiny_cfg.t_pred, 6)
    assert observed.shape == (2, tiny_cfg.t_his, 9)
    assert not torch.equal(observed, changed_observed)
    # An inverse DCT must recover the future alone, including its first frame.
    torch.testing.assert_close(tiny_cfg.idct_m_pred @ latent, target[:, tiny_cfg.t_his:])


def test_condition_never_reads_future_even_for_scene_selection(tiny_cfg, raw_motion):
    # CoMaD used to infer whether a robot was present from the whole sequence.
    tiny_cfg.dataset = "comad"
    tiny_cfg.use_hr_robot_condition = True
    tiny_cfg.use_hh_human_condition = True
    tiny_cfg.comad_p1_joints = 1
    tiny_cfg.comad_p2_joints = 1
    tiny_cfg.comad_robot_joints = 1
    raw_motion[:, :tiny_cfg.t_his, 2] = 0
    changed = raw_motion.clone()
    changed[:, tiny_cfg.t_his:] = float("nan")

    _, condition = prepare_motion_inputs(raw_motion, tiny_cfg)
    _, changed_condition = prepare_motion_inputs(changed, tiny_cfg)
    torch.testing.assert_close(condition, changed_condition, rtol=0, atol=0)
    encoded = encode_observation(condition, tiny_cfg)
    assert encoded.shape == (2, tiny_cfg.t_his, 9)
    torch.testing.assert_close(encoded, encode_observation(changed_condition, tiny_cfg), rtol=0, atol=0)
    assert torch.isfinite(encoded).all()


@pytest.mark.parametrize("use_velocity", [False, True])
def test_decode_copies_history_and_recovers_future(tiny_cfg, raw_motion, use_velocity):
    tiny_cfg.use_velocity_input = use_velocity
    target, _ = prepare_motion_inputs(raw_motion, tiny_cfg)
    future_latent = encode_future(target, tiny_cfg)
    # Make reference futures invalid: reconstruction must use history only.
    reference = raw_motion.clone()
    reference[:, tiny_cfg.t_his:] = float("nan")
    reconstructed = decode_future_motion(future_latent, reference, tiny_cfg)
    positions = raw_motion[:, :, :2].flatten(-2)

    assert reconstructed.shape == (2, tiny_cfg.t_his + tiny_cfg.t_pred, 6)
    torch.testing.assert_close(reconstructed[:, :tiny_cfg.t_his], positions[:, :tiny_cfg.t_his], rtol=0, atol=0)
    torch.testing.assert_close(reconstructed[:, tiny_cfg.t_his:], positions[:, tiny_cfg.t_his:], atol=2e-6, rtol=2e-6)


def test_decode_supports_multiple_samples_from_one_observation(tiny_cfg, raw_motion):
    target, _ = prepare_motion_inputs(raw_motion[:1], tiny_cfg)
    latent = encode_future(target, tiny_cfg).repeat(3, 1, 1)
    decoded = decode_future_motion(latent, raw_motion[:1, :tiny_cfg.t_his], tiny_cfg)
    assert decoded.shape == (3, tiny_cfg.t_his + tiny_cfg.t_pred, 6)
    torch.testing.assert_close(decoded[0], decoded[1], rtol=0, atol=0)


def test_explicit_truncated_future_dct_has_prediction_horizon(tiny_cfg, raw_motion):
    tiny_cfg.n_pre = 3
    target, _ = prepare_motion_inputs(raw_motion, tiny_cfg)
    latent = encode_future(target, tiny_cfg)
    decoded = decode_future_motion(latent, raw_motion, tiny_cfg)
    assert latent.shape == (2, 3, 6)
    assert decoded.shape == (2, tiny_cfg.t_his + tiny_cfg.t_pred, 6)
    torch.testing.assert_close(decoded[:, tiny_cfg.t_his:], tiny_cfg.idct_m_pred[:, :3] @ latent)


def test_encode_future_rejects_observation_only_input(tiny_cfg):
    with pytest.raises(ValueError):
        encode_future(torch.randn(2, tiny_cfg.t_his, 6), tiny_cfg)
