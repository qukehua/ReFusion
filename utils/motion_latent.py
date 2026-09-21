"""Future-only DCT targets and observation-only conditioning for ReFusion.

T is the observation horizon, P the prediction horizon, and M <= P the
retained future coefficients (M = P for the paper configuration).
"""

import torch

from utils.util import get_position_inputs, split_motion_inputs


def prepare_motion_inputs(raw_motion, cfg):
    """Return target [B,T+P,D] and observed condition [B,T,D_cond].

    Conditions are constructed from the observed raw joints independently.
    This also prevents dataset-specific condition selection (e.g. CoMaD scene
    availability) from inspecting ground-truth future frames. Backward
    differences in the optional velocity target preserve the T -> T+1 edge.
    """
    if raw_motion.ndim != 4 or raw_motion.shape[1] < cfg.t_his:
        raise ValueError("Expected raw joints [B,L,J,3] with L >= t_his.")
    target, _ = split_motion_inputs(raw_motion, cfg)
    _, observed_condition = split_motion_inputs(raw_motion[:, :cfg.t_his], cfg)
    return target, observed_condition


def encode_future(traj, cfg):
    """DCT_P of future [B,P,D] -> [B,M,D]; never encode observed frames."""
    if traj.ndim != 3 or traj.shape[1] != cfg.t_his + cfg.t_pred:
        raise ValueError("Training target must have shape [B,t_his+t_pred,D].")
    future = traj[:, cfg.t_his:cfg.t_his + cfg.t_pred]
    return torch.matmul(cfg.dct_m_pred[:cfg.n_pre].to(future), future)


def encode_observation(traj_cond, cfg):
    """DCT_T of observed condition [B,T,D_cond] -> [B,T,D_cond]."""
    if traj_cond.ndim != 3 or traj_cond.shape[1] < cfg.t_his:
        raise ValueError("Condition must have shape [B,L,D_cond], L >= t_his.")
    observed = traj_cond[:, :cfg.t_his]
    return torch.matmul(cfg.dct_m_obs.to(observed), observed)


def decode_future_motion(future_dct, reference_traj, cfg):
    """Decode [B,M,D] -> positions [B,T+P,D] for metrics and visualization.

    The only reference frames accessed are the T observations. If the optional
    velocity baseline is enabled, integrate predicted future displacements
    from the last observed pose. History is copied exactly after decoding.
    """
    if future_dct.ndim != 3 or future_dct.shape[1] != cfg.n_pre:
        raise ValueError("Expected future coefficients [B,n_pre,D].")
    observed_pos, _ = get_position_inputs(reference_traj[:, :cfg.t_his], cfg)
    observed_pos = torch.as_tensor(observed_pos, device=future_dct.device,
                                   dtype=future_dct.dtype)
    if observed_pos.shape[1] != cfg.t_his:
        raise ValueError("Reference must contain all t_his observed frames.")
    if observed_pos.shape[0] == 1:
        observed_pos = observed_pos.expand(future_dct.shape[0], -1, -1)
    if (observed_pos.shape[0] != future_dct.shape[0]
            or observed_pos.shape[2] != future_dct.shape[2]):
        raise ValueError("Reference batch/features do not match future coefficients.")
    future = torch.matmul(cfg.idct_m_pred[:, :cfg.n_pre].to(future_dct), future_dct)
    if getattr(cfg, 'use_velocity_input', False):
        future = observed_pos[:, -1:] + future.cumsum(dim=1)
    return torch.cat((observed_pos, future), dim=1)
