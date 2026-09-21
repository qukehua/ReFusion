from torch import tensor
from utils import *
from utils.script import sample_preprocessing
from utils.motion_latent import prepare_motion_inputs, decode_future_motion


def _select_visual_samples(traj_est, cfg):
    """
    Select displayed samples from candidate trajectories.

    Strategies:
      - random: keep random candidates (legacy behavior)
      - median: choose trajectories closest to the point-wise median trajectory
    """
    n_show = int(getattr(cfg, 'vis_col', traj_est.shape[0]))
    if traj_est.shape[0] <= n_show:
        return traj_est

    # Harper3D: keep legacy visualization (random picks); CHICO can use median-of-K from yaml.
    strategy = str(getattr(cfg, 'vis_sample_strategy', 'random')).lower()
    if getattr(cfg, 'dataset', None) == 'harper3d':
        strategy = 'random'
    if strategy == 'random':
        idx = np.random.choice(traj_est.shape[0], size=n_show, replace=False)
        return traj_est[idx]
    if strategy == 'median':
        median_traj = np.median(traj_est, axis=0, keepdims=True)
        dist = np.linalg.norm((traj_est - median_traj).reshape(traj_est.shape[0], -1), axis=1)
        idx = np.argsort(dist)[:n_show]
        return traj_est[idx]

    raise ValueError(
        f"Unsupported vis_sample_strategy '{strategy}'. Supported: 'random', 'median'."
    )


def _attach_robot_joints_for_vis(pred_human, gt_full, cfg):
    """
    Append recorded partner/robot joints solely as a visualization reference.

    These context trajectories are not model predictions; future context is
    never passed to the denoiser. Keeping the full scene here also matches the
    renderer's joint indexing for paired-person datasets.
    """
    if getattr(cfg, 'vis_output_only', False):
        return pred_human
    if not getattr(cfg, 'predict_human_only', False):
        return pred_human
    if gt_full.shape[1] <= cfg.output_total_joints:
        return pred_human
    if getattr(cfg, 'vis_skip_attach_robot', False):
        return pred_human

    ds = getattr(cfg, 'dataset', None)
    if ds == 'harper3d':
        if not getattr(cfg, 'include_spot', False):
            return pred_human
    elif ds == 'chico':
        if not getattr(cfg, 'include_robot', False):
            return pred_human
    elif ds == 'comad':
        if not (getattr(cfg, 'include_person2', False) or getattr(cfg, 'include_robot', False)):
            return pred_human
    elif ds in ('3dpw', 'cmu_mocap'):
        pass
    else:
        return pred_human

    # gt_full: [T, J_full, 3], pred_human: [N, T, J_human, 3]
    context_gt = gt_full[:, cfg.output_total_joints:, :]
    context_gt = np.expand_dims(context_gt, axis=0)
    context_gt = np.repeat(context_gt, pred_human.shape[0], axis=0)
    return np.concatenate([pred_human, context_gt], axis=2)


def _pose_for_visualization(pose, cfg):
    if getattr(cfg, 'vis_output_only', False):
        if getattr(cfg, 'dataset', None) == 'comad':
            from data_loader.comad_kinematics import comad_visual_joint_indices
            return pose[..., comad_visual_joint_indices(cfg), :].copy()
        return pose[..., :cfg.output_total_joints, :].copy()
    return pose


def pose_generator(data_set, model_select, diffusion, cfg, mode=None,
                   action=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            if mode == 'pred':
                data = data_set.sample_iter_action(action, cfg.dataset)
            else:
                raise NotImplementedError(f"unknown pose generator mode: {mode}")

            # gt
            gt = data[0].copy()
            if getattr(cfg, 'drop_root_joint', True):
                gt[:, :1, :] = 0
            vis_gt = _pose_for_visualization(gt, cfg)

            if mode == 'pred':
                if draw_order_indicator == -1:
                    poses['context'] = vis_gt
                    poses['gt'] = vis_gt
                else:
                    poses[f'TransFusion_{draw_order_indicator + 1}'] = vis_gt
                    poses[f'TransFusion_{draw_order_indicator + 2}'] = vis_gt
                gt = np.expand_dims(gt, axis=0)
                traj_np, traj_cond_np = prepare_motion_inputs(gt[:, :cfg.t_his], cfg)

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
            traj_cond = tensor(traj_cond_np, device=cfg.device, dtype=cfg.dtype)

            n_show = int(getattr(cfg, 'vis_col', 10))
            n_candidate = getattr(cfg, 'vis_candidate_k', n_show)
            n_candidate = max(n_show, int(n_candidate))
            mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(
                traj, cfg, mode=mode, traj_cond=traj_cond, sample_num=n_candidate
            )
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict)

            traj_est = decode_future_motion(sampled_motion, gt[:, :cfg.t_his], cfg)
            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)
            traj_est = _pose_for_visualization(traj_est, cfg)
            traj_est = _attach_robot_joints_for_vis(traj_est, gt[0], cfg)
            traj_est = _select_visual_samples(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'TransFusion_{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'TransFusion_{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1

        if cfg.dataset == 'comad' and getattr(cfg, 'comad_motive_to_interact_axes', False):
            from data_loader.comad_kinematics import comad_fix_orientation_motive_to_interact
            # Apply a display-only transform once, after inference, to every pose.
            poses = {key: comad_fix_orientation_motive_to_interact(value) for key, value in poses.items()}
        yield poses
