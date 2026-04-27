from torch import tensor
from utils import *
from utils.script import sample_preprocessing


def _attach_robot_joints_for_vis(pred_human, gt_full, cfg):
    """
    For HARPER human-only prediction, append GT robot joints so rendered poses
    share the same full skeleton (human + robot) as context/gt panels.
    """
    if getattr(cfg, 'dataset', None) != 'harper3d':
        return pred_human
    if not getattr(cfg, 'include_spot', False):
        return pred_human
    if not getattr(cfg, 'predict_human_only', False):
        return pred_human
    if gt_full.shape[1] <= cfg.output_total_joints:
        return pred_human

    # gt_full: [T, J_full, 3], pred_human: [N, T, J_human, 3]
    robot_gt = gt_full[:, cfg.output_total_joints:, :]
    robot_gt = np.expand_dims(robot_gt, axis=0)
    robot_gt = np.repeat(robot_gt, pred_human.shape[0], axis=0)
    return np.concatenate([pred_human, robot_gt], axis=2)


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
            gt[:, :1, :] = 0
            data[:, :, :1, :] = 0

            if mode == 'pred':
                if draw_order_indicator == -1:
                    poses['context'] = gt
                    poses['gt'] = gt
                else:
                    poses[f'TransFusion_{draw_order_indicator + 1}'] = gt
                    poses[f'TransFusion_{draw_order_indicator + 2}'] = gt
                gt = np.expand_dims(gt, axis=0)
                traj_np, traj_cond_np = split_motion_inputs(gt, cfg)

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
            traj_cond = tensor(traj_cond_np, device=cfg.device, dtype=cfg.dtype)

            mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode, traj_cond=traj_cond)
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict)

            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            traj_est = reconstruct_from_velocity(traj_est, gt, cfg)
            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)
            traj_est = _attach_robot_joints_for_vis(traj_est, gt[0], cfg)

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

        yield poses
