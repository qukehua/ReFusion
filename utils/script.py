import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import get_position_inputs
from utils.motion_latent import encode_observation
from utils.visualization import render_animation
from models.default import MotionTransformer
from models.condition_two_stage import MotionTransformerTwoStage
from models.diffusion import Diffusion
from importlib import import_module
from scipy.spatial.distance import pdist, squareform


class LegacyConditionLengthAdapter(MotionTransformer):
    """Keep the optional default baseline usable with T != M.

    This baseline interpolates observed DCT coefficients to its existing token
    length; it is not ReFusion's learned observation-to-future projection.
    No ground-truth future or repeated-pose time padding is used.
    """

    def forward(self, x, timesteps, mod=None):
        if mod is not None and mod.shape[1] != x.shape[1]:
            mod = torch.nn.functional.interpolate(
                mod.transpose(1, 2), size=x.shape[1], mode='linear',
                align_corners=False).transpose(1, 2)
        return super().forward(x, timesteps, mod=mod)


def create_model_and_diffusion(cfg):
    """
    create TransLinear model and Diffusion
    """
    model_variant = getattr(cfg, 'model_variant', 'default')
    if model_variant == 'two_stage':
        model_cls = MotionTransformerTwoStage
    elif model_variant == 'default':
        model_cls = LegacyConditionLengthAdapter
    else:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. "
            "Supported values are: 'default', 'two_stage'."
        )
    model = model_cls(
        input_feats=3 * cfg.joint_num,  # 3 means x, y, z
        cond_feats=3 * cfg.cond_joint_num,
        human_cond_joint_num=cfg.joint_num,
        num_frames=cfg.t_pred if model_variant == 'two_stage' else cfg.n_pre,
        cond_frames=cfg.t_his,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        dropout=cfg.dropout,
        stage1_num_layers=cfg.stage1_num_layers,
        dit_attn_mode=cfg.dit_attn_mode,
    ).to(cfg.device)
    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(cfg.n_pre, 3 * cfg.joint_num),  # 3 means x, y, z
        device=cfg.device, padding=cfg.padding,
        EnableComplete=cfg.Complete,
        ddim_timesteps=cfg.ddim_timesteps,
        scheduler=cfg.scheduler,
        mod_test=cfg.mod_test,
        dct=cfg.dct_m_pred,
        idct=cfg.idct_m_pred,
        n_pre=cfg.n_pre
    )
    return model, diffusion


def dataset_split(cfg):
    """
    output: dataset_dict, dataset_multi_test
    dataset_dict: 'train' always; 'test' always (final eval / multimodal). For 3dpw, 'val'
    is the official validation split used during training instead of test (protocol).
    dataset_multi_test is used to create multi-modal data for metrics.
    """
    names = {'harper3d': 'DatasetHarper3D', 'chico': 'DatasetCHICO',
             'comad': 'DatasetCoMad', '3dpw': 'Dataset3DPW', 'cmu_mocap': 'DatasetCMUMocap'}
    if cfg.dataset not in names:
        raise ValueError(f'Unsupported dataset {cfg.dataset!r}.')
    module = 'data_loader.dataset_' + cfg.dataset
    try:
        dataset_cls = getattr(import_module(module), names[cfg.dataset])
        dataset_cls_multi = getattr(import_module(module + '_multimodal'), names[cfg.dataset] + '_multi')
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == 'data_loader' or exc.name.startswith('data_loader.')):
            raise RuntimeError(
                'Dataset loaders are private pending paper acceptance. Place the authors\' '
                'loader sources in data_loader/ to train or evaluate. Core model tests do not require them.'
            ) from exc
        raise

    if cfg.dataset == 'harper3d':
        dataset = dataset_cls('train', cfg.t_his, cfg.t_pred, actions='all',
                              data_path=cfg.data_path, include_spot=cfg.include_spot,
                              spot_joint_indices=cfg.harper_spot_joint_indices,
                              fps=cfg.fps,
                              use_data_aug=cfg.use_data_aug,
                              aug_rotate_prob=cfg.aug_rotate_prob,
                              aug_reverse_prob=cfg.aug_reverse_prob)
        dataset_test = dataset_cls('test', cfg.t_his, cfg.t_pred, actions='all',
                                   data_path=cfg.data_path, include_spot=cfg.include_spot,
                                   spot_joint_indices=cfg.harper_spot_joint_indices,
                                   fps=cfg.fps,
                                   use_data_aug=False)
        dataset_multi_test = dataset_cls_multi('test', cfg.t_his, cfg.t_pred,
                                               data_path=cfg.data_path,
                                               include_spot=cfg.include_spot,
                                               spot_joint_indices=cfg.harper_spot_joint_indices,
                                               fps=cfg.fps,
                                               multimodal_path=cfg.multimodal_path,
                                               data_candi_path=cfg.data_candi_path)
    elif cfg.dataset == 'chico':
        dataset = dataset_cls(
            'train',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            include_robot=cfg.include_robot,
            exclude_crash=cfg.chico_exclude_crash,
            use_data_aug=cfg.use_data_aug,
            aug_rotate_prob=cfg.aug_rotate_prob,
            aug_reverse_prob=cfg.aug_reverse_prob,
        )
        dataset_val = dataset_cls(
            'val', cfg.t_his, cfg.t_pred, actions='all', data_path=cfg.data_path,
            include_robot=cfg.include_robot, exclude_crash=cfg.chico_exclude_crash,
        )
        dataset_test = dataset_cls(
            'test',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            include_robot=cfg.include_robot,
            exclude_crash=cfg.chico_exclude_crash,
        )
        dataset_multi_test = dataset_cls_multi(
            'test',
            cfg.t_his,
            cfg.t_pred,
            data_path=cfg.data_path,
            include_robot=cfg.include_robot,
            multimodal_path=cfg.multimodal_path,
            data_candi_path=cfg.data_candi_path,
            exclude_crash=cfg.chico_exclude_crash,
        )
        return {'train': dataset, 'val': dataset_val, 'test': dataset_test}, dataset_multi_test
    elif cfg.dataset == 'comad':
        comad_test_if = getattr(cfg, 'comad_test_interactions', None)
        dataset = dataset_cls('train', cfg.t_his, cfg.t_pred, actions='all',
                              data_path=cfg.data_path,
                              include_person2=cfg.include_person2,
                              include_robot=cfg.include_robot,
                              use_data_aug=cfg.use_data_aug,
                              aug_rotate_prob=cfg.aug_rotate_prob,
                              aug_reverse_prob=cfg.aug_reverse_prob,
                              eval_interaction_filter=getattr(cfg, 'comad_train_interactions', None),
                              p1_joints=cfg.comad_p1_joints,
                              p2_joints=cfg.comad_p2_joints,
                              robot_joints=cfg.comad_robot_joints,
                              p1_joint_indices=cfg.comad_p1_joint_indices,
                              p1_fallback_joint_indices=cfg.comad_p1_fallback_joint_indices,
                              robot_joint_indices=cfg.comad_robot_joint_indices,
                              robot_fallback_joint_indices=cfg.comad_robot_fallback_joint_indices)
        dataset_test = dataset_cls('test', cfg.t_his, cfg.t_pred, actions='all',
                                   data_path=cfg.data_path,
                                   include_person2=cfg.include_person2,
                                   include_robot=cfg.include_robot,
                                   use_data_aug=False,
                                   eval_interaction_filter=comad_test_if,
                                   p1_joints=cfg.comad_p1_joints,
                                   p2_joints=cfg.comad_p2_joints,
                                   robot_joints=cfg.comad_robot_joints,
                                   p1_joint_indices=cfg.comad_p1_joint_indices,
                                   p1_fallback_joint_indices=cfg.comad_p1_fallback_joint_indices,
                                   robot_joint_indices=cfg.comad_robot_joint_indices,
                                   robot_fallback_joint_indices=cfg.comad_robot_fallback_joint_indices)
        dataset_multi_test = dataset_cls_multi('test', cfg.t_his, cfg.t_pred,
                                               data_path=cfg.data_path,
                                               include_person2=cfg.include_person2,
                                               include_robot=cfg.include_robot,
                                               multimodal_path=cfg.multimodal_path,
                                               data_candi_path=cfg.data_candi_path,
                                               eval_interaction_filter=comad_test_if,
                                               p1_joints=cfg.comad_p1_joints,
                                               p2_joints=cfg.comad_p2_joints,
                                               robot_joints=cfg.comad_robot_joints,
                                               p1_joint_indices=cfg.comad_p1_joint_indices,
                                               p1_fallback_joint_indices=cfg.comad_p1_fallback_joint_indices,
                                               robot_joint_indices=cfg.comad_robot_joint_indices,
                                               robot_fallback_joint_indices=cfg.comad_robot_fallback_joint_indices)
    elif cfg.dataset == '3dpw':
        train_scene = getattr(cfg, 'scene_filter_train', None)
        test_scene = getattr(cfg, 'scene_filter_test', None)
        val_scene = getattr(cfg, 'scene_filter_val', None)
        if val_scene is None:
            val_scene = test_scene
        dataset = dataset_cls(
            'train',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            scene_filter=train_scene,
            require_two_person=getattr(cfg, 'require_two_person', True),
            use_data_aug=cfg.use_data_aug,
            aug_rotate_prob=cfg.aug_rotate_prob,
            aug_reverse_prob=cfg.aug_reverse_prob,
        )
        dataset_val = dataset_cls(
            'val',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            scene_filter=val_scene,
            require_two_person=getattr(cfg, 'require_two_person', True),
            use_data_aug=False,
        )
        dataset_test = dataset_cls(
            'test',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            scene_filter=test_scene,
            require_two_person=getattr(cfg, 'require_two_person', True),
            use_data_aug=False,
        )
        dataset_multi_test = dataset_cls_multi(
            'test',
            cfg.t_his,
            cfg.t_pred,
            data_path=cfg.data_path,
            scene_filter=test_scene,
            require_two_person=getattr(cfg, 'require_two_person', True),
            multimodal_path=cfg.multimodal_path,
            data_candi_path=cfg.data_candi_path,
        )
        return {'train': dataset, 'val': dataset_val, 'test': dataset_test}, dataset_multi_test
    elif cfg.dataset == 'cmu_mocap':
        cmu_scene = getattr(cfg, 'cmu_scene_filter', None)
        cmu_file = getattr(cfg, 'cmu_file_filter', None)
        dataset = dataset_cls(
            'train',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            scene_filter=cmu_scene,
            file_filter=cmu_file,
            person_joint_num=cfg.cmu_person_joint_num,
            use_data_aug=cfg.use_data_aug,
            aug_rotate_prob=cfg.aug_rotate_prob,
            aug_reverse_prob=cfg.aug_reverse_prob,
        )
        dataset_test = dataset_cls(
            'test',
            cfg.t_his,
            cfg.t_pred,
            actions='all',
            data_path=cfg.data_path,
            scene_filter=cmu_scene,
            file_filter=cmu_file,
            person_joint_num=cfg.cmu_person_joint_num,
            use_data_aug=False,
        )
        dataset_multi_test = dataset_cls_multi(
            'test',
            cfg.t_his,
            cfg.t_pred,
            data_path=cfg.data_path,
            scene_filter=cmu_scene,
            file_filter=cmu_file,
            person_joint_num=cfg.cmu_person_joint_num,
            multimodal_path=cfg.multimodal_path,
            data_candi_path=cfg.data_candi_path,
        )
    return {'train': dataset, 'test': dataset_test}, dataset_multi_test


def get_multimodal_gt_full(logger, dataset_multi_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    num_samples = 0
    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    for data, _ in data_gen_multi_test:
        num_samples += 1
        data_group.append(data)
    if not data_group:
        raise ValueError('No complete evaluation windows for the configured observation/prediction horizons.')
    data_group = np.concatenate(data_group, axis=0)
    all_data, _ = get_position_inputs(data_group, cfg)
    gt_group = all_data[:, cfg.t_his:, :]

    if args.multimodal_threshold <= 0:
        raise ValueError('multimodal_threshold must be positive (the query itself is a neighbor).')
    # Protocol: Euclidean distance of flattened target joints at the LAST
    # observed frame. No skeleton scaling or multiple-history matching.
    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    neighbor_indices = []
    for i in tqdm(
        range(pd.shape[0]),
        desc='Eval prep: multimodal neighbors',
        unit='seq',
    ):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        neighbor_indices.append(ind[0])
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))
    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'Test set size: {num_samples}')
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    logger.info('done...')
    logger.info('=' * 80)
    return {'traj_gt_arr': traj_gt_arr,
            'data_group': data_group,
            'gt_group': gt_group,
            'neighbor_indices': neighbor_indices,
            'window_ids': list(dataset_multi_test.iter_window_ids(step=cfg.t_his))
                if hasattr(dataset_multi_test, 'iter_window_ids') else list(range(num_samples)),
            'num_samples': num_samples}


def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            del log_dict[key]
    del log_dict['zero_index']
    del log_dict['idx_pad']
    logger.info(log_dict)
    logger.info('=' * 80)


def sample_preprocessing(traj, cfg, mode, traj_cond=None, sample_num=None):
    """Prepare observed DCT_T [B,T,D_cond] for future-only sampling.

    traj may contain history alone or an evaluation reference. Future frames
    are never accessed. The second return value is always None, replacing the
    old inpainting-target slot; mode_dict contains no observation mask.
    """
    if traj_cond is None:
        traj_cond = traj
    if traj.ndim != 3 or traj_cond.ndim != 3 or traj.shape[0] != traj_cond.shape[0]:
        raise ValueError("Target and condition must be [B,L,D] tensors with matching batches.")
    traj_dct_mod = encode_observation(traj_cond, cfg)
    if mode == 'pred':
        n = cfg.vis_col if sample_num is None else int(sample_num)
        if traj.shape[0] != 1 or n < 1:
            raise ValueError("Visualization expects one observed trajectory and a positive sample count.")
        traj_dct_mod = traj_dct_mod.repeat(n, 1, 1)
    elif mode == 'metrics':
        n = traj.shape[0]
    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
    if np.random.random() > cfg.mod_test:
        traj_dct_mod = None
    return {'sample_num': n, 'mode': mode}, None, traj_dct_mod
