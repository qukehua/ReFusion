import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import padding_traj, split_motion_inputs, get_position_inputs
from utils.visualization import render_animation
from models.default import MotionTransformer
from models.condition_two_stage import MotionTransformerTwoStage
from models.diffusion import Diffusion
from data_loader.dataset_harper3d import DatasetHarper3D
from data_loader.dataset_harper3d_multimodal import DatasetHarper3D_multi
from data_loader.dataset_chico import DatasetCHICO
from data_loader.dataset_chico_multimodal import DatasetCHICO_multi
from data_loader.dataset_comad import DatasetCoMad
from data_loader.dataset_comad_multimodal import DatasetCoMad_multi
from data_loader.dataset_3dpw import Dataset3DPW
from data_loader.dataset_3dpw_multimodal import Dataset3DPW_multi
from data_loader.dataset_cmu_mocap import DatasetCMUMocap
from data_loader.dataset_cmu_mocap_multimodal import DatasetCMUMocap_multi
from scipy.spatial.distance import pdist, squareform


def create_model_and_diffusion(cfg):
    """
    create TransLinear model and Diffusion
    """
    model_variant = getattr(cfg, 'model_variant', 'default')
    if model_variant == 'two_stage' and cfg.stage1_num_layers != 0:
        model_cls = MotionTransformerTwoStage
    elif model_variant in ('default', 'two_stage'):
        model_cls = MotionTransformer
    else:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. "
            "Supported values are: 'default', 'two_stage'."
        )
    model = model_cls(
        input_feats=3 * cfg.joint_num,  # 3 means x, y, z
        cond_feats=3 * cfg.cond_joint_num,
        human_cond_joint_num=cfg.joint_num,
        num_frames=cfg.n_pre,
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
        dct=cfg.dct_m_all,
        idct=cfg.idct_m_all,
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
    if cfg.dataset == 'harper3d':
        dataset_cls = DatasetHarper3D
        dataset_cls_multi = DatasetHarper3D_multi
    elif cfg.dataset == 'chico':
        dataset_cls = DatasetCHICO
        dataset_cls_multi = DatasetCHICO_multi
    elif cfg.dataset == 'comad':
        dataset_cls = DatasetCoMad
        dataset_cls_multi = DatasetCoMad_multi
    elif cfg.dataset == '3dpw':
        dataset_cls = Dataset3DPW
        dataset_cls_multi = Dataset3DPW_multi
    elif cfg.dataset == 'cmu_mocap':
        dataset_cls = DatasetCMUMocap
        dataset_cls_multi = DatasetCMUMocap_multi
    else:
        raise ValueError(f"Unsupported dataset '{cfg.dataset}'. Supported: 'harper3d', 'chico', 'comad', '3dpw', 'cmu_mocap'.")

    if cfg.dataset == 'harper3d':
        dataset = dataset_cls('train', cfg.t_his, cfg.t_pred, actions='all',
                              data_path=cfg.data_path, include_spot=cfg.include_spot,
                              fps=cfg.fps,
                              use_data_aug=cfg.use_data_aug,
                              aug_rotate_prob=cfg.aug_rotate_prob,
                              aug_reverse_prob=cfg.aug_reverse_prob)
        dataset_test = dataset_cls('test', cfg.t_his, cfg.t_pred, actions='all',
                                   data_path=cfg.data_path, include_spot=cfg.include_spot,
                                   fps=cfg.fps,
                                   use_data_aug=False)
        dataset_multi_test = dataset_cls_multi('test', cfg.t_his, cfg.t_pred,
                                               data_path=cfg.data_path,
                                               include_spot=cfg.include_spot,
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
            use_data_aug=False,
        )
        dataset_multi_test = dataset_cls_multi(
            'test',
            cfg.t_his,
            cfg.t_pred,
            data_path=cfg.data_path,
            scene_filter=cmu_scene,
            file_filter=cmu_file,
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
    data_group = np.concatenate(data_group, axis=0)
    all_data, _ = get_position_inputs(data_group, cfg)
    gt_group = all_data[:, cfg.t_his:, :]

    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in tqdm(
        range(pd.shape[0]),
        desc='Eval prep: multimodal neighbors',
        unit='seq',
    ):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
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
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """

    if traj_cond is None:
        traj_cond = traj

    if mode == 'pred':
        n = cfg.vis_col if sample_num is None else int(sample_num)
        traj = traj.repeat(n, 1, 1)
        traj_cond = traj_cond.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        traj_cond_pad = padding_traj(traj_cond, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
        traj_dct_mod = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_cond_pad)
        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'pred'}, traj_dct, traj_dct_mod

    elif mode == 'metrics':
        n = traj.shape[0]

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        traj_cond_pad = padding_traj(traj_cond, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
        traj_dct_mod = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_cond_pad)
        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'metrics'}, traj_dct, traj_dct_mod
    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
