"""Forecast visualization must preserve observed markers and scene indexing."""

import numpy as np
import pytest
import torch

from utils.pose_gen import _attach_robot_joints_for_vis, pose_generator
from utils.script import create_model_and_diffusion


class OneVisualizationSequence:
    def __init__(self, batch):
        self.batch = batch

    def sample_iter_action(self, action, dataset):
        return self.batch


@pytest.mark.parametrize("drop_root", [False, True])
def test_pose_generator_preserves_markers_and_does_not_mutate_dataset(tiny_cfg, raw_motion, drop_root):
    tiny_cfg.drop_root_joint = drop_root
    tiny_cfg.vis_output_only = True
    if drop_root:
        tiny_cfg.joint_num = 1
        tiny_cfg.cond_joint_num = 2
    raw = raw_motion[:1].numpy()
    original = raw.copy()
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    generator = pose_generator(OneVisualizationSequence(raw), model, diffusion, tiny_cfg, mode="pred")
    poses = next(generator)
    generator.close()

    np.testing.assert_array_equal(raw, original)
    expected = original[0, :, :2].copy()
    if drop_root:
        expected[:, 0] = 0
    np.testing.assert_array_equal(poses["gt"], expected)
    assert len(poses) == 2 + tiny_cfg.vis_col
    for key, pose in poses.items():
        if key.startswith("TransFusion_"):
            assert pose.shape == (tiny_cfg.t_his + tiny_cfg.t_pred, 2, 3)
            np.testing.assert_array_equal(pose[:tiny_cfg.t_his], expected[:tiny_cfg.t_his])


@pytest.mark.parametrize("dataset,person_joints", [("3dpw", 24), ("cmu_mocap", 39)])
def test_paired_person_visualization_has_full_scene_joint_count(tiny_cfg, dataset, person_joints):
    tiny_cfg.dataset = dataset
    tiny_cfg.output_total_joints = person_joints
    predicted_person = np.ones((3, 8, person_joints, 3))
    reference = np.random.randn(8, 2 * person_joints, 3)
    scene = _attach_robot_joints_for_vis(predicted_person, reference, tiny_cfg)

    assert scene.shape == (3, 8, 2 * person_joints, 3)
    np.testing.assert_array_equal(scene[:, :, :person_joints], predicted_person)
    np.testing.assert_array_equal(scene[:, :, person_joints:],
                                  np.repeat(reference[None, :, person_joints:], 3, axis=0))


def test_comad_first_marker_is_preserved(tiny_cfg):
    pytest.importorskip('data_loader.comad_kinematics', reason='Private CoMaD visualization mapping')
    tiny_cfg.dataset = "comad"
    tiny_cfg.comad_p1_joints = tiny_cfg.joint_num = tiny_cfg.output_total_joints = 9
    tiny_cfg.comad_p2_joints = 0
    tiny_cfg.comad_robot_joints = 2
    tiny_cfg.cond_joint_num = 11
    tiny_cfg.use_hr_robot_condition = True
    tiny_cfg.use_hh_human_condition = False
    tiny_cfg.vis_output_only = True
    raw = np.random.randn(1, 8, 11, 3).astype(np.float32)
    original = raw.copy()
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    generator = pose_generator(OneVisualizationSequence(raw), model, diffusion, tiny_cfg, mode="pred")
    poses = next(generator)
    generator.close()
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(poses["gt"][:, 0], original[0, :, 0])
    np.testing.assert_array_equal(poses["TransFusion_0"][:tiny_cfg.t_his, 0], original[0, :tiny_cfg.t_his, 0])


@pytest.mark.parametrize("dataset", ["3dpw", "cmu_mocap"])
def test_partner_future_affects_visual_reference_but_not_sampler_condition(tiny_cfg, dataset):
    tiny_cfg.dataset = dataset
    tiny_cfg.use_partner_condition = True
    tiny_cfg.cond_joint_num = 4
    raw = np.random.randn(1, 8, 4, 3).astype(np.float32)
    conditions = []

    class RecordingSampler:
        def sample_ddim(self, model, target, condition, options):
            assert target is None
            conditions.append(condition.clone())
            return torch.zeros(options["sample_num"], tiny_cfg.n_pre, 6)

    def generate(sequence):
        generator = pose_generator(OneVisualizationSequence(sequence), None, RecordingSampler(), tiny_cfg, mode="pred")
        poses = next(generator)
        generator.close()
        return poses["TransFusion_0"]

    before = generate(raw)
    changed = raw.copy()
    changed[:, tiny_cfg.t_his:, 2:] += 100
    after = generate(changed)
    torch.testing.assert_close(conditions[0], conditions[1], rtol=0, atol=0)
    assert before.shape == after.shape == (8, 4, 3)
    np.testing.assert_array_equal(before[:, :2], after[:, :2])
    np.testing.assert_allclose(after[tiny_cfg.t_his:, 2:] - before[tiny_cfg.t_his:, 2:], 100)
