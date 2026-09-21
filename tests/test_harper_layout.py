"""Excluded HARPER points must not affect the model's observed conditions."""

import numpy as np
import pytest
import torch

from utils.harper_layout import resolve_spot_joint_indices, select_spot_joints
from utils.motion_latent import encode_future, encode_observation, prepare_motion_inputs


def test_excluded_spot_points_cannot_affect_condition_or_target(tiny_cfg):
    tiny_cfg.dataset = 'harper3d'
    tiny_cfg.use_spot_condition = True
    tiny_cfg.output_total_joints = 21
    tiny_cfg.drop_root_joint = True
    human = np.random.randn(8, 21, 3).astype(np.float32)
    spot = np.random.randn(8, 23, 3).astype(np.float32)

    def inputs(points):
        raw = np.concatenate([human, select_spot_joints(points)], axis=1)
        raw[:, 1:] -= raw[:, :1]
        raw[:, :1] = 0
        return prepare_motion_inputs(torch.from_numpy(raw[None]), tiny_cfg)

    target, condition = inputs(spot)
    changed = spot.copy()
    changed[:, 21:] = np.nan  # Even invalid excluded coordinates never reach DCT.
    changed_target, changed_condition = inputs(changed)
    assert target.shape == (1, 8, 60)
    assert condition.shape == (1, 3, 123)
    torch.testing.assert_close(encode_future(target, tiny_cfg), encode_future(changed_target, tiny_cfg))
    torch.testing.assert_close(encode_observation(condition, tiny_cfg), encode_observation(changed_condition, tiny_cfg))
    retained_changed = spot.copy()
    retained_changed[:, 20] += 1
    _, retained_condition = inputs(retained_changed)
    assert not torch.equal(condition, retained_condition)


@pytest.mark.parametrize('raw_count', [21, 22, 24])
def test_raw_layout_must_match_author_confirmed_23_point_convention(raw_count):
    with pytest.raises(ValueError, match='Expected raw Spot'):
        select_spot_joints(np.zeros((5, raw_count, 3)))


@pytest.mark.parametrize('indices', [list(range(20)) + [21], list(range(20)) + [22], list(reversed(range(21)))])
def test_excluded_points_and_reordered_subset_are_rejected(indices):
    with pytest.raises(ValueError, match='raw Spot points 21 and 22 are excluded'):
        resolve_spot_joint_indices(indices)
