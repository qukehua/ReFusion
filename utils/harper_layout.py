"""ReFusion's author-confirmed HARPER input subset (zero-based indices).

The authors exclude raw Spot points 21/22 as camera-related points. Official
links.py names them wrist/hand; this selection records the authors' protocol,
not a revision of the upstream anatomical labels.
"""

HARPER_SPOT_RAW_JOINTS = 23
HARPER_SPOT_RETAINED_INDICES = tuple(range(21))


def resolve_spot_joint_indices(indices=None):
    """Keep 0..20 in order; prevent excluded points from re-entering the model."""
    if indices is None:
        return list(HARPER_SPOT_RETAINED_INDICES)
    if (not isinstance(indices, (list, tuple))
            or any(type(i) is not int for i in indices)
            or tuple(indices) != HARPER_SPOT_RETAINED_INDICES):
        raise ValueError('harper_spot_joint_indices must be 0..20 in order; '
                         'raw Spot points 21 and 22 are excluded by the ReFusion protocol.')
    return list(indices)


def select_spot_joints(spot_sequence, indices=None):
    """Select before centering, augmentation, DCT, or metric preparation."""
    indices = resolve_spot_joint_indices(indices)
    if spot_sequence.ndim != 3 or spot_sequence.shape[1:] != (HARPER_SPOT_RAW_JOINTS, 3):
        raise ValueError('Expected raw Spot trajectories [F,23,3] for the author-confirmed '
                         f'index convention; got {spot_sequence.shape}.')
    return spot_sequence[:, indices, :]
