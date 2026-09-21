"""Exercise the real metrics/CSV path with known future predictions."""

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import pytest

from utils.evaluation import compute_stats


@pytest.mark.parametrize('sample_count', [1, 3, 50])
def test_compute_stats_scores_only_future_and_writes_csv(tiny_cfg, raw_motion, tmp_path, sample_count):
    tiny_cfg.result_dir = str(tmp_path)
    tiny_cfg.eval_samples = sample_count
    # Every coordinate's true future is 1,2,3,4,5; every sampled future is 0.
    raw_motion[:, tiny_cfg.t_his:, :2] = torch.arange(1, 6).view(1, 5, 1, 1)
    ground_truth = raw_motion[:, tiny_cfg.t_his:, :2].flatten(-2).numpy().copy()
    # Ground-truth futures are supplied exclusively to the metric. The model
    # input cannot use them, even though the full data container has a tail.
    data = raw_motion.numpy().copy()
    data[:, tiny_cfg.t_his:] = np.nan
    observed = torch.from_numpy(data[:, :tiny_cfg.t_his].reshape(2, tiny_cfg.t_his, -1))

    class ZeroFutureSampler:
        calls = 0

        def sample_ddim(self, model, target, condition, options):
            self.calls += 1
            assert target is None and "mask" not in options
            assert condition.shape == (2, tiny_cfg.t_his, 9)
            torch.testing.assert_close(tiny_cfg.idct_m_obs @ condition, observed)
            return torch.zeros(2, tiny_cfg.n_pre, 6)

    sampler = ZeroFutureSampler()
    run_dir = compute_stats(sampler, {
        "data_group": data, "gt_group": ground_truth,
        "traj_gt_arr": [future[None] for future in ground_truth], "num_samples": 2,
    }, model=None, logger=logging.getLogger("alignment-eval-test"), cfg=tiny_cfg)

    assert sampler.calls == sample_count
    from utils.rescore_evaluation import rescore
    rescore(run_dir)
    assert (Path(run_dir) / 'summary_rescored.csv').is_file()
    for filename in ("stats_latest.csv", "stats.csv"):
        with (Path(tiny_cfg.result_dir) / filename).open(newline="") as stream:
            results = {row["Metric"]: float(row["ReFusion"]) for row in csv.DictReader(stream)}
        assert len(results) == 13
        assert results["APD"] == 0
        for metric, actual in results.items():
            if "ADE" in metric:
                np.testing.assert_allclose(actual, 3 * np.sqrt(6), rtol=1e-6)
            elif "FDE" in metric:
                np.testing.assert_allclose(actual, 5 * np.sqrt(6), rtol=1e-6)
