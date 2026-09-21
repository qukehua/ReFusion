"""Dataset joint partitions and reproducible configuration records."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from config import Config, update_config
from utils.experiment import save_experiment_manifest
from utils.motion_latent import encode_future, encode_observation, prepare_motion_inputs
from utils.script import create_model_and_diffusion


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset,person_joints", [("3dpw", 24), ("cmu_mocap", 39)])
@pytest.mark.parametrize("use_partner", [False, True])
def test_two_person_datasets_predict_person_one_and_condition_on_observations(
    tmp_path, monkeypatch, dataset, person_joints, use_partner,
):
    config_dict = yaml.safe_load((REPO_ROOT / "cfg" / (dataset + ".yml")).read_text(encoding="utf-8"))
    config_dict.update(t_his=3, t_pred=5, n_pre=5, latent_dims=16, num_heads=2,
                       num_layers=2, stage1_num_layers=1, noise_steps=11,
                       ddim_timesteps=4, dropout=0.0, predict_human_only=True,
                       use_partner_condition=use_partner)
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / (dataset + ".yml")).write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = update_config(Config(dataset), {"cfg": dataset, "mode": "train", "device": "cpu"})
    assert cfg.joint_num == person_joints - 1
    expected_condition_joints = 2 * person_joints - 1 if use_partner else person_joints - 1
    assert cfg.cond_joint_num == expected_condition_joints
    assert cfg.dct_m_obs.shape == (3, 3) and cfg.dct_m_pred.shape == (5, 5)
    assert not hasattr(cfg, "dct_m_all")

    raw = torch.randn(1, 8, person_joints * 2, 3)
    target, observed = prepare_motion_inputs(raw, cfg)
    assert target.shape == (1, 8, 3 * (person_joints - 1))
    assert observed.shape == (1, 3, 3 * expected_condition_joints)
    torch.testing.assert_close(target, raw[:, :, 1:person_joints].flatten(-2))
    changed = raw.clone()
    changed[:, cfg.t_his:, person_joints:] = float("nan")
    changed_target, changed_observed = prepare_motion_inputs(changed, cfg)
    torch.testing.assert_close(changed_target, target, rtol=0, atol=0)
    torch.testing.assert_close(changed_observed, observed, rtol=0, atol=0)

    model, _ = create_model_and_diffusion(cfg)
    assert model.robot_cond_joint_num == (person_joints if use_partner else 0)
    output = model(encode_future(target, cfg), torch.tensor([3]), mod=encode_observation(observed, cfg))
    assert output.shape == (1, 5, 3 * (person_joints - 1))
    assert torch.isfinite(output).all()
    output.square().sum().backward()


def test_experiment_manifest_records_dimensions_source_and_checkpoint(tiny_cfg, tmp_path):
    tiny_cfg.id = "chico"
    tiny_cfg.mode = "train"
    tiny_cfg.cfg_dir = str(tmp_path)
    tiny_cfg.implementation_version = "refusion_future_dct_v1"
    checkpoint = tmp_path / "input_weights.pt"
    checkpoint.write_bytes(b"checkpoint fingerprint test")
    tiny_cfg.resume = str(checkpoint)
    model, _ = create_model_and_diffusion(tiny_cfg)
    first_path = Path(save_experiment_manifest(tiny_cfg, model))
    second_path = Path(save_experiment_manifest(tiny_cfg, model))
    assert first_path != second_path and first_path.is_file() and second_path.is_file()
    record = json.loads(first_path.read_text(encoding="utf-8"))

    assert record["tensor_shapes"]["observation_dct"] == ["B", 3, 9]
    assert record["tensor_shapes"]["future_dct"] == ["B", 5, 6]
    assert record["tensor_shapes"]["predicted_future"] == ["B", 5, 6]
    assert record["config"]["weight_decay"] == tiny_cfg.weight_decay
    assert all(not key.startswith(("dct_m", "idct_m")) for key in record["config"])
    assert record["source_sha256"]["utils/motion_latent.py"] == hashlib.sha256(
        (REPO_ROOT / "utils" / "motion_latent.py").read_bytes()).hexdigest()
    assert record["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
