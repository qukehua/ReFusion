"""Reviewer 3: equations, documented metrics, and checkpoint provenance."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from models.condition_two_stage import GraphConvBlock, MotionTransformerTwoStage
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_all_metrics
from utils.script import create_model_and_diffusion


def test_graph_matches_explicit_frequency_spatial_kronecker_operator():
    block = GraphConvBlock(2, 4, 3, input_size=3)
    x = torch.randn(2, 3, 2, 3)
    with torch.no_grad():
        block.spatial_adj.copy_(torch.randn(2, 2))
        block.frequency_adj.copy_(torch.randn(3, 3))
    channels = block.channel_proj(block.norm(x))
    operator = torch.kron(block.frequency_adj, block.spatial_adj)
    expected = torch.relu(operator @ channels.reshape(2, 6, 4)).reshape(2, 3, 2, 4)
    torch.testing.assert_close(block(x), expected)


@pytest.mark.parametrize('partner_joints', [1, 3, 7])
def test_frequency_attention_explicitly_handles_unequal_joint_counts(partner_joints):
    model = MotionTransformerTwoStage(6, cond_feats=3 * (2 + partner_joints), num_frames=5,
                                     cond_frames=3, latent_dim=16, num_heads=2, num_layers=2,
                                     stage1_num_layers=1, dropout=0)
    tokens = torch.randn(2, 3, partner_joints, 16, requires_grad=True)
    context = model.frequency_partner_context(tokens)
    assert context.shape == (2, 3, 2, 16)
    torch.testing.assert_close(context[:, :, 0], tokens.mean(dim=2))
    context.sum().backward()
    assert (tokens.grad != 0).all()
    # Exercise the complete attention path, rather than only its shape adapter.
    nn.init.normal_(model.final_layer.out_proj.weight, std=0.1)
    observed = torch.randn(2, 3, 3 * (2 + partner_joints), requires_grad=True)
    model(torch.randn(2, 5, 6), torch.tensor([1, 2]), observed).square().mean().backward()
    assert observed.grad[:, :, 6:].abs().sum() > 0


def test_checkpoint_contract_rejects_same_shape_different_stage_allocation(tiny_cfg, tmp_path):
    model, _ = create_model_and_diffusion(tiny_cfg)
    path = tmp_path / 'versioned.pt'
    save_checkpoint(path, model, tiny_cfg, epoch=1, validation_loss=0.1)
    restored, _ = create_model_and_diffusion(tiny_cfg)
    load_checkpoint(path, restored, tiny_cfg)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    other = copy.copy(tiny_cfg)
    other.stage1_num_layers = 0
    with pytest.raises(ValueError, match='configuration mismatch'):
        load_checkpoint(path, restored, other)
    torch.save(model.state_dict(), tmp_path / 'old.pt')
    with pytest.raises(ValueError, match='implementation version'):
        load_checkpoint(tmp_path / 'old.pt', restored, tiny_cfg)


def test_metric_reductions_have_known_values_and_lower_median():
    pred = torch.zeros(4, 1, 3)
    pred[:, 0, 0] = torch.tensor([0, 2, 4, 6])
    gt = np.zeros((1, 1, 3), dtype=np.float32)
    multi = gt.copy()
    multi[0, 0, 0] = 10
    result = [float(v) for v in compute_all_metrics(pred, gt, multi)]
    np.testing.assert_allclose(result, [20 / 6, 0, 0, 4, 4, 2, 2, 6, 6, 6, 6, 10, 10])
    with pytest.raises(ValueError, match='nonempty'):
        compute_all_metrics(pred, gt, multi[:0])


def test_chico_training_prefers_validation_over_test(tiny_cfg):
    import logging
    from utils.training import Trainer

    class Split:
        def __init__(self):
            self.used = False

        def sampling_generator(self, **kwargs):
            self.used = True
            return iter(())

    model, diffusion = create_model_and_diffusion(tiny_cfg)
    val, test = Split(), Split()
    trainer = Trainer(model, diffusion, {'val': val, 'test': test}, tiny_cfg,
                      logging.getLogger(__name__), None)
    trainer.before_val_step()
    assert val.used and not test.used
