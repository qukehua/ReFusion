"""Regressions for frequency graphs, relation conditioning and Table VII."""

import pytest
import torch
from torch import nn

from models.condition_two_stage import GraphConvBlock, MotionTransformerTwoStage, timestep_embedding


def small_model(*, observed=3, predicted=5, layers=2, stage1=1, cond_feats=9):
    return MotionTransformerTwoStage(
        input_feats=6, cond_feats=cond_feats, human_cond_joint_num=2,
        num_frames=predicted, cond_frames=observed, latent_dim=16,
        ff_size=32, num_layers=layers, stage1_num_layers=stage1,
        num_heads=2, dropout=0.0,
    )


def test_frequency_graph_transfers_information_and_receives_gradients():
    graph = GraphConvBlock(num_nodes=2, hidden_size=3, num_frames=4)
    with torch.no_grad():
        graph.channel_proj.weight.copy_(torch.eye(3))
    x = torch.zeros(1, 4, 2, 3)
    x[0, 0, 0] = torch.tensor([1.0, -1.0, 0.0])
    x[0, 2, 0] = torch.tensor([0.0, 1.0, -1.0])
    changed = x.clone()
    changed[0, 0, 0] = torch.tensor([1.0, 0.0, -1.0])

    # With identity frequency adjacency, a frequency cannot affect another.
    torch.testing.assert_close(graph(x)[:, 2], graph(changed)[:, 2], rtol=0, atol=0)
    with torch.no_grad():
        graph.frequency_adj[2, 0] = 1.0
    assert not torch.equal(graph(x)[:, 2], graph(changed)[:, 2])

    graph(x)[:, 2].sum().backward()
    assert graph.frequency_adj.grad[2, 0].abs() > 0
    assert torch.isfinite(graph.frequency_adj.grad).all()
    assert graph.spatial_adj.grad.abs().sum() > 0


@pytest.mark.parametrize("observed,predicted,retained", [(3, 5, 5), (6, 2, 2), (3, 5, 4)])
@pytest.mark.parametrize("cond_feats", [6, 9])
def test_unequal_horizons_have_real_condition_gradient(observed, predicted, retained, cond_feats):
    model = small_model(observed=observed, predicted=predicted, cond_feats=cond_feats)
    # DiT intentionally initializes the readout to zero; enable it to inspect
    # gradient flow through both learned observation-to-future projections.
    nn.init.normal_(model.final_layer.out_proj.weight, std=0.1)
    x = torch.randn(2, retained, 6, requires_grad=True)
    condition = torch.randn(2, observed, cond_feats, requires_grad=True)
    output = model(x, torch.tensor([1, 7]), mod=condition)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    (output - torch.randn_like(output)).square().mean().backward()

    for gradient in (x.grad, condition.grad, model.human_frequency_proj.weight.grad,
                     model.human_gcn[0].frequency_adj.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    if cond_feats > 6:
        assert model.interaction_frequency_proj.weight.grad.abs().sum() > 0


def test_stage_two_global_modulation_retains_both_relations():
    model = small_model(layers=4, stage1=2)
    observed = torch.randn(2, 3, 9)
    x = torch.randn(2, 5, 6)
    t = torch.tensor([2, 7])
    human_tokens, _, intra = model.encode_human_condition(observed)
    _, inter = model.encode_interaction_condition(observed, human_tokens)
    base = model.t_embedder(timestep_embedding(t, model.latent_dim))
    assert intra.abs().sum() > 0 and inter.abs().sum() > 0

    seen = []
    handles = [block.register_forward_pre_hook(lambda _module, args: seen.append(args[1].detach().clone()))
               for block in model.blocks]
    final_seen = []
    handles.append(model.final_layer.register_forward_pre_hook(
        lambda _module, args: final_seen.append(args[1].detach().clone())))
    try:
        model(x, t, mod=observed)
    finally:
        for handle in handles:
            handle.remove()

    assert len(seen) == 4
    for actual in seen[:2]:
        torch.testing.assert_close(actual, base + intra)
    for actual in seen[2:] + final_seen:
        torch.testing.assert_close(actual, base + intra + inter)


def test_all_intra_endpoint_is_independent_of_robot_observations():
    model = small_model(layers=3, stage1=3)
    nn.init.normal_(model.final_layer.out_proj.weight, std=0.1)
    observed = torch.randn(2, 3, 9)
    changed = observed.clone()
    changed[:, :, 6:] = 100 * torch.randn_like(changed[:, :, 6:])
    x = torch.randn(2, 5, 6)
    t = torch.tensor([1, 3])
    torch.testing.assert_close(model(x, t, mod=observed), model(x, t, mod=changed), rtol=0, atol=0)


@pytest.mark.parametrize("stage1", range(10))
def test_table_vii_factory_constructs_all_ten_allocations(tiny_cfg, stage1):
    from utils.script import create_model_and_diffusion

    tiny_cfg.num_layers = 9
    tiny_cfg.stage1_num_layers = stage1
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    assert type(model) is MotionTransformerTwoStage
    assert len(model.blocks) == 9
    assert model.stage1_num_layers == stage1
    assert model.cond_frames == tiny_cfg.t_his
    assert diffusion.motion_size == (tiny_cfg.n_pre, 6)
    output = model(torch.randn(1, tiny_cfg.n_pre, 6), torch.tensor([3]), mod=torch.randn(1, tiny_cfg.t_his, 9))
    assert output.shape == (1, tiny_cfg.n_pre, 6)
    output.square().sum().backward()


@pytest.mark.parametrize("stage1", [-1, 10])
def test_stage_allocation_rejects_values_outside_table_bounds(stage1):
    with pytest.raises(ValueError):
        small_model(layers=9, stage1=stage1)


def test_condition_length_mismatch_is_reported():
    model = small_model()
    with pytest.raises(ValueError, match="[Oo]bserved|[Cc]ondition"):
        model(torch.randn(1, 5, 6), torch.tensor([1]), mod=torch.randn(1, 5, 9))


def test_factory_truncation_retains_full_future_projection_capacity(tiny_cfg):
    from utils.script import create_model_and_diffusion

    tiny_cfg.n_pre = 2
    model, diffusion = create_model_and_diffusion(tiny_cfg)
    assert model.num_frames == tiny_cfg.t_pred
    assert model.human_frequency_proj.weight.shape == (tiny_cfg.t_pred, tiny_cfg.t_his)
    assert diffusion.motion_size == (2, 6)
    output = model(torch.randn(1, 2, 6), torch.tensor([3]), mod=torch.randn(1, tiny_cfg.t_his, 9))
    assert output.shape == (1, 2, 6)
