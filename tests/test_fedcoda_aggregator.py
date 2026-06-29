import copy
import random
from collections import OrderedDict
import pathlib
import sys
import types
import unittest

import torch
from torch import nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM = ROOT / "third_party" / "PFLlib" / "system"
sys.path.insert(0, str(SYSTEM))

from flcore.servers.servercoda import FedCodaAggregator, _aggregate_head, _client_consensus_conflicts, _cosine_conflict, _select_shareable_clients, _weighted_head_sum  # noqa: E402


class TinySplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 2, bias=False))
        self.head = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.head(self.base(x))


class TinyNoBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)


class TinyClient:
    def __init__(self, id):
        self.id = id


class TinyNamedSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(OrderedDict([
            ("low", nn.Linear(2, 2, bias=False)),
            ("fc1", nn.Linear(2, 2, bias=False)),
        ]))
        self.head = nn.Linear(2, 2, bias=False)


class TinyNormSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(OrderedDict([
            ("low", nn.Linear(2, 2, bias=False)),
            ("bn", nn.BatchNorm1d(2, affine=True)),
        ]))
        self.head = nn.Linear(2, 2, bias=False)


def _model_with_base(value):
    model = TinySplit()
    with torch.no_grad():
        model.base[0].weight.copy_(torch.eye(2) * value)
        model.head.weight.fill_(99.0)
    return model


def _named_model(low_value, fc1_value):
    model = TinyNamedSplit()
    with torch.no_grad():
        model.base.low.weight.copy_(torch.eye(2) * low_value)
        model.base.fc1.weight.copy_(torch.eye(2) * fc1_value)
    return model


def _norm_model(low_value, bn_weight):
    model = TinyNormSplit()
    with torch.no_grad():
        model.base.low.weight.copy_(torch.eye(2) * low_value)
        model.base.bn.weight.fill_(bn_weight)
        model.base.bn.bias.zero_()
    return model


class FedCodaAggregatorTest(unittest.TestCase):
    def test_conv_conflict_is_computed_per_output_channel(self):
        delta = torch.zeros(2, 1, 1, 1)
        reference = torch.zeros(2, 1, 1, 1)
        delta[0, 0, 0, 0] = 1.0
        reference[0, 0, 0, 0] = -1.0
        delta[1, 0, 0, 0] = 1000.0
        reference[1, 0, 0, 0] = 1000.0

        self.assertAlmostEqual(_cosine_conflict(delta, reference), 0.5)

    def test_keeps_head_local_and_damps_conflicting_backbone_updates(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=1.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
        )
        global_model = _model_with_base(1.0)
        client_a = _model_with_base(2.0)
        client_b = _model_with_base(0.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(
            global_model,
            uploaded_weights=[0.5, 0.5],
            uploaded_models=[client_a, client_b],
        )

        self.assertTrue(torch.allclose(global_model.head.weight, torch.full_like(global_model.head.weight, 99.0)))
        self.assertTrue(torch.allclose(global_model.base[0].weight, torch.eye(2)))
        self.assertGreater(diagnostics["mean_spatial_conflict"], 0.9)
        self.assertLess(diagnostics["mean_eligibility"], 0.1)

    def test_applies_compatible_backbone_updates(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=1.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
        )
        global_model = _model_with_base(1.0)
        client_a = _model_with_base(2.0)
        client_b = _model_with_base(2.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(
            global_model,
            uploaded_weights=[0.5, 0.5],
            uploaded_models=[client_a, client_b],
        )

        self.assertTrue(torch.allclose(global_model.base[0].weight, torch.eye(2) * 2.0))
        self.assertLess(diagnostics["mean_spatial_conflict"], 0.1)
        self.assertGreater(diagnostics["mean_eligibility"], 0.9)

    def test_requires_base_head_split_model(self):
        args = types.SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "BaseHeadSplit"):
            FedCodaAggregator(args, TinyNoBase())

    def test_temporal_state_uses_ema(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
            coda_temporal_ema=0.5,
        )
        global_model = _model_with_base(1.0)
        client = _model_with_base(2.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        aggregator.aggregate(global_model, [1.0], [client])

        expected = torch.eye(2) / torch.linalg.vector_norm(torch.eye(2))
        self.assertTrue(torch.allclose(
            aggregator.previous_delta["0.weight"],
            expected,
        ))

    def test_temporal_state_stores_unit_direction_not_raw_scale(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
        )
        global_model = _model_with_base(1.0)
        client = _model_with_base(11.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        aggregator.aggregate(global_model, [1.0], [client])

        self.assertAlmostEqual(float(torch.linalg.vector_norm(aggregator.previous_delta["0.weight"])), 1.0)

    def test_norm_affine_parameters_stay_local_only(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
        )
        global_model = _norm_model(1.0, 1.0)
        client = _norm_model(2.0, 5.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(global_model, [1.0], [client])

        self.assertTrue(torch.allclose(global_model.base.low.weight, torch.eye(2) * 2.0))
        self.assertTrue(torch.allclose(global_model.base.bn.weight, torch.ones_like(global_model.base.bn.weight)))
        self.assertNotIn("bn.weight", aggregator.previous_delta)
        self.assertEqual(diagnostics["skipped_norm_tensors"], 2)

    def test_spatial_pair_sampling_is_bounded_and_reported(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=1.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
            coda_spatial_pairs=1,
            coda_pair_seed=0,
        )
        global_model = _model_with_base(1.0)
        clients = [_model_with_base(value) for value in (2.0, 0.0, 3.0, -1.0)]

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(global_model, [0.25] * 4, clients)

        self.assertEqual(diagnostics["spatial_pairs"], 1)

    def test_conflict_gate_skips_matching_high_conflict_tensor_only(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
            coda_conflict_gate=0.6,
            coda_gate_keywords="fc1",
        )
        global_model = _named_model(1.0, 1.0)
        client_a = _named_model(2.0, 2.0)
        client_b = _named_model(2.0, -0.5)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(global_model, [0.75, 0.25], [client_a, client_b])

        self.assertTrue(torch.allclose(global_model.base.low.weight, torch.eye(2) * 2.0))
        self.assertTrue(torch.allclose(global_model.base.fc1.weight, torch.eye(2)))
        self.assertEqual(diagnostics["gated_tensors"], 1)

    def test_projection_removes_component_opposite_to_consensus(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
            coda_projection=True,
        )
        aggregate = torch.tensor([-1.0, 2.0])
        consensus = torch.tensor([1.0, 0.0])

        aggregator = FedCodaAggregator(args, _model_with_base(1.0))
        projected, changed = aggregator._project_conflict_free(aggregate, consensus)

        self.assertTrue(changed)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 2.0])))

    def test_projection_uses_weighted_consensus_direction_in_aggregate(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.0,
            coda_projection=True,
        )
        global_model = _model_with_base(1.0)
        aligned = _model_with_base(2.0)
        outlier = _model_with_base(-9.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(global_model, [0.99, 0.01], [aligned, outlier])

        self.assertGreater(float(global_model.base[0].weight[0, 0].detach()), 1.5)
        self.assertEqual(diagnostics["projected_tensors"], 0)

    def test_adaptive_gate_tightens_after_accuracy_drop_and_relaxes_after_gain(self):
        args = types.SimpleNamespace(
            coda_adaptive_gate=True,
            coda_conflict_gate=0.6,
            coda_gate_step=0.05,
            coda_gate_min=0.4,
            coda_gate_max=0.8,
        )

        aggregator = FedCodaAggregator(args, _model_with_base(1.0))
        aggregator.update_accuracy(0.8)
        aggregator.update_accuracy(0.7)
        self.assertAlmostEqual(aggregator.conflict_gate, 0.55)

        aggregator.update_accuracy(0.75)
        self.assertAlmostEqual(aggregator.conflict_gate, 0.6)

    def test_layer_freeze_skips_high_conflict_tensor_even_with_min_eligibility(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.7,
            coda_layer_freeze=True,
            coda_freeze_spatial=0.6,
            coda_freeze_temporal=1.0,
            coda_freeze_patience=1,
        )
        global_model = _model_with_base(1.0)
        client_a = _model_with_base(2.0)
        client_b = _model_with_base(0.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        diagnostics = aggregator.aggregate(global_model, [0.75, 0.25], [client_a, client_b])

        self.assertTrue(torch.allclose(global_model.base[0].weight, torch.eye(2)))
        self.assertEqual(diagnostics["frozen_tensors"], 1)

    def test_layer_freeze_waits_for_persistent_conflict(self):
        args = types.SimpleNamespace(
            coda_spatial_strength=0.0,
            coda_temporal_strength=0.0,
            coda_min_eligibility=0.7,
            coda_layer_freeze=True,
            coda_freeze_spatial=0.6,
            coda_freeze_temporal=1.0,
            coda_freeze_patience=2,
        )
        global_model = _model_with_base(1.0)
        client_a = _model_with_base(2.0)
        client_b = _model_with_base(0.0)

        aggregator = FedCodaAggregator(args, copy.deepcopy(global_model))
        first = aggregator.aggregate(global_model, [0.75, 0.25], [client_a, client_b])
        second = aggregator.aggregate(global_model, [0.75, 0.25], [client_a, client_b])

        self.assertEqual(first["frozen_tensors"], 0)
        self.assertEqual(second["frozen_tensors"], 1)

    def test_shareability_selection_mixes_top_score_and_random_clients(self):
        clients = [TinyClient(i) for i in range(4)]
        scores = {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.0}
        rng = random.Random(0)

        selected = _select_shareable_clients(clients, scores, join_count=2, random_ratio=0.5, rng=rng)

        selected_ids = {client.id for client in selected}
        self.assertEqual(len(selected), 2)
        self.assertIn(1, selected_ids)

    def test_client_selection_conflict_uses_consensus_update_direction(self):
        global_model = _model_with_base(1.0)
        aligned = _model_with_base(2.0)
        opposite = _model_with_base(0.0)

        conflicts = _client_consensus_conflicts(global_model, [0.75, 0.25], [aligned, opposite])

        self.assertLess(conflicts[0], 0.1)
        self.assertGreater(conflicts[1], 0.9)

    def test_server_builds_reference_head_from_uploaded_personalized_heads(self):
        global_model = _model_with_base(1.0)
        client_a = _model_with_base(1.0)
        client_b = _model_with_base(1.0)
        with torch.no_grad():
            global_model.head.weight.zero_()
            client_a.head.weight.fill_(2.0)
            client_b.head.weight.fill_(4.0)

        _aggregate_head(global_model, [0.25, 0.75], [client_a, client_b])

        self.assertTrue(torch.allclose(global_model.head.weight, torch.full_like(global_model.head.weight, 3.5)))

    def test_weighted_head_sum_builds_client_specific_collaborative_head(self):
        client_a = _model_with_base(1.0)
        client_b = _model_with_base(1.0)
        with torch.no_grad():
            client_a.head.weight.fill_(2.0)
            client_b.head.weight.fill_(4.0)

        head = _weighted_head_sum([client_a.head, client_b.head], [0.25, 0.75])

        self.assertTrue(torch.allclose(head.weight, torch.full_like(head.weight, 3.5)))


if __name__ == "__main__":
    unittest.main()
