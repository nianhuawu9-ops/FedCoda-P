import pathlib
import sys
import types
import unittest

import torch
from torch import nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM = ROOT / "third_party" / "PFLlib" / "system"
sys.path.insert(0, str(SYSTEM))

from flcore.clients.clientcoda import clientCoda  # noqa: E402
from flcore.servers.servercoda import _average_prototypes, _ema_prototypes  # noqa: E402


class TinySplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.head(self.base(x))


def _args():
    model = TinySplit()
    with torch.no_grad():
        model.base.weight.copy_(torch.eye(2))
        model.head.weight.copy_(torch.eye(2))
    return types.SimpleNamespace(
        model=model,
        algorithm="FedCoda-P",
        dataset="dummy",
        device="cpu",
        num_classes=2,
        save_folder_name="items",
        batch_size=2,
        local_learning_rate=0.1,
        local_epochs=1,
        few_shot=0,
        learning_rate_decay_gamma=0.99,
        learning_rate_decay=False,
        plocal_epochs=1,
        coda_use_prototypes=True,
        coda_proto_weight=0.1,
        coda_proto_warmup=0,
        coda_proto_ramp=1,
        coda_align_weight=0.0,
        coda_anchor_tau=0.2,
        coda_local_init_alpha=1.0,
        coda_local_init_min_alpha=0.5,
        coda_local_init_warmup=0,
        coda_head_init_alpha=0.0,
        coda_head_init_warmup=0,
    )


class FedCodaPrototypeTest(unittest.TestCase):
    def test_server_averages_class_prototypes_by_count(self):
        averaged = _average_prototypes([
            {0: (torch.tensor([1.0, 3.0]), 2)},
            {0: (torch.tensor([5.0, 7.0]), 1), 1: (torch.tensor([2.0, 2.0]), 4)},
        ])

        self.assertTrue(torch.allclose(averaged[0], torch.tensor([7.0 / 3.0, 13.0 / 3.0])))
        self.assertTrue(torch.allclose(averaged[1], torch.tensor([2.0, 2.0])))

    def test_client_uses_global_prototypes_for_head_loss(self):
        client = clientCoda(_args(), id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)
        client.set_prototypes({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])})

        loss = client._prototype_head_loss()

        self.assertGreater(float(loss.detach()), 0.0)

    def test_client_ignores_global_prototypes_during_warmup(self):
        args = _args()
        args.coda_proto_warmup = 2
        client = clientCoda(args, id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)
        client.set_prototypes({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])})

        client.set_round(1)
        warmup_loss = client._prototype_head_loss()
        client.set_round(2)
        active_loss = client._prototype_head_loss()

        self.assertEqual(float(warmup_loss.detach()), 0.0)
        self.assertGreater(float(active_loss.detach()), 0.0)

    def test_client_scales_prototype_weight_by_ramp_and_batch_count(self):
        args = _args()
        args.coda_proto_weight = 0.2
        args.coda_proto_warmup = 10
        args.coda_proto_ramp = 5
        client = clientCoda(args, id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)

        client.set_round(9)
        before_warmup = client._prototype_weight(batch_count=4)
        client.set_round(10)
        first_active = client._prototype_weight(batch_count=4)
        client.set_round(14)
        fully_active = client._prototype_weight(batch_count=4)

        self.assertEqual(before_warmup, 0.0)
        self.assertAlmostEqual(first_active, 0.01)
        self.assertAlmostEqual(fully_active, 0.05)

    def test_client_anchor_alignment_prefers_matching_class_features(self):
        args = _args()
        args.coda_align_weight = 0.1
        client = clientCoda(args, id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)
        client.set_prototypes({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])})

        aligned = client._alignment_loss(torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1]))
        swapped = client._alignment_loss(torch.tensor([[0.0, 3.0], [3.0, 0.0]]), torch.tensor([0, 1]))

        self.assertLess(float(aligned.detach()), float(swapped.detach()))

    def test_server_updates_prototypes_with_ema(self):
        previous = {0: torch.tensor([1.0, 0.0])}
        current = {0: torch.tensor([0.0, 1.0]), 1: torch.tensor([2.0, 2.0])}

        updated = _ema_prototypes(previous, current, momentum=0.5)

        self.assertTrue(torch.allclose(updated[0], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(updated[1], torch.tensor([2.0, 2.0])))

    def test_client_adaptive_local_initialization_blends_global_and_local_base(self):
        args = _args()
        args.coda_local_init_min_alpha = 0.0
        client = clientCoda(args, id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)
        client.set_local_init_alpha(0.25)
        with torch.no_grad():
            client.model.base.weight.copy_(torch.eye(2) * 3.0)

        global_model = TinySplit()
        with torch.no_grad():
            global_model.base.weight.copy_(torch.eye(2))

        client.set_parameters(global_model)

        self.assertTrue(torch.allclose(client.model.base.weight, torch.eye(2) * 2.5))

    def test_client_can_soft_initialize_personalized_head_from_reference_head(self):
        args = _args()
        args.coda_head_init_alpha = 0.25
        client = clientCoda(args, id=0, train_samples=2, test_samples=2, train_slow=False, send_slow=False)
        with torch.no_grad():
            client.model.head.weight.copy_(torch.eye(2) * 3.0)

        global_model = TinySplit()
        with torch.no_grad():
            global_model.base.weight.copy_(torch.eye(2))
            global_model.head.weight.copy_(torch.eye(2))

        client.set_parameters(global_model)

        self.assertTrue(torch.allclose(client.model.head.weight, torch.eye(2) * 2.5))

    def test_client_collects_fedpac_statistics_with_prototypes(self):
        client = clientCoda(_args(), id=0, train_samples=4, test_samples=2, train_slow=False, send_slow=False)
        client.load_train_data = lambda: [
            (torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]), torch.tensor([0, 0, 1, 1]))
        ]

        client.collect_prototypes()

        self.assertTrue(torch.allclose(client.local_prototypes[0][0], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.allclose(client.local_prototypes[1][0], torch.tensor([0.0, 3.0])))
        self.assertTrue(torch.allclose(client.h.cpu(), torch.tensor([[1.0, 0.0], [0.0, 1.5]])))
        self.assertGreater(client.V, 0.0)


if __name__ == "__main__":
    unittest.main()
