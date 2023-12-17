import unittest

import torch

from radial_layer.model import PartialRadialLayer

class PartialRadialLayerTests(unittest.TestCase):

    def create_prl(self):
        return PartialRadialLayer(input_width=16, inner_width=8, depth=3)

    def test_init(self):
        _ = self.create_prl()

    def test_distribution(self):
        prl = self.create_prl()
        dist = prl.distribution_by_angles(torch.linspace(0, 1, 20))

        self.assertEqual(dist.shape, (20, 8))
        self.assertAlmostEquals((torch.ones(20) - dist.sum(dim=1)).mean().item(), 0)

    def test_output_shape(self):
        prl = self.create_prl()
        in_tensor = torch.ones(4, prl.input_width)
        out_tensor = prl(in_tensor)

        self.assertEqual(out_tensor.shape, (4, prl.inner_width))

        out_tensor2 = prl.eval_forward(in_tensor)

        self.assertEqual(out_tensor2.shape, (4, prl.inner_width))

    def test_ema_init(self):
        prl = self.create_prl()
        print(prl.ema_weights)
        print(prl.ema_history)
        old_history = torch.empty_like(prl.ema_history).copy_(prl.ema_history)
        print(prl.spread_penalty_multiplier)
        print(prl.spread_loss(torch.rand(64, 16)))
        history_diff = ((prl.ema_history-old_history)**2).mean()
        print(history_diff)
        print(prl.ema_history)
        self.assertNotEqual(history_diff, 0)
        self.assertFalse(prl.ema_weights.requires_grad)
        self.assertFalse(prl.ema_history.requires_grad)
        self.assertFalse(prl.spread_penalty_multiplier.requires_grad)


if __name__ == '__main__':
    unittest.main()
