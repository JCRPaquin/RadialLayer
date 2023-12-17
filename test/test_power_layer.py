import unittest

import torch

from radial_layer.power_layer import PowerLayer


class PowerLayerTests(unittest.TestCase):

    def test_power_layer_forward(self):
        pw_layer = PowerLayer(input_width=8, power=3)
        out = pw_layer(torch.ones(8, 8))
        self.assertEqual(out.shape, (8, 8*3))


if __name__ == '__main__':
    unittest.main()
