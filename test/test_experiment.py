import unittest

import torch.jit

from experiments.mnist_partial_radial import PartialRadialLayerMNISTClassifier


class ExperimentTestCases(unittest.TestCase):
    def test_jit(self):
        model = PartialRadialLayerMNISTClassifier(lr_rate=1e-3)
        model.eval()

        jit_rl1 = torch.jit.script(model.rl1)
        jit_rl1(torch.ones(1, 28*28))


if __name__ == '__main__':
    unittest.main()
