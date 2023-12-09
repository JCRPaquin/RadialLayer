import timeit
import unittest

import torch

from radial_layer.model import dist_fn_from_idx
from radial_layer.indexing import get_indices, get_indexing_matrix


@torch.jit.script
def replace_zeros(x: torch.Tensor) -> torch.Tensor:
    x[x == 0] = 1.0
    return x

class IndexingTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = 128
        self.depth = 11
        self.indices = get_indices(self.depth)
        self.data = torch.rand(self.batch_size, 2 ** self.depth - 1)

    def test_indexing_matrix(self):
        indexing_matrix = get_indexing_matrix(self.indices)
        mask = indexing_matrix != 0
        addition_matrix = indexing_matrix.clone()
        addition_matrix[mask] = 0
        addition_matrix[~mask] = 1.
        indexing_matrix_fn = lambda probs: (addition_matrix+torch.einsum('bp,op->bop',
                                                        torch.hstack([probs, 1-probs]),
                                                        indexing_matrix)).prod(dim=2)
        indexing_matrix_fn = torch.jit.trace(indexing_matrix_fn, example_inputs=self.data)
        indexing_matrix_lambda = lambda: indexing_matrix_fn(self.data)
        print(indexing_matrix_lambda()[0])

    def test_cpu_time(self):
        select_fn = dist_fn_from_idx(torch.Tensor(self.indices).long())
        select_profiling_lambda = lambda: select_fn(self.data)

        select_time = timeit.timeit(select_profiling_lambda, number=1000)
        print(f"Time for select: {select_time}")

        indexing_matrix = get_indexing_matrix(self.indices)
        addition_matrix = indexing_matrix.clone()
        addition_matrix = torch.abs(addition_matrix-1)
        indexing_matrix_fn = lambda probs: (addition_matrix+torch.einsum('bp,op->bop',
                                                        torch.hstack([probs, 1-probs]),
                                                        indexing_matrix)).prod(dim=2)
        indexing_matrix_fn = torch.jit.trace(indexing_matrix_fn, example_inputs=self.data)
        indexing_matrix_lambda = lambda: indexing_matrix_fn(self.data)

        indexing_matrix_time = timeit.timeit(indexing_matrix_lambda, number=1000)
        print(f"Time for indexing matrix: {indexing_matrix_time}")

        self.assertAlmostEqual(((select_profiling_lambda()-indexing_matrix_lambda())**2).mean().item(), 0)

    def test_cuda_time(self):
        cuda_data = self.data.cuda()
        select_fn = dist_fn_from_idx(torch.Tensor(self.indices).long().cuda())
        select_profiling_lambda = lambda: select_fn(cuda_data)

        select_time = timeit.timeit(select_profiling_lambda)
        print(f"Time for select: {select_time}")

        indexing_matrix = get_indexing_matrix(self.indices).cuda()
        addition_matrix = indexing_matrix.clone()
        addition_matrix = torch.abs(addition_matrix-1)
        indexing_matrix_fn = lambda probs: (addition_matrix+torch.einsum('bp,op->bop',
                                                        torch.hstack([probs, 1-probs]),
                                                        indexing_matrix)).prod(dim=2)
        indexing_matrix_fn = torch.jit.trace(indexing_matrix_fn, example_inputs=cuda_data)
        indexing_matrix_lambda = lambda: indexing_matrix_fn(cuda_data)

        indexing_matrix_time = timeit.timeit(indexing_matrix_lambda)
        print(f"Time for indexing matrix: {indexing_matrix_time}")



if __name__ == '__main__':
    unittest.main()
