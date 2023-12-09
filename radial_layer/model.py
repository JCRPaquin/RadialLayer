from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .indexing import get_indices


def dist_fn_from_idx(indices: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    :param indices: Indices for calculating a distribution
    :return: A lambda that takes a tensor of probabilities and returns a distribution
    """
    flat_view = indices.view(-1)
    bin_count = indices.shape[0]
    tree_depth = indices.shape[1]

    return lambda probs: torch.hstack([
        probs,
        1-probs
    ]).index_select(1, flat_view).view(probs.shape[0], bin_count, tree_depth).prod(dim=2)

class PartialRadialLayer(nn.Module):
    """
    Implementation of the radial layer without an output projection.
    """

    input_width: int
    inner_width: int
    depth: int

    ray: nn.Parameter
    a_i: nn.Parameter
    w_i: nn.Parameter
    b_i: nn.Parameter

    inner_transforms: nn.Parameter

    quantile_targets: torch.Tensor
    quantiles: torch.Tensor

    make_distribution: torch.jit.ScriptModule

    def __init__(self, input_width: int, depth: int, inner_width: int):
        super().__init__()

        self.input_width = input_width
        self.depth = depth
        self.inner_width = inner_width

        self.ray = nn.Parameter(torch.zeros((1, input_width)), requires_grad=True)
        nn.init.kaiming_normal_(self.ray)

        # Max derivative of sigmoid(a*(x + b)) is at -b
        self.a_i = nn.Parameter(5 * torch.ones((1, 2 ** depth - 1)), requires_grad=False)
        self.w_i = nn.Parameter(torch.zeros((1, 2 ** depth - 1)), requires_grad=False)
        self.b_i = nn.Parameter(torch.ones((1, 2 ** depth - 1)), requires_grad=False)
        self.init_tree_weights()

        self.inner_transforms = nn.Parameter(torch.ones((2 ** depth, input_width, inner_width)), requires_grad=True)
        self.init_transform_weights()

        dist_idx = torch.Tensor(get_indices(depth)).long()
        self.make_distribution = torch.jit.trace(dist_fn_from_idx(dist_idx),
                                                 example_inputs=torch.ones(2, 2**depth-1, requires_grad=True))

    def init_tree_weights(self):
        """
        Initialize the tree weights such that they evenly divide the range [0,1].
        """
        self.b_i[0][0] = -0.5

        used = 1
        for i in range(1, self.depth):
            n_nodes = 2 ** i
            div = float(2 ** (i + 1))

            for node in range(n_nodes):
                idx = used + node
                self.b_i[0][idx] = -(2 * node + 1) / div

            used += n_nodes

        self.quantile_targets = -self.b_i.detach().clone()
        self.quantiles = self.quantile_targets.clone()

        self.b_i = nn.Parameter(torch.log(-self.b_i / (1 + self.b_i)).detach(), requires_grad=False)

    def toggle_inner_node_gradients(self, requires_grad=False):
        self.w_i.requires_grad = requires_grad
        self.b_i.requires_grad = requires_grad

    def init_transform_weights(self):
        # Open to suggestions wrt alternative init schemes
        nn.init.kaiming_normal_(self.inner_transforms)

    def tree_loss(self) -> torch.Tensor:

        node_values = -torch.sigmoid(self.b_i) / (0.5 + torch.sigmoid(self.w_i))

        loss = torch.zeros(1)
        used = 0
        for i in range(self.depth - 1):
            n_nodes = 2 ** i

            for node in range(n_nodes):
                idx = used + node
                left_child_idx = used + n_nodes + 2 * node

                self_value = node_values[0][idx]
                left_value = node_values[0][left_child_idx]
                right_value = node_values[0][left_child_idx + 1]

                loss -= self_value - left_value
                loss -= right_value - self_value

            used += n_nodes

        return loss

    def angles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input vectors to their angles with respect to the axis of this RadialLayer.

        Notes:
        - Masking doesn't confer much in the way of performance and might even contribute to overfitting.
        """
        # Technically not required but I prefer the values to be in [0,1]
        angles = torch.arccos(F.cosine_similarity(x, self.ray)) / torch.pi

        return angles

    def scaled_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get decision values for each inner node of the decision tree,
        then compile those into a probability distribution.
        The original distribution is scaled by a factor of alpha,
        then softmaxed to concentrate prob mass around the max bin.

        Does an absurd amount of memory movement, causing this to take ~45-50% of compute time.
        Most of that time is spent in the torch.cat call.
        """
        angles = self.angles(x)
        node_fns = (0.5 + torch.sigmoid(self.w_i)) * angles.unsqueeze(-1) - torch.sigmoid(self.b_i)
        decisions = F.sigmoid(node_fns * (1 + self.a_i))

        distribution = self.make_distribution(decisions)

        return distribution

    def distribution_by_angles(self, angles: torch.Tensor) -> torch.Tensor:
        node_fns = (0.5 + torch.sigmoid(self.w_i)) * angles.unsqueeze(-1) - torch.sigmoid(self.b_i)
        decisions = F.sigmoid(node_fns * (1 + self.a_i))

        distribution = self.make_distribution(decisions)

        return distribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a soft distribution over total bins, applying all bins for all inputs.
        """
        assert x.shape == (x.shape[0], self.input_width)

        scaled_distribution = self.scaled_distribution(x)

        inner_transform = torch.einsum('bi,liw->blw', x, self.inner_transforms)

        weighted_output = torch.einsum('bl,blo->bo', scaled_distribution, inner_transform)

        return weighted_output

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model with "hard" binning, i.e. use torch.argmax

        Shouldn't we be able to just multiple & truncate? Eventually, yes.
        For now that results in incorrect bins and I haven't debugged it.
        """
        assert x.shape == (x.shape[0], self.input_width)

        scaled_distribution = self.scaled_distribution(x)
        bins = torch.argmax(scaled_distribution, dim=1)

        # self.inner_transforms likely copies memory? Need something more efficient
        inner_fns = torch.einsum('bi,biw->bw', x, self.inner_transforms[bins])

        return inner_fns

    def spread_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage the model's overall bin distribution to tend towards a uniform distribution.
        Individual results should still have sharp-ish peaks,
        but over the total input space try to spread out across available bins.
        """
        assert x.shape == (x.shape[0], self.input_width)

        decisions, angles = self.scaled_distribution(x)
        anti_decisions = F.softmax(decisions.mean(dim=0) * 100, dim=0)
        non_primary_decisions = decisions + (1 - anti_decisions)
        spread = F.kl_div(decisions.log(), F.softmax(non_primary_decisions * 100, dim=1))

        return spread


class MultiAxisRadialLayer(nn.Module):
    """
    Version of the PartialRadialLayer that concats the outputs of several sub-layers together,
    essentially approximating decision trees with multiple linear decision layers but with a
    more parallelizable design.
    """

    def __init__(self, n_axes: int, input_width: int, inner_width: int, depth: int):
        super().__init__()

        self.segments = nn.ModuleList([
            PartialRadialLayer(input_width=input_width,
                               depth=depth,
                               inner_width=inner_width)
            for _ in range(n_axes)
        ])

    def toggle_inner_node_gradients(self, requires_grad=False):
        for segment in self.segments:
            segment.toggle_inner_node_gradients(requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack([
            segment(x) for segment in self.segments
        ])

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack([
            segment.eval_forward(x) for segment in self.segments
        ])

    def ortho_loss(self) -> torch.Tensor:
        loss = torch.zeros(1)

        for i in range(len(self.segments)):
            for j in range(len(self.segments)):
                if i >= j:
                    continue

                loss += F.cosine_similarity(self.segments[i].axis, self.segments[j].axis) ** 2

        return loss

    def spread_loss(self, x: torch.Tensor) -> torch.Tensor:
        spread_loss = torch.hstack([segment.spread_loss(x)
                                    for segment in self.segments]).sum()

        return spread_loss

    def tree_loss(self) -> torch.Tensor:
        return torch.hstack([segment.tree_loss() for segment in self.segments]).sum()