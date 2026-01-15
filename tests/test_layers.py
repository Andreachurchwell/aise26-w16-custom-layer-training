import torch
from layers import LearnedAffine


def test_learned_affine_shape_and_param_count():
    dim = 8
    layer = LearnedAffine(dim)

    x = torch.randn(4, dim)
    y = layer(x)

    # Shape sanity check
    assert y.shape == x.shape

    # Parameter count sanity check: gamma(dim) + beta(dim) = 2*dim
    num_params = sum(p.numel() for p in layer.parameters())
    assert num_params == 2 * dim