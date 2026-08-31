import pytest
import torch

from vm_compiler.ste import fp4_ste, masked_hardmax_ste


def test_masked_hardmax_uses_lowest_tied_index_and_soft_surrogate_gradient():
    logits = torch.tensor([[0.0, 0.0, 7.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])

    hard = masked_hardmax_ste(logits, mask, temperature=0.5)
    hard.backward(torch.tensor([[2.0, -1.0, 99.0]]))

    assert hard.tolist() == [[1.0, 0.0, 0.0]]
    assert logits.grad.tolist() == [[1.5, -1.5, 0.0]]


def test_fp4_ste_quantizes_forward_and_passes_floating_gradient():
    values = torch.tensor([0.49, 0.76, -2.4], requires_grad=True)

    quantized = fp4_ste(values, scale=1.0)
    quantized.sum().backward()

    assert quantized.tolist() == [0.5, 1.0, -2.0]
    assert values.grad.tolist() == [1.0, 1.0, 1.0]


def test_random_hardmax_is_exactly_binary_without_ste_cancellation():
    logits = torch.randn((100, 3), generator=torch.Generator().manual_seed(0), requires_grad=True)
    actual = masked_hardmax_ste(logits, torch.ones_like(logits, dtype=torch.bool), 1.)
    expected = torch.nn.functional.one_hot(logits.argmax(-1), 3).to(logits.dtype)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("scale", [1., .5, 2.])
def test_fp4_exact_saturation_and_identity_gradient_for_finite_extremes(scale):
    values = torch.tensor([1.1, 6.1, 1e10, -1e10, -6.1, -1.1]) * scale
    values.requires_grad_()
    actual = fp4_ste(values, scale)
    upstream = torch.tensor([1., -2., 3., 4., -5., 6.])
    actual.backward(upstream)
    assert torch.equal(actual, torch.tensor([1., 6., 6., -6., -6., -1.]) * scale)
    assert torch.equal(values.grad, upstream)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf"), 0., -1.])
def test_hardmax_rejects_nonfinite_or_nonpositive_temperature(temperature):
    with pytest.raises(ValueError):
        masked_hardmax_ste(torch.ones((2, 3)), torch.ones((2, 3), dtype=torch.bool), temperature)


def test_hardmax_rejects_an_empty_mask_row():
    with pytest.raises(ValueError):
        masked_hardmax_ste(torch.ones((2, 3)), torch.tensor([[True, False, False], [False] * 3]), 1.)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf"), 0., -1.])
def test_fp4_rejects_nonfinite_or_nonpositive_scale(scale):
    with pytest.raises(ValueError):
        fp4_ste(torch.tensor([1.1]), scale)
