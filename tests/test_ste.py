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
