"""Independent selected-softmax derivatives for exact hard attention."""

import pytest
import torch

from vm_compiler import reference_executor
from vm_compiler.artifact import Artifact
from vm_compiler.graph import OpCode, Operation


@pytest.fixture(params=["artifact", "primitive"])
def attention(request):
    if request.param == "artifact":
        return _artifact_attention
    return reference_executor.attention_2d_ste


def _artifact_attention(queries, keys, values, candidates, competitor_count, temperature):
    # Exercise the actual artifact dispatch with differentiable dynamic Q/K/V.
    # V has one channel in these fixtures; pad to the Q/K width for row routing.
    value_rows = torch.cat((values, torch.zeros_like(values)), dim=-1)
    source = torch.cat((queries, keys, value_rows), dim=1)
    q, n = queries.shape[1], keys.shape[1]
    artifact = Artifact(tensors=(), operations=(
        Operation(OpCode.ROW_ROUTE, (0,), (1,), (0, q)),
        Operation(OpCode.ROW_ROUTE, (0,), (2,), (q, q + n)),
        Operation(OpCode.ROW_ROUTE, (0,), (3,), (q + n, q + 2 * n)),
        Operation(OpCode.HULL_ATTN_2D, (1, 2, 3), (4,),
                  (competitor_count, round(temperature * 1000), *candidates)),
    ))
    return reference_executor.execute_artifact_reference(artifact, source)[..., :1]


def _selected_softmax(queries, keys, values, candidates, k, temperature):
    # Per-row dense development oracle: Python ranking by (score, original row),
    # followed by a plain differentiable softmax expression only on that set.
    rows = []
    hard_rows = []
    for batch in range(len(queries)):
        batch_rows, batch_hard = [], []
        for query in queries[batch]:
            scores = keys[batch] @ query
            selected = sorted(candidates, key=lambda i: (-float(scores[i].detach()), i))[:k]
            probabilities = torch.softmax(scores[selected] / temperature, dim=0)
            batch_rows.append(probabilities @ values[batch, selected])
            batch_hard.append(values[batch, selected[0]].detach())
        rows.append(torch.stack(batch_rows))
        hard_rows.append(torch.stack(batch_hard))
    return torch.stack(rows), torch.stack(hard_rows)


def test_top_two_surrogate_has_literal_query_and_value_gradients(attention):
    q = torch.tensor([[[1., 0.]]], requires_grad=True)
    k = torch.tensor([[[2., 0.], [1., 0.], [0., 0.]]], requires_grad=True)
    v = torch.tensor([[[0.], [10.], [100.]]], requires_grad=True)
    output = attention(q, k, v, [0, 1, 2], 2, 1.)
    soft, _ = _selected_softmax(q, k, v, [0, 1, 2], 2, 1.)
    expected_grads = torch.autograd.grad(soft.sum(), (q, k, v))
    output.sum().backward()
    assert output.item() == 0.
    torch.testing.assert_close(q.grad, torch.tensor([[[-1.9661194, 0.]]]))
    torch.testing.assert_close(v.grad, torch.tensor([[[.7310586], [.2689414], [0.]]]))
    for actual, expected in zip((q.grad, k.grad, v.grad), expected_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_batched_unsorted_ties_match_selected_softmax_for_all_gradients(attention, count):
    q = torch.tensor([[[1., 0.], [0., 0.]], [[0., 1.], [1., 1.]]], requires_grad=True)
    k = torch.tensor([[[1., 0.], [1., 1.], [-1., 2.]],
                      [[0., 1.], [1., 1.], [2., -1.]]], requires_grad=True)
    v = torch.tensor([[[.1], [6.1], [100.]], [[9.], [4.], [-2.]]], requires_grad=True)
    upstream = torch.tensor([[[2.], [-.5]], [[3.], [1.]]])
    candidates = [2, 1, 0]
    actual = attention(q, k, v, candidates, count, .5)
    expected, hard = _selected_softmax(q, k, v, candidates, count, .5)
    actual_grad = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grad = torch.autograd.grad(expected, (q, k, v), upstream)
    assert torch.equal(actual, hard)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf"), 0., -1.])
def test_attention_rejects_nonfinite_or_nonpositive_temperatures(temperature):
    with pytest.raises(ValueError):
        reference_executor.attention_2d_ste(
            torch.ones((1, 1, 2)), torch.ones((1, 1, 2)), torch.ones((1, 1, 1)),
            [0], 1, temperature,
        )


@pytest.mark.parametrize("count", [0, -1, 2])
def test_attention_rejects_competitor_count_outside_candidate_capacity(count):
    with pytest.raises(ValueError):
        reference_executor.attention_2d_ste(
            torch.ones((1, 1, 2)), torch.ones((1, 1, 2)), torch.ones((1, 1, 1)),
            [0], count, 1.,
        )


def test_attention_candidate_subset_uses_original_rows_with_vector_values():
    q = torch.tensor([[[1., 0.], [0., 0.]]], requires_grad=True)
    k = torch.tensor([[[99., 99.], [2., 1.], [88., 88.], [2., -1.]]], requires_grad=True)
    v = torch.tensor([[[100., 200., 300.], [1., 2., 3.],
                       [400., 500., 600.], [4., 5., 6.]]], requires_grad=True)
    actual = reference_executor.attention_2d_ste(q, k, v, [3, 1], 2, 1.)
    soft, hard = _selected_softmax(q, k, v, [3, 1], 2, 1.)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
    expected_grads = torch.autograd.grad(soft.sum(), (q, k, v))
    assert torch.equal(actual, hard)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
