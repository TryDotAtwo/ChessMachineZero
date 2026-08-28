#include "cmz/hullkv.h"

#include <torch/autograd.h>

namespace cmz {
namespace {

class HullAttentionSte final : public torch::autograd::Function<HullAttentionSte> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* context,
        torch::Tensor queries,
        torch::Tensor keys,
        torch::Tensor values,
        torch::Tensor hull_indices,
        std::int64_t competitor_count,
        double temperature) {
        TORCH_CHECK(temperature > 0.0, "HullKV temperature must be positive");
        TORCH_CHECK(keys.size(0) == values.size(0), "HullKV keys and values must align");
        const auto competitors =
            support_topk_2d_cuda(queries, keys, hull_indices, competitor_count);
        const auto winners = competitors.select(1, 0);
        context->save_for_backward({queries, keys, values, competitors});
        context->saved_data["temperature"] = temperature;
        return values.index_select(0, winners);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        const auto& queries = saved[0];
        const auto& keys = saved[1];
        const auto& values = saved[2];
        const auto& competitors = saved[3];
        const auto temperature = context->saved_data["temperature"].toDouble();
        const auto query_count = queries.size(0);
        const auto competitor_count = competitors.size(1);
        const auto value_width = values.size(1);
        const auto flat_indices = competitors.reshape({-1});
        const auto selected_keys =
            keys.index_select(0, flat_indices).reshape({query_count, competitor_count, 2});
        const auto selected_values = values.index_select(0, flat_indices)
                                         .reshape({query_count, competitor_count, value_width});
        const auto raw_logits = (selected_keys * queries.unsqueeze(1)).sum(-1);
        const auto probabilities = torch::softmax(raw_logits / temperature, -1);
        const auto& upstream = gradients[0];
        const auto utilities = (selected_values * upstream.unsqueeze(1)).sum(-1);
        const auto expected_utility = (probabilities * utilities).sum(-1, true);
        const auto logit_gradient =
            probabilities * (utilities - expected_utility) / temperature;

        const auto query_gradient =
            (logit_gradient.unsqueeze(-1) * selected_keys).sum(1);
        auto key_gradient = torch::zeros_like(keys);
        key_gradient.index_add_(
            0,
            flat_indices,
            (logit_gradient.unsqueeze(-1) * queries.unsqueeze(1)).reshape({-1, 2}));
        auto value_gradient = torch::zeros_like(values);
        value_gradient.index_add_(
            0,
            flat_indices,
            (probabilities.unsqueeze(-1) * upstream.unsqueeze(1))
                .reshape({-1, value_width}));
        return {
            query_gradient,
            key_gradient,
            value_gradient,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
        };
    }
};

class BatchedHullAttentionSte final
    : public torch::autograd::Function<BatchedHullAttentionSte> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* context,
        torch::Tensor queries,
        torch::Tensor keys,
        torch::Tensor values,
        torch::Tensor hull_indices,
        std::int64_t competitor_count,
        double temperature) {
        TORCH_CHECK(queries.dim() == 3 && queries.size(2) == 2);
        TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 2);
        TORCH_CHECK(values.dim() == 3 && values.size(0) == queries.size(0));
        TORCH_CHECK(values.size(1) == keys.size(0));
        TORCH_CHECK(temperature > 0.0, "HullKV temperature must be positive");
        const auto batch_count = queries.size(0);
        const auto query_count = queries.size(1);
        const auto key_count = keys.size(0);
        const auto value_width = values.size(2);
        const auto competitors = support_topk_2d_cuda(
            queries.reshape({batch_count * query_count, 2}),
            keys,
            hull_indices,
            competitor_count);
        const auto batch_offsets =
            torch::arange(batch_count, competitors.options())
                .reshape({batch_count, 1})
                .expand({batch_count, query_count})
                .reshape({-1}) * key_count;
        const auto winner_indices = competitors.select(1, 0) + batch_offsets;
        context->save_for_backward({queries, keys, values, competitors});
        context->saved_data["temperature"] = temperature;
        return values.reshape({batch_count * key_count, value_width})
            .index_select(0, winner_indices)
            .reshape({batch_count, query_count, value_width});
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        const auto& queries = saved[0];
        const auto& keys = saved[1];
        const auto& values = saved[2];
        const auto& competitors = saved[3];
        const auto temperature = context->saved_data["temperature"].toDouble();
        const auto batch_count = queries.size(0);
        const auto query_count = queries.size(1);
        const auto key_count = keys.size(0);
        const auto value_width = values.size(2);
        const auto competitor_count = competitors.size(1);
        const auto flat_queries = queries.reshape({batch_count * query_count, 2});
        const auto flat_indices = competitors.reshape({-1});
        const auto selected_keys = keys.index_select(0, flat_indices).reshape(
            {batch_count * query_count, competitor_count, 2});
        const auto batch_offsets =
            torch::arange(batch_count, competitors.options())
                .reshape({batch_count, 1, 1})
                .expand({batch_count, query_count, competitor_count}) * key_count;
        const auto value_indices = competitors.reshape(
            {batch_count, query_count, competitor_count}) + batch_offsets;
        const auto selected_values = values.reshape({batch_count * key_count, value_width})
                                         .index_select(0, value_indices.reshape({-1}))
                                         .reshape({batch_count * query_count,
                                                   competitor_count,
                                                   value_width});
        const auto logits = (selected_keys * flat_queries.unsqueeze(1)).sum(-1);
        const auto probabilities = torch::softmax(logits / temperature, -1);
        const auto upstream = gradients[0].reshape({batch_count * query_count, value_width});
        const auto utilities = (selected_values * upstream.unsqueeze(1)).sum(-1);
        const auto logit_gradient = probabilities *
            (utilities - (probabilities * utilities).sum(-1, true)) / temperature;

        const auto query_gradient =
            (logit_gradient.unsqueeze(-1) * selected_keys).sum(1).reshape_as(queries);
        auto key_gradient = torch::zeros_like(keys);
        key_gradient.index_add_(
            0,
            flat_indices,
            (logit_gradient.unsqueeze(-1) * flat_queries.unsqueeze(1)).reshape({-1, 2}));
        auto value_gradient = torch::zeros_like(values).reshape(
            {batch_count * key_count, value_width});
        value_gradient.index_add_(
            0,
            value_indices.reshape({-1}),
            (probabilities.unsqueeze(-1) * upstream.unsqueeze(1)).reshape({-1, value_width}));
        return {
            query_gradient,
            key_gradient,
            value_gradient.reshape_as(values),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
        };
    }
};

class DynamicBatchedHullAttentionSte final
    : public torch::autograd::Function<DynamicBatchedHullAttentionSte> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* context,
        torch::Tensor queries,
        torch::Tensor keys,
        torch::Tensor values,
        torch::Tensor hull_indices,
        std::int64_t competitor_count,
        double temperature) {
        TORCH_CHECK(queries.dim() == 3 && queries.size(2) == 2);
        TORCH_CHECK(keys.dim() == 3 && keys.size(0) == queries.size(0) && keys.size(2) == 2);
        TORCH_CHECK(values.dim() == 3 && values.size(0) == queries.size(0));
        TORCH_CHECK(values.size(1) == keys.size(1));
        TORCH_CHECK(temperature > 0.0, "HullKV temperature must be positive");
        const auto batch_count = queries.size(0);
        const auto query_count = queries.size(1);
        const auto key_count = keys.size(1);
        const auto value_width = values.size(2);
        const auto competitors = support_topk_2d_batched_cuda(
            queries.contiguous(), keys.contiguous(), hull_indices, competitor_count);
        const auto offsets = torch::arange(batch_count, competitors.options())
                                 .reshape({batch_count, 1})
                                 .expand({batch_count, query_count})
                                 .reshape({-1}) * key_count;
        const auto winners = competitors.select(1, 0) + offsets;
        context->save_for_backward({queries, keys, values, competitors});
        context->saved_data["temperature"] = temperature;
        return values.reshape({batch_count * key_count, value_width})
            .index_select(0, winners)
            .reshape({batch_count, query_count, value_width});
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        const auto& queries = saved[0];
        const auto& keys = saved[1];
        const auto& values = saved[2];
        const auto& competitors = saved[3];
        const auto temperature = context->saved_data["temperature"].toDouble();
        const auto batch_count = queries.size(0);
        const auto query_count = queries.size(1);
        const auto key_count = keys.size(1);
        const auto value_width = values.size(2);
        const auto competitor_count = competitors.size(1);
        const auto offsets = torch::arange(batch_count, competitors.options())
                                 .reshape({batch_count, 1, 1})
                                 .expand({batch_count, query_count, competitor_count}) * key_count;
        const auto global_indices =
            competitors.reshape({batch_count, query_count, competitor_count}) + offsets;
        const auto selected_keys = keys.reshape({batch_count * key_count, 2})
                                       .index_select(0, global_indices.reshape({-1}))
                                       .reshape({batch_count * query_count, competitor_count, 2});
        const auto selected_values = values.reshape({batch_count * key_count, value_width})
                                         .index_select(0, global_indices.reshape({-1}))
                                         .reshape({batch_count * query_count,
                                                   competitor_count,
                                                   value_width});
        const auto flat_queries = queries.reshape({batch_count * query_count, 2});
        const auto probabilities = torch::softmax(
            (selected_keys * flat_queries.unsqueeze(1)).sum(-1) / temperature, -1);
        const auto upstream = gradients[0].reshape({batch_count * query_count, value_width});
        const auto utilities = (selected_values * upstream.unsqueeze(1)).sum(-1);
        const auto logit_gradient = probabilities *
            (utilities - (probabilities * utilities).sum(-1, true)) / temperature;
        const auto query_gradient =
            (logit_gradient.unsqueeze(-1) * selected_keys).sum(1).reshape_as(queries);
        auto key_gradient = torch::zeros_like(keys).reshape({batch_count * key_count, 2});
        key_gradient.index_add_(
            0, global_indices.reshape({-1}),
            (logit_gradient.unsqueeze(-1) * flat_queries.unsqueeze(1)).reshape({-1, 2}));
        auto value_gradient = torch::zeros_like(values).reshape(
            {batch_count * key_count, value_width});
        value_gradient.index_add_(
            0, global_indices.reshape({-1}),
            (probabilities.unsqueeze(-1) * upstream.unsqueeze(1)).reshape({-1, value_width}));
        return {query_gradient, key_gradient.reshape_as(keys), value_gradient.reshape_as(values),
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

}  // namespace

torch::Tensor hull_attention_2d_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature) {
    return HullAttentionSte::apply(
        queries, keys, values, hull_indices, competitor_count, temperature);
}

torch::Tensor hull_attention_2d_batched_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature) {
    return BatchedHullAttentionSte::apply(
        queries, keys, values, hull_indices, competitor_count, temperature);
}

torch::Tensor hull_attention_2d_dynamic_batched_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature) {
    return DynamicBatchedHullAttentionSte::apply(
        queries, keys, values, hull_indices, competitor_count, temperature);
}

}  // namespace cmz
