#include "cmz/hullkv.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cmz {
namespace {

constexpr int kMaximumCompetitors = 32;

__global__ void support_topk_kernel(
    const float* queries,
    const float* keys,
    const std::int64_t* hull,
    std::int64_t query_count,
    std::int64_t key_count,
    std::int64_t hull_count,
    std::int64_t k,
    std::int64_t* output) {
    const auto query = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count) {
        return;
    }
    for (std::int64_t slot = 0; slot < k; ++slot) output[query * k + slot] = 0;

    float top_scores[kMaximumCompetitors];
    std::int64_t top_indices[kMaximumCompetitors];
#pragma unroll
    for (int slot = 0; slot < kMaximumCompetitors; ++slot) {
        top_scores[slot] = -3.402823466e+38F;
        top_indices[slot] = 0x7fffffffffffffffLL;
    }

    const float qx = queries[query * 2];
    const float qy = queries[query * 2 + 1];
    for (std::int64_t position = 0; position < hull_count; ++position) {
        const auto key_index = hull[position];
        if (key_index < 0 || key_index >= key_count) {
            CUDA_KERNEL_ASSERT(false && "attention candidate out of range");
            return;
        }
        const float score = qx * keys[key_index * 2] + qy * keys[key_index * 2 + 1];
        if (!isfinite(score)) {
            CUDA_KERNEL_ASSERT(false && "attention score must be finite");
            return;
        }
        std::int64_t insertion = k;
        for (std::int64_t slot = 0; slot < k; ++slot) {
            if (score > top_scores[slot] ||
                (score == top_scores[slot] && key_index < top_indices[slot])) {
                insertion = slot;
                break;
            }
        }
        if (insertion < k) {
            for (std::int64_t slot = k - 1; slot > insertion; --slot) {
                top_scores[slot] = top_scores[slot - 1];
                top_indices[slot] = top_indices[slot - 1];
            }
            top_scores[insertion] = score;
            top_indices[insertion] = key_index;
        }
    }
    for (std::int64_t slot = 0; slot < k; ++slot) {
        output[query * k + slot] = top_indices[slot];
    }
}

__global__ void support_topk_batched_kernel(
    const float* queries,
    const float* keys,
    const std::int64_t* hull,
    std::int64_t batch_count,
    std::int64_t query_count,
    std::int64_t key_count,
    std::int64_t hull_count,
    std::int64_t k,
    std::int64_t* output) {
    const auto flat_query = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat_query >= batch_count * query_count) {
        return;
    }
    for (std::int64_t slot = 0; slot < k; ++slot) output[flat_query * k + slot] = 0;
    float top_scores[kMaximumCompetitors];
    std::int64_t top_indices[kMaximumCompetitors];
#pragma unroll
    for (int slot = 0; slot < kMaximumCompetitors; ++slot) {
        top_scores[slot] = -3.402823466e+38F;
        top_indices[slot] = 0x7fffffffffffffffLL;
    }
    const auto batch = flat_query / query_count;
    const float qx = queries[flat_query * 2];
    const float qy = queries[flat_query * 2 + 1];
    for (std::int64_t position = 0; position < hull_count; ++position) {
        const auto key_index = hull[position];
        if (key_index < 0 || key_index >= key_count) {
            CUDA_KERNEL_ASSERT(false && "attention candidate out of range");
            return;
        }
        const auto key_offset = (batch * key_count + key_index) * 2;
        const float score = qx * keys[key_offset] + qy * keys[key_offset + 1];
        if (!isfinite(score)) {
            CUDA_KERNEL_ASSERT(false && "attention score must be finite");
            return;
        }
        std::int64_t insertion = k;
        for (std::int64_t slot = 0; slot < k; ++slot) {
            if (score > top_scores[slot] ||
                (score == top_scores[slot] && key_index < top_indices[slot])) {
                insertion = slot;
                break;
            }
        }
        if (insertion < k) {
            for (std::int64_t slot = k - 1; slot > insertion; --slot) {
                top_scores[slot] = top_scores[slot - 1];
                top_indices[slot] = top_indices[slot - 1];
            }
            top_scores[insertion] = score;
            top_indices[insertion] = key_index;
        }
    }
    for (std::int64_t slot = 0; slot < k; ++slot) {
        output[flat_query * k + slot] = top_indices[slot];
    }
}

}  // namespace

torch::Tensor support_topk_2d_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& hull_indices,
    std::int64_t k) {
    TORCH_CHECK(queries.is_cuda() && keys.is_cuda() && hull_indices.is_cuda());
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32 && keys.scalar_type() == torch::kFloat32);
    TORCH_CHECK(hull_indices.scalar_type() == torch::kInt64);
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 2);
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 2);
    TORCH_CHECK(hull_indices.dim() == 1 && hull_indices.numel() > 0);
    TORCH_CHECK(queries.device() == keys.device() && keys.device() == hull_indices.device());
    TORCH_CHECK(k > 0 && k <= hull_indices.numel() && k <= kMaximumCompetitors);
    TORCH_CHECK(keys.size(0) > 0 && k <= keys.size(0), "attention key capacity mismatch");
    TORCH_CHECK(queries.is_contiguous() && keys.is_contiguous() && hull_indices.is_contiguous());

    c10::cuda::CUDAGuard guard(queries.device());
    auto output = torch::empty(
        {queries.size(0), k}, queries.options().dtype(torch::kInt64));
    if (queries.size(0) == 0) return output;
    constexpr int threads = 128;
    TORCH_CHECK((queries.size(0) + threads - 1) / threads <= std::numeric_limits<int>::max(),
                "attention query capacity overflow");
    const auto blocks = static_cast<int>((queries.size(0) + threads - 1) / threads);
    support_topk_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        hull_indices.data_ptr<std::int64_t>(),
        queries.size(0),
        keys.size(0),
        hull_indices.numel(),
        k,
        output.data_ptr<std::int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor support_topk_2d_batched_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& hull_indices,
    std::int64_t k) {
    TORCH_CHECK(queries.is_cuda() && keys.is_cuda() && hull_indices.is_cuda());
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32 && keys.scalar_type() == torch::kFloat32);
    TORCH_CHECK(hull_indices.scalar_type() == torch::kInt64);
    TORCH_CHECK(queries.dim() == 3 && queries.size(2) == 2);
    TORCH_CHECK(keys.dim() == 3 && keys.size(0) == queries.size(0) && keys.size(2) == 2);
    TORCH_CHECK(hull_indices.dim() == 1 && hull_indices.numel() > 0);
    TORCH_CHECK(queries.device() == keys.device() && keys.device() == hull_indices.device());
    TORCH_CHECK(k > 0 && k <= hull_indices.numel() && k <= kMaximumCompetitors);
    TORCH_CHECK(keys.size(1) > 0 && k <= keys.size(1), "attention key capacity mismatch");
    TORCH_CHECK(queries.is_contiguous() && keys.is_contiguous() && hull_indices.is_contiguous());

    c10::cuda::CUDAGuard guard(queries.device());
    auto output = torch::empty(
        {queries.size(0) * queries.size(1), k}, queries.options().dtype(torch::kInt64));
    constexpr int threads = 128;
    const auto flat_queries = queries.size(0) * queries.size(1);
    if (flat_queries == 0) return output;
    TORCH_CHECK((flat_queries + threads - 1) / threads <= std::numeric_limits<int>::max(),
                "attention query capacity overflow");
    const auto blocks = static_cast<int>((flat_queries + threads - 1) / threads);
    support_topk_batched_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        queries.data_ptr<float>(), keys.data_ptr<float>(), hull_indices.data_ptr<std::int64_t>(),
        queries.size(0), queries.size(1), keys.size(1), hull_indices.numel(), k,
        output.data_ptr<std::int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace cmz
