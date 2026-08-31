#include "cmz/hullkv.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {
const auto gpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
const auto ints = gpu.dtype(torch::kInt64);

// Independent scalar-double derivative: sort literal scores on CPU, then apply
// dS_i=p_i*(g.V_i-sum_j(p_j*g.V_j))/T. No VM/autograd/kernel reference helpers.
void compare_derivatives(int mode, bool isolate_first_batch) {
    // Establish ordering before uploading any inputs, not merely before dispatch.
    const auto stream = at::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    constexpr int B = 2, Q = 3, K = 5, D = 2, top = 3;
    constexpr double temperature = 0.7;
    const std::vector<float> qs = {1,0, 1,1, 0,0, 0,1, -1,0.5F, 1,-1};
    const std::vector<float> ks = {0,0, 1,0, 0,1, -1,0, 1,0,
                                 0,0, -1,0, 0,1, 1,0, -1,0};
    const std::vector<float> vs = {0,0, 2,1, -3,4, 5,-2, 7,6,
                                 11,-3, 1,-5, 6,2, -7,3, 9,4};
    std::vector<float> gs = {2,-1, 0.5F,3, -2,1, -1,4, 2,-0.5F, 3,2};
    if (isolate_first_batch) std::fill(gs.begin() + Q*D, gs.end(), 0.0F);
    const int batches = mode == 0 ? 1 : B;
    auto q_cpu = torch::from_blob(const_cast<float*>(qs.data()), {batches,Q,2}, torch::kFloat32).clone();
    auto k_cpu = torch::from_blob(const_cast<float*>(ks.data()), {mode == 2 ? B : 1,K,2}, torch::kFloat32).clone();
    auto v_cpu = torch::from_blob(const_cast<float*>(vs.data()), {batches,K,D}, torch::kFloat32).clone();
    auto g_cpu = torch::from_blob(gs.data(), {batches,Q,D}, torch::kFloat32).clone();
    if (mode == 0) { q_cpu = q_cpu.squeeze(0); v_cpu = v_cpu.squeeze(0); g_cpu = g_cpu.squeeze(0); }
    if (mode != 2) k_cpu = k_cpu.squeeze(0);
    auto queries = q_cpu.to(torch::kCUDA).set_requires_grad(true);
    auto keys = k_cpu.to(torch::kCUDA).set_requires_grad(true);
    auto values = v_cpu.to(torch::kCUDA).set_requires_grad(true);
    // Shuffled candidates make index-order tie handling observable.
    const auto candidates = torch::tensor({4,3,2,1,0}, ints);
    auto dq = torch::zeros_like(q_cpu, torch::kFloat64);
    auto dk = torch::zeros_like(k_cpu, torch::kFloat64);
    auto dv = torch::zeros_like(v_cpu, torch::kFloat64);
    auto expected = torch::zeros_like(g_cpu);
    for (int b = 0; b < batches; ++b) for (int row = 0; row < Q; ++row) {
        const int qi = (b * Q + row) * 2;
        const int kb = mode == 2 ? b : 0;
        std::array<int,K> order = {0,1,2,3,4};
        std::array<double,K> scores{};
        for (int i = 0; i < K; ++i)
            scores[i] = double(qs[qi])*ks[(kb*K+i)*2] + double(qs[qi+1])*ks[(kb*K+i)*2+1];
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            return scores[x] == scores[y] ? x < y : scores[x] > scores[y];
        });
        std::array<double,top> probability{}, utility{};
        double norm = 0, mean = 0;
        for (int slot = 0; slot < top; ++slot) {
            probability[slot] = std::exp((scores[order[slot]] - scores[order[0]]) / temperature);
            norm += probability[slot];
            for (int d = 0; d < D; ++d)
                utility[slot] += double(gs[(b*Q+row)*D+d]) * vs[(b*K+order[slot])*D+d];
        }
        for (int slot = 0; slot < top; ++slot) { probability[slot] /= norm; mean += probability[slot]*utility[slot]; }
        for (int d = 0; d < D; ++d)
            expected.data_ptr<float>()[(b*Q+row)*D+d] = vs[(b*K+order[0])*D+d];
        for (int slot = 0; slot < top; ++slot) {
            const int i = order[slot];
            const double ds = probability[slot]*(utility[slot]-mean)/temperature;
            for (int d = 0; d < 2; ++d) {
                dq.data_ptr<double>()[qi+d] += ds*ks[(kb*K+i)*2+d];
                dk.data_ptr<double>()[(kb*K+i)*2+d] += ds*qs[qi+d];
            }
            for (int d = 0; d < D; ++d)
                dv.data_ptr<double>()[(b*K+i)*D+d] += probability[slot]*gs[(b*Q+row)*D+d];
        }
    }
    // A non-default stream verifies that custom selection honors caller ordering.
    auto output = mode == 0 ? cmz::hull_attention_2d_ste(queries, keys, values, candidates, top, temperature)
        : mode == 1 ? cmz::hull_attention_2d_batched_ste(queries, keys, values, candidates, top, temperature)
                    : cmz::hull_attention_2d_dynamic_batched_ste(queries, keys, values, candidates, top, temperature);
    TORCH_CHECK(torch::equal(output.cpu(), expected), "exact winner mismatch mode=", mode);
    output.backward(g_cpu.to(torch::kCUDA));
    const std::array<torch::Tensor,3> got = {queries.grad().cpu().to(torch::kFloat64),
        keys.grad().cpu().to(torch::kFloat64), values.grad().cpu().to(torch::kFloat64)};
    const std::array<torch::Tensor,3> want = {dq,dk,dv};
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(torch::allclose(got[i], want[i], 2e-5, 2e-6), "Q/K/V derivative mismatch mode=", mode, " index=", i);
    }
    if (isolate_first_batch) {
        TORCH_CHECK(queries.grad()[1].count_nonzero().item<std::int64_t>() == 0);
        TORCH_CHECK(values.grad()[1].count_nonzero().item<std::int64_t>() == 0);
        if (mode == 2) TORCH_CHECK(keys.grad()[1].count_nonzero().item<std::int64_t>() == 0);
    }
    std::cout << "PASS exact winners + scalar-double QKV derivatives mode=" << mode
              << " isolated=" << isolate_first_batch << " rtol=2e-5 atol=2e-6\n";
}

int invalid_indices(const std::string& mode) {
    const bool batched = mode.find("batched") != std::string::npos;
    auto q = torch::ones(batched ? std::vector<std::int64_t>{1,2,2} : std::vector<std::int64_t>{2,2}, gpu);
    auto k = q.clone();
    const auto indices = torch::tensor({0, mode.find("negative") != std::string::npos ? -1 : 100000000}, ints);
    try {
        if (batched) cmz::support_topk_2d_batched_cuda(q,k,indices,1);
        else cmz::support_topk_2d_cuda(q,k,indices,1);
        torch::cuda::synchronize();
    } catch (const c10::Error& error) {
        TORCH_CHECK(std::string(error.what()).find("device-side assert") != std::string::npos,
                    "expected bounds assertion, not an invalid memory access: ", error.what());
        std::cout << "PASS isolated index bounds assertion " << mode << '\n';
        return 0;
    }
    std::cerr << "FAIL invalid index was accepted\n";
    return 1;
}

int validate_metadata() {
    int failures = 0;
    auto q = torch::ones({2,2}, gpu), k = q.clone(), v = q.clone();
    const auto idx = torch::tensor({0,1}, ints);
    const auto rejects = [&](const char* name, torch::Tensor value, double temperature) {
        try { cmz::hull_attention_2d_ste(q,k,value,idx,1,temperature); }
        catch (const c10::Error&) { std::cout << "PASS " << name << '\n'; return; }
        std::cerr << "FAIL " << name << " accepted\n"; ++failures;
    };
    rejects("nonfinite temperature", v, std::numeric_limits<double>::infinity());
    rejects("value rank", torch::ones({2}, gpu), 0.5);
    rejects("value precision", v.to(torch::kFloat64), 0.5);
    std::cout << "metadata failures=" << failures << '\n';
    return failures ? 1 : 0;
}
}

int main(int argc, char** argv) {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    if (argc == 2) {
        if (std::string(argv[1]) == "metadata") return validate_metadata();
        return invalid_indices(argv[1]);
    }
    for (int mode = 0; mode < 3; ++mode) compare_derivatives(mode, false);
    compare_derivatives(1, true);
    compare_derivatives(2, true);
    return 0;
}
