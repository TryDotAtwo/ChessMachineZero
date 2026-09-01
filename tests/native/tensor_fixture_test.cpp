#include "cmz/vm_module.h"
#include <ATen/Context.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
void read_exact(std::istream& stream, void* destination, std::size_t size) {
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
    TORCH_CHECK(stream.good(), "truncated tensor fixture");
}
std::uint32_t read_u32(std::istream& stream) {
    unsigned char bytes[4];
    read_exact(stream, bytes, 4);
    return bytes[0] | (std::uint32_t(bytes[1]) << 8) | (std::uint32_t(bytes[2]) << 16) |
           (std::uint32_t(bytes[3]) << 24);
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    TORCH_CHECK(argc == 3, "artifact and independent tensor fixture paths required");
    std::ifstream artifact_file(std::filesystem::path(argv[1]), std::ios::binary);
    TORCH_CHECK(artifact_file.good(), "cannot open artifact");
    const std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(artifact_file), std::istreambuf_iterator<char>()};
    auto artifact = cmz::Artifact::from_bytes(bytes.data(), bytes.size());
    std::ifstream stream(std::filesystem::path(argv[2]), std::ios::binary);
    TORCH_CHECK(stream.good(), "cannot open tensor fixtures");
    char magic[8];
    read_exact(stream, magic, 8);
    TORCH_CHECK(std::string(magic, 8) == "CMZARR01", "tensor fixture version mismatch");
    const auto cases = read_u32(stream), rows = read_u32(stream), columns = read_u32(stream);
    TORCH_CHECK(cases > 0 && cases <= 1000 && rows > 0 && columns > 0 &&
                std::uint64_t(rows) * columns <= 1048576, "tensor fixture capacity mismatch");
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    cmz::FrozenVm vm(std::move(artifact), torch::Device("cuda:0"));
    torch::Tensor gradient_input_fixture;
    torch::Tensor recurrent_context;
    std::string recurrent_sequence;
    std::uint32_t recurrent_ply = 0;
    {
        torch::NoGradGuard guard;
        for (std::uint32_t index = 0; index < cases; ++index) {
            const auto length = read_u32(stream);
            TORCH_CHECK(length > 0 && length <= 256, "tensor fixture label capacity");
            std::string label(length, '\0');
            read_exact(stream, label.data(), length);
            auto input = torch::empty({1, 2048, 128}, torch::kFloat32);
            auto expected = torch::empty({1, rows, columns}, torch::kFloat32);
            read_exact(stream, input.data_ptr<float>(), input.numel() * sizeof(float));
            read_exact(stream, expected.data_ptr<float>(), expected.numel() * sizeof(float));
            if (gradient_input_fixture.defined() == false) {
                gradient_input_fixture = input.clone();
            }

            auto execution_input = input.to(torch::kCUDA);
            const auto ply_marker = label.rfind("-ply");
            if (ply_marker != std::string::npos) {
                const auto sequence = label.substr(0, ply_marker);
                const auto ply = static_cast<std::uint32_t>(std::stoul(label.substr(ply_marker + 4)));
                if (ply == 1U) {
                    recurrent_sequence = sequence;
                    recurrent_ply = 1U;
                    recurrent_context = vm.initial_context(1);
                } else {
                    TORCH_CHECK(sequence == recurrent_sequence && ply == recurrent_ply + 1U,
                                "non-contiguous recurrent fixture sequence: ", label);
                    recurrent_ply = ply;
                }
                TORCH_CHECK(torch::equal(recurrent_context.cpu(), input.slice(1, 3, 2048)),
                            "fixture prior context differs from native feedback: ", label);
                execution_input = torch::cat(
                    {execution_input.slice(1, 0, 3), recurrent_context}, 1);
            }

            const auto actual_cuda = vm.forward(execution_input);
            const auto actual = actual_cuda.cpu();
            TORCH_CHECK(actual.sizes() == expected.sizes(), "tensor output shape mismatch: ", label);
            TORCH_CHECK(torch::equal(actual, expected), "exact tensor mismatch: ", label);
            if (ply_marker != std::string::npos) {
                recurrent_context = actual_cuda;
            }
            std::cout << "PASS exact recurrent tensor " << label << '\n';
        }
    }
    TORCH_CHECK(stream.peek() == std::char_traits<char>::eof(), "trailing tensor fixture bytes");

    auto gradient_input = gradient_input_fixture.to(torch::kCUDA).detach().clone();
    gradient_input.set_requires_grad(true);
    const auto gradient_output = vm.forward(gradient_input);
    const auto weights = torch::linspace(
        -1.0F, 1.0F, 768 * 128,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
        .reshape({1, 768, 128});
    (gradient_output.slice(1, 1200, 1968) * weights).sum().backward();
    const auto request_gradient = gradient_input.grad().slice(1, 0, 3);
    TORCH_CHECK(request_gradient.defined(), "requested-move gradient is undefined");
    TORCH_CHECK(torch::isfinite(request_gradient).all().item<bool>(),
                "requested-move gradient is non-finite");
    TORCH_CHECK(request_gradient.count_nonzero().item<std::int64_t>() > 0,
                "generated LEGAL_SET has zero gradient to the requested move");
    std::cout << "PASS generated LEGAL_SET backward to requested move; nonzero="
              << request_gradient.count_nonzero().item<std::int64_t>()
              << " abs_sum=" << request_gradient.abs().sum().item<float>() << '\n';
    std::cout << "PASS " << cases
              << " full recurrent tensors; native feedback, FP32, TF32 disabled, exact equality\n";
}
