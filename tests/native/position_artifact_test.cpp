#include "cmz/artifact.h"
#include "cmz/vm_module.h"
#include <ATen/Context.h>
#include <torch/torch.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
void read_exact(std::istream& stream, void* destination, std::size_t size) {
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
    TORCH_CHECK(stream.good(), "truncated native board fixture");
}
std::uint32_t read_u32(std::istream& stream) {
    unsigned char bytes[4];
    read_exact(stream, bytes, sizeof(bytes));
    return bytes[0] | (std::uint32_t(bytes[1]) << 8) | (std::uint32_t(bytes[2]) << 16) |
           (std::uint32_t(bytes[3]) << 24);
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    TORCH_CHECK(argc == 3, "position artifact and oracle fixture paths are required");
    std::ifstream artifact_stream(std::filesystem::path(argv[1]), std::ios::binary);
    TORCH_CHECK(artifact_stream.good(), "position artifact cannot be opened");
    const std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(artifact_stream), std::istreambuf_iterator<char>()};
    auto artifact = cmz::Artifact::from_bytes(bytes.data(), bytes.size());
    std::ifstream fixtures(std::filesystem::path(argv[2]), std::ios::binary);
    TORCH_CHECK(fixtures.good(), "oracle fixture cannot be opened");
    char magic[8];
    read_exact(fixtures, magic, sizeof(magic));
    TORCH_CHECK(std::string(magic, 8) == "CMZPOS01", "invalid fixture version");
    const auto count = read_u32(fixtures);
    TORCH_CHECK(count > 0 && count < 1000, "invalid fixture count");
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    torch::NoGradGuard no_grad;
    cmz::FrozenVm vm(std::move(artifact), torch::Device("cuda:0"));
    const auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    bool saw_400 = false;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto length = read_u32(fixtures);
        TORCH_CHECK(length > 0 && length <= 256, "invalid fixture label");
        std::string name(length, '\0');
        read_exact(fixtures, name.data(), length);
        auto source = torch::empty({1, 2048, 128}, cpu);
        auto expected = torch::empty({1, 64, 128}, cpu);
        read_exact(fixtures, source.data_ptr<float>(), source.numel() * sizeof(float));
        read_exact(fixtures, expected.data_ptr<float>(), expected.numel() * sizeof(float));
        const auto actual = vm.execute_graph(source.to(torch::kCUDA));
        TORCH_CHECK(actual.sizes() == expected.sizes(), "board shape mismatch: ", name);
        TORCH_CHECK(torch::equal(actual.cpu(), expected), "exact full-board mismatch: ", name);
        saw_400 |= name == "seed1-ply400";
        std::cout << "PASS exact FP32 board " << name << '\n';
    }
    TORCH_CHECK(saw_400, "400-ply counterexample fixture must be present");
    TORCH_CHECK(fixtures.peek() == std::char_traits<char>::eof(), "trailing fixture bytes");
    std::cout << "PASS " << count << " full native boards; FP32, TF32 disabled, exact equality\n";
    return 0;
}
