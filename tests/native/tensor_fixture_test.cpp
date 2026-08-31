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
    torch::NoGradGuard guard;
    cmz::FrozenVm vm(std::move(artifact), torch::Device("cuda:0"));
    for (std::uint32_t index = 0; index < cases; ++index) {
        const auto length = read_u32(stream);
        TORCH_CHECK(length > 0 && length <= 256, "tensor fixture label capacity");
        std::string label(length, '\0');
        read_exact(stream, label.data(), length);
        auto input = torch::empty({1, 2048, 128}, torch::kFloat32);
        auto expected = torch::empty({1, rows, columns}, torch::kFloat32);
        read_exact(stream, input.data_ptr<float>(), input.numel() * sizeof(float));
        read_exact(stream, expected.data_ptr<float>(), expected.numel() * sizeof(float));
        const auto actual = vm.execute_graph(input.to(torch::kCUDA)).cpu();
        TORCH_CHECK(actual.sizes() == expected.sizes(), "tensor output shape mismatch: ", label);
        TORCH_CHECK(torch::equal(actual, expected), "exact tensor mismatch: ", label);
        std::cout << "PASS exact tensor " << label << '\n';
    }
    TORCH_CHECK(stream.peek() == std::char_traits<char>::eof(), "trailing tensor fixture bytes");
    std::cout << "PASS " << cases << " full tensors; FP32, TF32 disabled, exact equality\n";
}
