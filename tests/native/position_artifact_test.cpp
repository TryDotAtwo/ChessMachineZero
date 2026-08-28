#include "cmz/artifact.h"
#include "cmz/vm_module.h"

#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr std::int64_t kEmpty = 95;
constexpr std::int64_t kWhitePawn = 96;
constexpr std::int64_t kWhiteKnight = 97;
constexpr std::int64_t kWhiteBishop = 98;
constexpr std::int64_t kWhiteRook = 99;
constexpr std::int64_t kWhiteQueen = 100;
constexpr std::int64_t kWhiteKing = 101;
constexpr std::int64_t kBlackPawn = 102;
constexpr std::int64_t kBlackKnight = 103;
constexpr std::int64_t kBlackBishop = 104;
constexpr std::int64_t kBlackRook = 105;
constexpr std::int64_t kBlackQueen = 106;
constexpr std::int64_t kBlackKing = 107;

std::int64_t square(std::int64_t native) {
    return (native / 10 - 1) * 8 + native % 10 - 1;
}

void set_one_hot(torch::Tensor tensor, std::int64_t batch, std::int64_t row, std::int64_t channel) {
    tensor.index_put_({batch, row}, 0.0F);
    tensor.index_put_({batch, row, channel}, 1.0F);
}

void set_move(
    torch::Tensor input,
    std::int64_t batch,
    std::int64_t ply,
    std::int64_t source,
    std::int64_t destination,
    std::int64_t piece) {
    const auto row = 3 + ply * 3;
    set_one_hot(input, batch, row, source);
    set_one_hot(input, batch, row + 1, destination);
    set_one_hot(input, batch, row + 2, piece);
}

torch::Tensor initial_board(const torch::TensorOptions& options, std::int64_t batch) {
    auto board = torch::zeros({batch, 64, 128}, options.requires_grad(false));
    board.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), kEmpty}, 1.0F);
    constexpr std::int64_t white_back[] = {
        kWhiteRook, kWhiteKnight, kWhiteBishop, kWhiteQueen,
        kWhiteKing, kWhiteBishop, kWhiteKnight, kWhiteRook};
    constexpr std::int64_t black_back[] = {
        kBlackRook, kBlackKnight, kBlackBishop, kBlackQueen,
        kBlackKing, kBlackBishop, kBlackKnight, kBlackRook};
    for (std::int64_t file = 1; file <= 8; ++file) {
        set_one_hot(board, 0, square(file * 10 + 1), white_back[file - 1]);
        set_one_hot(board, 0, square(file * 10 + 2), kWhitePawn);
        set_one_hot(board, 0, square(file * 10 + 7), kBlackPawn);
        set_one_hot(board, 0, square(file * 10 + 8), black_back[file - 1]);
    }
    return board.index({0}).unsqueeze(0).expand({batch, 64, 128}).clone();
}

void set_piece(torch::Tensor board, std::int64_t batch, std::int64_t native, std::int64_t piece) {
    set_one_hot(board, batch, square(native), piece);
}

}  // namespace

int main(int argc, char** argv) {
    TORCH_CHECK(argc == 2, "position artifact path is required");
    std::ifstream stream(argv[1], std::ios::binary);
    TORCH_CHECK(stream.good(), "position artifact cannot be opened");
    const std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto artifact = cmz::Artifact::from_bytes(bytes.data(), bytes.size());

    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto input = torch::zeros({3, 2048, 128}, options);
    input.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, 1.0F);

    set_move(input, 0, 0, 52, 54, kWhitePawn);
    set_move(input, 0, 1, 47, 45, kBlackPawn);
    set_move(input, 0, 2, 54, 45, kWhitePawn);
    set_move(input, 1, 0, 51, 71, kWhiteKing);
    set_move(input, 2, 0, 52, 54, kWhitePawn);
    set_move(input, 2, 1, 17, 16, kBlackPawn);
    set_move(input, 2, 2, 54, 55, kWhitePawn);
    set_move(input, 2, 3, 47, 45, kBlackPawn);
    set_move(input, 2, 4, 55, 46, kWhitePawn);
    input.set_requires_grad(true);

    auto expected = initial_board(options, 3);
    set_piece(expected, 0, 52, kEmpty);
    set_piece(expected, 0, 54, kEmpty);
    set_piece(expected, 0, 47, kEmpty);
    set_piece(expected, 0, 45, kWhitePawn);
    set_piece(expected, 1, 51, kEmpty);
    set_piece(expected, 1, 71, kWhiteKing);
    set_piece(expected, 1, 81, kEmpty);
    set_piece(expected, 1, 61, kWhiteRook);
    set_piece(expected, 2, 52, kEmpty);
    set_piece(expected, 2, 54, kEmpty);
    set_piece(expected, 2, 55, kEmpty);
    set_piece(expected, 2, 17, kEmpty);
    set_piece(expected, 2, 16, kBlackPawn);
    set_piece(expected, 2, 47, kEmpty);
    set_piece(expected, 2, 45, kEmpty);
    set_piece(expected, 2, 46, kWhitePawn);

    cmz::FrozenVm vm(std::move(artifact), input.device());
    const auto board = vm.execute_graph(input);
    TORCH_CHECK(board.sizes() == torch::IntArrayRef({3, 64, 128}));
    TORCH_CHECK(torch::equal(board, expected));
    board.index({0, square(45), kWhitePawn}).backward();
    TORCH_CHECK(input.grad().defined());
    TORCH_CHECK(input.grad().index({0, 11, kWhitePawn}).item<float>() != 0.0F);
    return 0;
}
