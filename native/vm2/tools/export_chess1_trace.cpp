#include <iostream>
#include <stdexcept>

#include "cmz_vm2/chess1_compiler.h"
#include "cmz_vm2/trace_exporter.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: cmz_vm2_export_chess1_trace OUTPUT COMMIT\n";
        return 2;
    }
    cmz::vm2::Chess1Board board{};
    board.squares[12] = cmz::vm2::Chess1Piece::WhitePawn;
    const auto image = cmz::vm2::compile_chess1(board, 12);
    cmz::vm2::write_trace_json(argv[1], image, image.tokens, argv[2]);
    return 0;
}
