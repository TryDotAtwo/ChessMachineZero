# Knight geometry gate — 2026-08-13

TDD RED failed because `cmz_vm2/knight_geometry.h` did not exist.

GREEN builds a frozen token/matrix circuit and exhaustively checks all 4096
source-target pairs. Runtime evaluation is only matrix multiplication, hardmax,
one-hot and matrix multiplication. This gate proves geometry only; piece color,
target occupancy, check legality and board writes remain pending.
