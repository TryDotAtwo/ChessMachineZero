# Public correction of the previous architecture claim

## Русский

Предыдущая версия ChessMachineZero была некорректно заявлена как реализация, в
которой шахматные правила полностью исполняются frozen self-attention.

Это утверждение было завышенным по следующим причинам:

1. Production C++/CUDA-код содержал шахматно-специфические `if`, циклы,
   сравнения фигур и координат, включая процедурную обработку рокировки, атак и
   переходов доски.
2. Названия функций вида `*_attention_*` и поля метаданных
   `full_frozen_attention_only=true` описывали намерение архитектуры, но сами по
   себе не доказывали механизм вычисления.
3. Часть пути действительно использовала `torch::matmul`, CUTLASS и QK-подобный
   выбор, однако host C++ также формировал списки ходов, управлял переходами и
   выполнял предметную логику. Поэтому нельзя было честно утверждать, что вся
   шахматная программа вычисляется только механизмом self-attention.
4. Тесты подтверждали совпадение результатов и отдельные границы исходников, но
   не доказывали, что каждое семантическое изменение состояния зависело только
   от `QKᵀ → hardmax → AV`.

`python-chess` был ограничен тестовым oracle и не являлся основной причиной
ошибки. Ошибка состояла именно в завышенной интерпретации собственного
C++/CUDA-runtime.

Эта orphan-ветка начинает реализацию заново. В production runtime отсутствуют
ветвления по opcode и шахматная логика. Программа, регистры, константы, ALU,
program counter и HALT представлены токенами. Переход VM выполняется шестью
фиксированными self-attention-этапами через матричные проекции, `QKᵀ`,
детерминированный hardmax и `AV`.

Это ранняя минимальная VM, а не готовый шахматный компьютер и не заявление о
воспроизведении неопубликованных технических деталей Percepta.

## English

The previous ChessMachineZero implementation was incorrectly presented as if
all chess-rule semantics were executed exclusively by frozen self-attention.

That claim was overstated because production C++/CUDA still contained
chess-specific branches, loops, piece/coordinate comparisons, castling logic,
attack logic, board transitions, and host orchestration that materially
participated in execution. Attention-style names and metadata such as
`full_frozen_attention_only=true` described an architectural intention; they did
not prove the actual semantic boundary. Some paths genuinely used LibTorch
matrix multiplication, CUTLASS, and QK-style selection, but the complete chess
program was not executed solely as `QKᵀ → hardmax → AV` over tokens.

The old tests established result equivalence and selected source boundaries,
not full semantic dependence on self-attention. `python-chess` remained a test
oracle and was not the main issue; the issue was the overstated interpretation
of the native C++/CUDA runtime.

This orphan branch starts over with a minimal token-native VM. Production
runtime contains no opcode-specific or chess-specific execution branches.
Program instructions, registers, constants, ALU relations, PC, and HALT are
tokens, and VM transitions use six fixed classical self-attention stages.

This is an early minimal VM, not a complete chess computer and not a claim to
reproduce Percepta's unpublished implementation details.
