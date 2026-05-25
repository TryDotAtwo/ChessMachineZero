"""Dashboard state over parametric Percepta rule weights."""

from __future__ import annotations

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.model.percepta_parametric_selfplay import PerceptaParametricMoveEvent, PerceptaParametricSelfPlaySession
from chess_machine_zero.rng import DEFAULT_SEED


class DashboardMoveError(ValueError):
    """Dashboard-level move rejection with a stable API error code."""

    def __init__(self, message: str, code: str = "illegal_move") -> None:
        super().__init__(message)
        self.code = code


DashboardMoveEvent = PerceptaParametricMoveEvent


class CMZDashboardSession:
    """Mutable dashboard session with parametric rule weights over arbitrary boards."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        start_fen: str = STARTING_FEN,
        max_plies: int = 64,
        temperature: float = 0.0,
    ) -> None:
        if temperature != 0.0:
            raise ValueError("pure decoder self-play uses deterministic temperature 0.0")
        self.seed = int(seed)
        self.start_fen = start_fen
        self.max_plies = int(max_plies)
        self.temperature = float(temperature)
        self._session = PerceptaParametricSelfPlaySession.create(
            start_fen=start_fen,
            seed=self.seed,
            max_plies=self.max_plies,
        )

    def reset(
        self,
        start_fen: str = STARTING_FEN,
        seed: int | None = None,
        max_plies: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        if temperature is not None and temperature != 0.0:
            raise ValueError("pure decoder self-play uses deterministic temperature 0.0")
        if seed is not None:
            self.seed = int(seed)
        if max_plies is not None:
            self.max_plies = int(max_plies)
        self.start_fen = start_fen
        self._session = PerceptaParametricSelfPlaySession.create(
            start_fen=self.start_fen,
            seed=self.seed,
            max_plies=self.max_plies,
        )
        return self.snapshot()

    def legal_moves(self) -> tuple[str, ...]:
        return self._session.legal_moves()

    def step_transformer(self) -> DashboardMoveEvent:
        try:
            return self._session.step()
        except ValueError as error:
            raise DashboardMoveError(str(error), code="terminal_position") from error

    def play_human_move(self, move_uci: str, auto_reply: bool = True) -> tuple[DashboardMoveEvent, ...]:
        try:
            human = self._session.play_human_move(move_uci)
            events = [human]
            if auto_reply and not self._session.terminal_status.is_terminal:
                events.append(self._session.step())
            return tuple(events)
        except ValueError as error:
            raise DashboardMoveError(str(error), code="illegal_move") from error

    def snapshot(self) -> dict:
        return self._session.snapshot()
