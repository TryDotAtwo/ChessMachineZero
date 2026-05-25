from __future__ import annotations

import pytest

from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo, square_index, square_name


def test_square_codec_round_trip() -> None:
    assert square_index("a1") == 0
    assert square_index("h8") == 63
    assert square_name(0) == "a1"
    assert square_name(63) == "h8"


def test_move_packet_tuple_uci_and_id_round_trip() -> None:
    packet = MovePacket.from_uci("a7a8q", MoveFlag.CAPTURE)
    assert packet.promo is Promo.QUEEN
    assert packet.flags & MoveFlag.PROMOTION
    assert packet.flags & MoveFlag.CAPTURE
    assert packet.to_uci() == "a7a8q"
    assert MovePacket.from_tuple(packet.to_tuple()) == packet
    assert MovePacket.from_move_id(packet.move_id, packet.flags) == packet


def test_move_packet_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MovePacket(-1, 0)
    with pytest.raises(ValueError):
        MovePacket.from_uci("e2e9")
    with pytest.raises(ValueError):
        MovePacket.from_tuple([0, 1, 2])
