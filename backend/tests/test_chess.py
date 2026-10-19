import pytest


def position(pieces, turn="red"):
    from app.schemas import Position
    return Position(pieces=[{"id": key, "color": color, "kind": kind,
                             "square": {"file": x, "rank": y}}
                            for key, color, kind, x, y in pieces], side_to_move=turn)


def kings():
    return [("rk", "red", "king", 4, 9), ("bk", "black", "king", 3, 0)]


def move(piece_id, x, y, tx, ty, color="red", captured=None):
    from app.schemas import Move
    return Move(index=1, color=color, piece_id=piece_id,
                **{"from": {"file": x, "rank": y}, "to": {"file": tx, "rank": ty}},
                captured_piece_id=captured)


def test_replay_capture_and_restore():
    from app.chess import replay
    p = position(kings() + [("r", "red", "rook", 0, 9), ("p", "black", "pawn", 0, 5)])
    states = replay(p, [move("r", 0, 9, 0, 5, captured="p")])
    assert len(states) == 2
    assert len(states[0].pieces) == len(p.pieces) == 4
    assert {i.id for i in states[1].pieces} == {"rk", "bk", "r"}
    assert states[1].side_to_move == "black"
    assert next(i for i in p.pieces if i.id == "r").square.rank == 9


def test_black_to_move_midgame():
    from app.chess import apply_move
    p = position(kings() + [("bp", "black", "pawn", 7, 5)], "black")
    result = apply_move(p, move("bp", 7, 5, 6, 5, "black"))
    assert next(i for i in result.pieces if i.id == "bp").square.file == 6
    assert result.side_to_move == "red"


@pytest.mark.parametrize("extra,action", [
    ([("h", "red", "horse", 1, 9), ("b", "red", "pawn", 1, 8)], ("h", 1, 9, 2, 7)),
    ([("c", "red", "cannon", 1, 9), ("t", "black", "pawn", 1, 3)], ("c", 1, 9, 1, 3)),
    ([("e", "red", "elephant", 2, 5)], ("e", 2, 5, 4, 3)),
    ([("p", "red", "pawn", 0, 6)], ("p", 0, 6, 1, 6)),
    ([("p", "red", "pawn", 0, 4)], ("p", 0, 4, 0, 5)),
])
def test_illegal_horse_cannon_elephant_and_pawn(extra, action):
    from app.chess import apply_move
    with pytest.raises(ValueError):
        apply_move(position(kings() + extra), move(*action))


def test_move_cannot_expose_facing_kings():
    from app.chess import apply_move
    p = position([("rk", "red", "king", 4, 9), ("bk", "black", "king", 4, 0),
                  ("r", "red", "rook", 4, 5)])
    with pytest.raises(ValueError):
        apply_move(p, move("r", 4, 5, 5, 5))


def test_cannon_capture_with_one_screen():
    from app.chess import apply_move
    p = position(kings() + [("c", "red", "cannon", 1, 7), ("s", "red", "pawn", 1, 5),
                            ("t", "black", "rook", 1, 1)])
    result = apply_move(p, move("c", 1, 7, 1, 1, captured="t"))
    assert len(result.pieces) == 4


def test_invalid_position_duplicate_square():
    with pytest.raises(ValueError):
        position(kings() + [("p", "red", "pawn", 4, 9)])


def test_legal_moves_keep_piece_identity():
    from app.chess import legal_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    actions = legal_moves(p)
    assert any(m.piece_id == "r" and m.target.file == 0 and m.target.rank == 5 for m in actions)
    assert all(m.color == "red" for m in actions)


def test_demonstration_endgame_keeps_four_black_cannons():
    p = position(kings() + [(f"bc{i}", "black", "cannon", i, 4) for i in range(4)])
    assert sum(piece.kind == "cannon" and piece.color == "black" for piece in p.pieces) == 4
