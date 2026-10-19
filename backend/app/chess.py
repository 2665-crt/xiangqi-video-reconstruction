from .schemas import Move, Piece, Position, Square


def _board(pieces):
    return {(p.square.file, p.square.rank): p for p in pieces}


def _pseudo(piece, target, board):
    x, y = piece.square.file, piece.square.rank
    tx, ty = target
    dx, dy = tx - x, ty - y
    if (dx, dy) == (0, 0):
        return False
    occupant = board.get(target)
    if occupant and occupant.color == piece.color:
        return False
    kind, red = piece.kind, piece.color == "red"
    if kind in ("rook", "cannon"):
        if dx and dy:
            return False
        distance = abs(dx or dy)
        sx, sy = (0 if not dx else (1 if dx > 0 else -1)), (0 if not dy else (1 if dy > 0 else -1))
        blockers = sum((x + sx * i, y + sy * i) in board for i in range(1, distance))
        return blockers == (1 if kind == "cannon" and occupant else 0)
    if kind == "horse":
        if (abs(dx), abs(dy)) not in ((1, 2), (2, 1)):
            return False
        leg = (x + dx // 2, y) if abs(dx) == 2 else (x, y + dy // 2)
        return leg not in board
    if kind == "elephant":
        return (abs(dx), abs(dy)) == (2, 2) and (ty >= 5 if red else ty <= 4) and (x + dx // 2, y + dy // 2) not in board
    if kind == "pawn":
        return (dx == 0 and dy == (-1 if red else 1)) or (dy == 0 and abs(dx) == 1 and (y <= 4 if red else y >= 5))
    palace = 3 <= tx <= 5 and (7 <= ty <= 9 if red else 0 <= ty <= 2)
    if kind == "advisor":
        return palace and (abs(dx), abs(dy)) == (1, 1)
    if kind == "king":
        if occupant and occupant.kind == "king" and dx == 0:
            return not any((x, i) in board for i in range(min(y, ty) + 1, max(y, ty)))
        return palace and abs(dx) + abs(dy) == 1
    return False


def _safe(pieces, color):
    king = next((p for p in pieces if p.kind == "king" and p.color == color), None)
    if king is None:
        return False
    board = _board(pieces)
    target = (king.square.file, king.square.rank)
    return not any(p.color != color and _pseudo(p, target, board) for p in pieces)


def _changed(pieces, piece, target):
    return [p.model_copy(update={"square": Square(file=target[0], rank=target[1])}) if p.id == piece.id else p
            for p in pieces if p.id == piece.id or (p.square.file, p.square.rank) != target]


def apply_move(position: Position, move: Move) -> Position:
    piece = next((p for p in position.pieces if p.id == move.piece_id), None)
    if not piece or piece.color != position.side_to_move or move.color != piece.color or piece.square != move.source:
        raise ValueError("起点棋子、走棋方或轮次不一致")
    board, target = _board(position.pieces), (move.target.file, move.target.rank)
    occupant = board.get(target)
    if occupant and occupant.kind == "king":
        raise ValueError("不将捕获将帅记为正常走棋")
    if move.captured_piece_id and (not occupant or occupant.id != move.captured_piece_id):
        raise ValueError("吃子对象不一致")
    if not _pseudo(piece, target, board):
        dx, dy = target[0] - piece.square.file, target[1] - piece.square.rank
        if piece.kind == 'horse' and (abs(dx), abs(dy)) in ((1, 2), (2, 1)):
            leg = (piece.square.file + dx // 2, piece.square.rank) if abs(dx) == 2 else (piece.square.file, piece.square.rank + dy // 2)
            if leg in board:
                raise ValueError('蹩马腿，走法不符合象棋规则')
        raise ValueError("走法不符合象棋规则")
    changed = _changed(position.pieces, piece, target)
    if not _safe(changed, piece.color):
        raise ValueError("行棋后己方将帅受攻击或将帅照面")
    return Position(pieces=changed, side_to_move="black" if piece.color == "red" else "red")


def legal_moves(position: Position) -> list[Move]:
    actions, board = [], _board(position.pieces)
    for piece in position.pieces:
        if piece.color != position.side_to_move:
            continue
        for x in range(9):
            for y in range(10):
                occupant = board.get((x, y))
                if occupant and occupant.kind == "king":
                    continue
                if _pseudo(piece, (x, y), board) and _safe(_changed(position.pieces, piece, (x, y)), piece.color):
                    actions.append(Move(index=1, color=piece.color, piece_id=piece.id, source=piece.square,
                                        target=Square(file=x, rank=y), captured_piece_id=occupant.id if occupant else None))
    return actions


def replay(initial: Position, moves: list[Move]) -> list[Position]:
    states = [initial.model_copy(deep=True)]
    for move in moves:
        states.append(apply_move(states[-1], move))
    return states


def board_signature(position: Position):
    return tuple(sorted((p.color, p.kind, p.square.file, p.square.rank) for p in position.pieces))
