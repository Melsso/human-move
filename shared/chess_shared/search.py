from __future__ import annotations

import math
import os
import chess

INF = 10_000_000
MATE = 100_000
MATE_BOUND = MATE - 1_000
MAX_PLY = 64
MAX_QPLY = 8
TT_MAX_ENTRIES = 400_000
DELTA_MARGIN = 200

EXACT, LOWER, UPPER = 0, 1, 2

_VALUE = (0, 100, 320, 330, 500, 900)
_PHASE = (0, 0, 1, 1, 2, 4)
_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
_FILE_BB = chess.BB_FILES
_RANK_BB = chess.BB_RANKS


def _flat(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(v for row in rows for v in row)


_PAWN = _flat(
    (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (50, 50, 50, 50, 50, 50, 50, 50),
        (10, 10, 20, 30, 30, 20, 10, 10),
        (5, 5, 10, 25, 25, 10, 5, 5),
        (0, 0, 0, 20, 20, 0, 0, 0),
        (5, -5, -10, 0, 0, -10, -5, 5),
        (5, 10, 10, -20, -20, 10, 10, 5),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )
)
_KNIGHT = _flat(
    (
        (-50, -40, -30, -30, -30, -30, -40, -50),
        (-40, -20, 0, 0, 0, 0, -20, -40),
        (-30, 0, 10, 15, 15, 10, 0, -30),
        (-30, 5, 15, 20, 20, 15, 5, -30),
        (-30, 0, 15, 20, 20, 15, 0, -30),
        (-30, 5, 10, 15, 15, 10, 5, -30),
        (-40, -20, 0, 5, 5, 0, -20, -40),
        (-50, -40, -30, -30, -30, -30, -40, -50),
    )
)
_BISHOP = _flat(
    (
        (-20, -10, -10, -10, -10, -10, -10, -20),
        (-10, 0, 0, 0, 0, 0, 0, -10),
        (-10, 0, 5, 10, 10, 5, 0, -10),
        (-10, 5, 5, 10, 10, 5, 5, -10),
        (-10, 0, 10, 10, 10, 10, 0, -10),
        (-10, 10, 10, 10, 10, 10, 10, -10),
        (-10, 5, 0, 0, 0, 0, 5, -10),
        (-20, -10, -10, -10, -10, -10, -10, -20),
    )
)
_ROOK = _flat(
    (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (5, 10, 10, 10, 10, 10, 10, 5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (-5, 0, 0, 0, 0, 0, 0, -5),
        (0, 0, 0, 5, 5, 0, 0, 0),
    )
)
_QUEEN = _flat(
    (
        (-20, -10, -10, -5, -5, -10, -10, -20),
        (-10, 0, 0, 0, 0, 0, 0, -10),
        (-10, 0, 5, 5, 5, 5, 0, -10),
        (-5, 0, 5, 5, 5, 5, 0, -5),
        (0, 0, 5, 5, 5, 5, 0, -5),
        (-10, 5, 5, 5, 5, 5, 0, -10),
        (-10, 0, 5, 0, 0, 0, 0, -10),
        (-20, -10, -10, -5, -5, -10, -10, -20),
    )
)
_KING_MG = _flat(
    (
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-20, -30, -30, -40, -40, -30, -30, -20),
        (-10, -20, -20, -20, -20, -20, -20, -10),
        (20, 20, 0, 0, 0, 0, 20, 20),
        (20, 30, 10, 0, 0, 10, 30, 20),
    )
)
_KING_EG = _flat(
    (
        (-50, -40, -30, -20, -20, -30, -40, -50),
        (-30, -20, -10, 0, 0, -10, -20, -30),
        (-30, -10, 20, 30, 30, 20, -10, -30),
        (-30, -10, 30, 40, 40, 30, -10, -30),
        (-30, -10, 30, 40, 40, 30, -10, -30),
        (-30, -10, 20, 30, 30, 20, -10, -30),
        (-30, -30, 0, 0, 0, 0, -30, -30),
        (-50, -30, -30, -30, -30, -30, -30, -50),
    )
)
_PST: tuple[tuple[int, ...], ...] = ((), _PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN)


def _popcount(bb: int) -> int:
    return bb.bit_count()


def _evaluate_basic(board: chess.Board) -> int:
    white = 0
    black = 0
    phase = 0
    for pt in _PIECE_TYPES:
        table = _PST[pt]
        value = _VALUE[pt]
        phase_weight = _PHASE[pt]
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)):
            white += value + table[sq ^ 56]
            phase += phase_weight
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)):
            black += value + table[sq]
            phase += phase_weight
    phase = min(phase, 24)

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        i = wk ^ 56
        white += (_KING_MG[i] * phase + _KING_EG[i] * (24 - phase)) // 24
    if bk is not None:
        black += (_KING_MG[bk] * phase + _KING_EG[bk] * (24 - phase)) // 24

    if _popcount(board.bishops & board.occupied_co[chess.WHITE]) >= 2:
        white += 30
    if _popcount(board.bishops & board.occupied_co[chess.BLACK]) >= 2:
        black += 30

    score = white - black
    if board.turn == chess.BLACK:
        score = -score
    return score + 10


def _adjacent_files(f: int) -> int:
    m = 0
    if f > 0:
        m |= _FILE_BB[f - 1]
    if f < 7:
        m |= _FILE_BB[f + 1]
    return m


def _rank_span(lo: int, hi: int) -> int:
    """Union of ranks lo..hi inclusive, clipped to the board."""
    m = 0
    for r in range(max(lo, 0), min(hi, 7) + 1):
        m |= _RANK_BB[r]
    return m


_ADJ = [_adjacent_files(f) for f in range(8)]
_PASSED_W, _PASSED_B = [0] * 64, [0] * 64
_BEHIND_W, _BEHIND_B = [0] * 64, [0] * 64
_SHIELD_W, _SHIELD_B = [0] * 64, [0] * 64
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _span = _FILE_BB[_f] | _ADJ[_f]
    _PASSED_W[_sq] = _span & _rank_span(_r + 1, 7)
    _PASSED_B[_sq] = _span & _rank_span(0, _r - 1)
    _BEHIND_W[_sq] = _FILE_BB[_f] & _rank_span(0, _r - 1)
    _BEHIND_B[_sq] = _FILE_BB[_f] & _rank_span(_r + 1, 7)
    _SHIELD_W[_sq] = _span & _rank_span(_r + 1, _r + 2)
    _SHIELD_B[_sq] = _span & _rank_span(_r - 2, _r - 1)

_MOBILITY = (0, 0, 4, 3, 2, 1)
_PASSED_BONUS = (0, 5, 10, 20, 35, 60, 100, 0)


def _evaluate_full(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side-to-move's perspective."""
    occ_w = board.occupied_co[chess.WHITE]
    occ_b = board.occupied_co[chess.BLACK]
    free_w = chess.BB_ALL ^ occ_w
    free_b = chess.BB_ALL ^ occ_b
    wp = board.pawns & occ_w
    bp = board.pawns & occ_b
    white = 0
    black = 0
    phase = 0

    for pt in _PIECE_TYPES:
        table = _PST[pt]
        value = _VALUE[pt]
        phase_weight = _PHASE[pt]
        mob = _MOBILITY[pt]
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)):
            white += value + table[sq ^ 56]
            phase += phase_weight
            if mob:
                white += mob * _popcount(board.attacks_mask(sq) & free_w)
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)):
            black += value + table[sq]
            phase += phase_weight
            if mob:
                black += mob * _popcount(board.attacks_mask(sq) & free_b)
    if phase > 24:
        phase = 24

    for sq in chess.scan_forward(board.rooks & occ_w):
        f = _FILE_BB[chess.square_file(sq)]
        if not (wp & f):
            white += 10 if (bp & f) else 20
    for sq in chess.scan_forward(board.rooks & occ_b):
        f = _FILE_BB[chess.square_file(sq)]
        if not (bp & f):
            black += 10 if (wp & f) else 20

    for sq in chess.scan_forward(wp):
        if not (bp & _PASSED_W[sq]):
            white += _PASSED_BONUS[chess.square_rank(sq)]
        if not (wp & _ADJ[chess.square_file(sq)]):
            white -= 12
        if wp & _BEHIND_W[sq]:
            white -= 10
    for sq in chess.scan_forward(bp):
        if not (wp & _PASSED_B[sq]):
            black += _PASSED_BONUS[7 - chess.square_rank(sq)]
        if not (bp & _ADJ[chess.square_file(sq)]):
            black -= 12
        if bp & _BEHIND_B[sq]:
            black -= 10

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        i = wk ^ 56
        white += (_KING_MG[i] * phase + _KING_EG[i] * (24 - phase)) // 24
        white += _popcount(wp & _SHIELD_W[wk]) * 10 * phase // 24
    if bk is not None:
        black += (_KING_MG[bk] * phase + _KING_EG[bk] * (24 - phase)) // 24
        black += _popcount(bp & _SHIELD_B[bk]) * 10 * phase // 24

    if _popcount(board.bishops & occ_w) >= 2:
        white += 30
    if _popcount(board.bishops & occ_b) >= 2:
        black += 30

    score = white - black
    if board.turn == chess.BLACK:
        score = -score
    return score + 10


evaluate = (
    _evaluate_basic if os.environ.get("CHESS_EVAL") == "basic" else _evaluate_full
)


def _key(board: chess.Board) -> tuple[object, ...]:
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.castling_rights,
        board.ep_square,
    )


class SearchAbort(Exception):
    pass


_TTEntry = tuple[int, int, int, "chess.Move | None"]


class Searcher:
    def __init__(self, max_nodes: int = 0) -> None:
        self.max_nodes = max_nodes
        self.tt: dict[tuple[object, ...], _TTEntry] = {}
        self.killers: list[list[chess.Move | None]] = []
        self.history: dict[tuple[int, int], int] = {}
        self.seen: dict[tuple[object, ...], int] = {}
        self.nodes = 0

    def prepare(self, board: chess.Board) -> None:
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history = {}
        self.seen = {}
        self.nodes = 0

        self.seen[_key(board)] = 1
        tmp = board.copy(stack=True)
        for _ in range(min(len(tmp.move_stack), tmp.halfmove_clock)):
            tmp.pop()
            k = _key(tmp)
            self.seen[k] = self.seen.get(k, 0) + 1

    def _tick(self) -> None:
        self.nodes += 1
        if self.max_nodes and self.nodes > self.max_nodes:
            raise SearchAbort

    def _move_score(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return 1_000_000
        victim = board.piece_type_at(move.to_square)
        if (
            victim is None
            and move.to_square == board.ep_square
            and board.piece_type_at(move.from_square) == chess.PAWN
        ):
            victim = chess.PAWN
        if victim is not None:
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            return 100_000 + 16 * victim - attacker
        if move.promotion:
            return 90_000 + move.promotion
        killers = self.killers[ply]
        if move == killers[0]:
            return 80_000
        if move == killers[1]:
            return 79_000
        return min(self.history.get((move.from_square, move.to_square), 0), 60_000)

    def _record_cutoff(self, move: chess.Move, ply: int, depth: int) -> None:
        killers = self.killers[ply]
        if killers[0] != move:
            killers[1] = killers[0]
            killers[0] = move
        k = (move.from_square, move.to_square)
        self.history[k] = self.history.get(k, 0) + depth * depth

    def qsearch(
        self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int
    ) -> int:
        self._tick()
        stand_pat = evaluate(board)
        if qply >= MAX_QPLY or ply >= MAX_PLY:
            return stand_pat
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        captures = list(board.generate_legal_captures())
        if not captures:
            return stand_pat

        def order(m: chess.Move) -> int:
            victim = board.piece_type_at(m.to_square) or chess.PAWN
            attacker = board.piece_type_at(m.from_square) or chess.PAWN
            return 16 * victim - attacker

        captures.sort(key=order, reverse=True)
        best = stand_pat
        for move in captures:
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            if stand_pat + _VALUE[victim] + DELTA_MARGIN < alpha and not move.promotion:
                continue
            board.push(move)
            score = -self.qsearch(board, -beta, -alpha, ply + 1, qply + 1)
            board.pop()
            if score >= beta:
                return score
            best = max(best, score)
            alpha = max(alpha, score)
        return best

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        self._tick()
        if ply >= MAX_PLY:
            return evaluate(board)
        if board.halfmove_clock >= 100:
            return 0
        key = _key(board)
        if self.seen.get(key, 0) > 0:
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_score, tt_move = entry
            if e_depth >= depth:
                s = e_score
                if s > MATE_BOUND:
                    s -= ply
                elif s < -MATE_BOUND:
                    s += ply
                if e_flag == EXACT:
                    return s
                if e_flag == LOWER and s >= beta:
                    return s
                if e_flag == UPPER and s <= alpha:
                    return s

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, 0)

        if _popcount(board.occupied) <= 4 and board.is_insufficient_material():
            return 0

        if (
            allow_null
            and not in_check
            and depth >= 3
            and beta < MATE_BOUND
            and (board.occupied_co[board.turn] & ~(board.pawns | board.kings))
            and evaluate(board) >= beta
        ):
            reduction = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            score = -self.negamax(
                board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
            )
            board.pop()
            if score >= beta:
                return beta if score >= MATE_BOUND else score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        ordered = sorted(
            moves, key=lambda m: self._move_score(board, m, tt_move, ply), reverse=True
        )

        alpha_orig = alpha
        best = -INF
        best_move: chess.Move | None = None
        self.seen[key] = self.seen.get(key, 0) + 1

        for i, move in enumerate(ordered):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and move.promotion is None
            board.push(move)
            gives_check = board.is_check()

            if i == 0:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                reduction = 0
                if (
                    depth >= 3
                    and i >= 3
                    and is_quiet
                    and not in_check
                    and not gives_check
                ):
                    reduction = 1 if i < 8 else 2
                score = -self.negamax(
                    board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                )
                if score > alpha and reduction:
                    score = -self.negamax(
                        board, depth - 1, -alpha - 1, -alpha, ply + 1, True
                    )
                if alpha < score < beta:
                    score = -self.negamax(
                        board, depth - 1, -beta, -alpha, ply + 1, True
                    )
            board.pop()

            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if is_quiet:
                        self._record_cutoff(move, ply, depth)
                    break

        self.seen[key] -= 1

        if best <= alpha_orig:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        stored = best
        if stored > MATE_BOUND:
            stored += ply
        elif stored < -MATE_BOUND:
            stored -= ply
        old = self.tt.get(key)
        if old is None or old[0] <= depth:
            if len(self.tt) >= TT_MAX_ENTRIES:
                self.tt.clear()
            self.tt[key] = (depth, flag, stored, best_move)
        return best


def rank_candidates(
    board: chess.Board,
    moves: list[chess.Move],
    log_probs: list[float],
    weight: float,
    depth: int,
    exact: bool = False,
    max_nodes: int = 0,
) -> list[float]:
    n = len(moves)
    scores = list(log_probs)
    if depth <= 0 or weight <= 0 or n == 0:
        return scores

    searcher = Searcher(max_nodes)
    searcher.prepare(board)
    work = board.copy(stack=False)
    order = sorted(range(n), key=lambda i: -log_probs[i])

    try:
        for d in range(1, depth + 1):
            new_scores = [0.0] * n
            best_comb = -math.inf
            have_best = False
            for i in order:
                if exact or not have_best:
                    lo = -INF
                else:
                    lo_cp = (best_comb - log_probs[i]) / weight * 100.0
                    lo = int(max(-INF + 1, min(INF - 1, math.floor(lo_cp))))
                work.push(moves[i])
                child = searcher.negamax(work, d - 1, -INF, -lo, 1, True)
                work.pop()
                e_pawns = max(-100.0, min(100.0, -child / 100.0))
                comb = log_probs[i] + weight * e_pawns
                new_scores[i] = comb
                if comb > best_comb:
                    best_comb = comb
                    have_best = True
            scores = new_scores
            order = sorted(range(n), key=lambda i: -scores[i])
    except SearchAbort:
        pass
    return scores
