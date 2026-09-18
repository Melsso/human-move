import chess
import numpy as np
import pytest
from chess_shared.inference import masked_softmax, select_move_index, top_k_moves
from chess_shared.move_encoding import NUM_MOVES, legal_move_mask, move_to_index


def test_masked_softmax_zeroes_illegal_moves_exactly():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=NUM_MOVES)
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    legal_indices = [10, 200, 3000]
    for i in legal_indices:
        mask[i] = 1.0

    probs = masked_softmax(logits, mask)
    assert np.isclose(probs.sum(), 1.0)
    illegal_mask = np.ones(NUM_MOVES, dtype=bool)
    illegal_mask[legal_indices] = False
    assert np.all(probs[illegal_mask] == 0.0)
    assert np.all(probs[legal_indices] > 0.0)


def test_masked_softmax_matches_real_board_legal_moves():
    board = chess.Board()
    mask = legal_move_mask(board)
    rng = np.random.default_rng(1)
    logits = rng.normal(size=NUM_MOVES)

    probs = masked_softmax(logits, mask)
    for idx in range(NUM_MOVES):
        if mask[idx] == 0:
            assert probs[idx] == 0.0
    assert np.isclose(probs.sum(), 1.0)


def test_masked_softmax_raises_on_no_legal_moves():
    logits = np.zeros(NUM_MOVES)
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    with pytest.raises(ValueError, match="no legal moves"):
        masked_softmax(logits, mask)


def test_select_move_index_greedy_picks_argmax():
    probs = np.zeros(NUM_MOVES)
    probs[42] = 0.9
    probs[7] = 0.1
    assert select_move_index(probs, temperature=0.0) == 42


def test_select_move_index_never_picks_a_zero_probability_move():
    """
    The property that actually matters end to end: whatever this returns,
    it should never be a move the mask said was illegal, across many
    random legal-move subsets and both greedy and sampled selection.
    """
    rng = np.random.default_rng(2)
    board = chess.Board()
    mask = legal_move_mask(board)

    for _ in range(200):
        logits = rng.normal(size=NUM_MOVES)
        probs = masked_softmax(logits, mask)

        greedy_idx = select_move_index(probs, temperature=0.0)
        assert mask[greedy_idx] == 1.0

        sampled_idx = select_move_index(probs, temperature=1.0, rng=rng)
        assert mask[sampled_idx] == 1.0


def test_top_k_moves_returns_sorted_highest_first():
    probs = np.zeros(NUM_MOVES)
    probs[1] = 0.1
    probs[2] = 0.5
    probs[3] = 0.4

    result = top_k_moves(probs, k=3)
    assert [idx for idx, _ in result] == [2, 3, 1]
    assert result[0][1] == pytest.approx(0.5)


def test_top_k_moves_handles_fewer_nonzero_than_k():
    probs = np.zeros(NUM_MOVES)
    probs[5] = 1.0
    result = top_k_moves(probs, k=5)
    assert len(result) == 1
    assert result[0][0] == 5


def test_full_pipeline_on_real_position_produces_a_legal_move():
    """
    End-to-end sanity check using real board/move encoding, not synthetic
    indices: from a real position, with random model logits standing in
    for an untrained model, the final selected move must decode back into
    something python-chess itself considers legal.
    """
    from chess_shared.move_encoding import index_to_move

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    mask = legal_move_mask(board)
    rng = np.random.default_rng(3)
    logits = rng.normal(size=NUM_MOVES)
    probs = masked_softmax(logits, mask)
    move_idx = select_move_index(probs, temperature=0.0)

    move = index_to_move(move_idx, board)
    assert board.is_legal(move)
    assert move_to_index(move) == move_idx
