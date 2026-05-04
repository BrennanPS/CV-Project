import chess
import pandas as pd


# -------------------------------
# 5. Board / FEN Feature Helpers
# -------------------------------

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]


def same_diagonal(sq1, sq2):
    return abs(chess.square_file(sq1) - chess.square_file(sq2)) == \
           abs(chess.square_rank(sq1) - chess.square_rank(sq2))


def is_clear_vertical(board, sq1, sq2):
    file_ = chess.square_file(sq1)
    r1 = chess.square_rank(sq1)
    r2 = chess.square_rank(sq2)

    if r1 == r2:
        return False

    step = 1 if r2 > r1 else -1

    for r in range(r1 + step, r2, step):
        if board.piece_at(chess.square(file_, r)) is not None:
            return False

    return True


def is_clear_diagonal(board, sq1, sq2):
    f1 = chess.square_file(sq1)
    r1 = chess.square_rank(sq1)
    f2 = chess.square_file(sq2)
    r2 = chess.square_rank(sq2)

    if abs(f1 - f2) != abs(r1 - r2):
        return False

    file_step = 1 if f2 > f1 else -1
    rank_step = 1 if r2 > r1 else -1

    f = f1 + file_step
    r = r1 + rank_step

    while f != f2 and r != r2:
        if board.piece_at(chess.square(f, r)) is not None:
            return False

        f += file_step
        r += rank_step

    return True


def count_rook_batteries(board, color):
    rooks = list(board.pieces(chess.ROOK, color))
    count = 0

    for i in range(len(rooks)):
        for j in range(i + 1, len(rooks)):
            if chess.square_file(rooks[i]) == chess.square_file(rooks[j]):
                if is_clear_vertical(board, rooks[i], rooks[j]):
                    count += 1

    return count


def count_queen_rook_batteries(board, color):
    rooks = list(board.pieces(chess.ROOK, color))
    queens = list(board.pieces(chess.QUEEN, color))
    count = 0

    for rook in rooks:
        for queen in queens:
            if chess.square_file(rook) == chess.square_file(queen):
                if is_clear_vertical(board, rook, queen):
                    count += 1

    return count


def count_bishop_queen_batteries(board, color):
    queens = list(board.pieces(chess.QUEEN, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    count = 0

    for queen in queens:
        for bishop in bishops:
            if same_diagonal(queen, bishop) and is_clear_diagonal(board, queen, bishop):
                count += 1

    return count


def has_alekhines_gun(board, color):
    rooks = list(board.pieces(chess.ROOK, color))
    queens = list(board.pieces(chess.QUEEN, color))

    if len(rooks) < 2 or len(queens) < 1:
        return False

    pieces = rooks + queens

    for file_ in set(chess.square_file(sq) for sq in pieces):
        aligned = [sq for sq in pieces if chess.square_file(sq) == file_]

        if len(aligned) >= 3:
            aligned_sorted = sorted(aligned, key=lambda sq: chess.square_rank(sq))

            clear = True
            for i in range(len(aligned_sorted) - 1):
                if not is_clear_vertical(board, aligned_sorted[i], aligned_sorted[i + 1]):
                    clear = False
                    break

            if clear:
                return True

    return False


def count_central_knights(board, color):
    return sum(1 for sq in board.pieces(chess.KNIGHT, color) if sq in EXTENDED_CENTER)


def count_doubled_pawns(board, color):
    pawn_files = [
        chess.square_file(sq)
        for sq in board.pieces(chess.PAWN, color)
    ]

    return sum(pawn_files.count(f) > 1 for f in set(pawn_files))


def count_isolated_pawns(board, color):
    pawn_files = set(
        chess.square_file(sq)
        for sq in board.pieces(chess.PAWN, color)
    )

    isolated = 0

    for f in pawn_files:
        if (f - 1 not in pawn_files) and (f + 1 not in pawn_files):
            isolated += 1

    return isolated


def count_passed_pawns(board, color):
    count = 0
    enemy = not color

    for pawn_sq in board.pieces(chess.PAWN, color):
        file_ = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)

        files_to_check = [f for f in [file_ - 1, file_, file_ + 1] if 0 <= f <= 7]

        is_passed = True

        for enemy_pawn_sq in board.pieces(chess.PAWN, enemy):
            enemy_file = chess.square_file(enemy_pawn_sq)
            enemy_rank = chess.square_rank(enemy_pawn_sq)

            if enemy_file in files_to_check:
                if color == chess.WHITE and enemy_rank > rank:
                    is_passed = False
                    break
                if color == chess.BLACK and enemy_rank < rank:
                    is_passed = False
                    break

        if is_passed:
            count += 1

    return count


def count_attacked_pieces(board, color):
    enemy = not color
    count = 0
    value_sum = 0

    for sq, piece in board.piece_map().items():
        if piece.color == color:
            if board.is_attacked_by(enemy, sq):
                count += 1
                value_sum += PIECE_VALUES.get(piece.piece_type, 0)

    return count, value_sum


def count_hanging_pieces(board, color):
    enemy = not color
    count = 0
    value_sum = 0

    for sq, piece in board.piece_map().items():
        if piece.color == color:
            attacked = board.is_attacked_by(enemy, sq)
            defended = board.is_attacked_by(color, sq)

            if attacked and not defended:
                count += 1
                value_sum += PIECE_VALUES.get(piece.piece_type, 0)

    return count, value_sum


def king_zone_squares(king_sq):
    if king_sq is None:
        return []

    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)

    squares = []

    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            nf = f + df
            nr = r + dr

            if 0 <= nf <= 7 and 0 <= nr <= 7:
                squares.append(chess.square(nf, nr))

    return squares


def count_king_zone_attackers(board, color):
    king_sq = board.king(color)

    if king_sq is None:
        return 0

    enemy = not color
    zone = king_zone_squares(king_sq)

    return sum(len(board.attackers(enemy, sq)) for sq in zone)


# -------------------------------
# 6. FEN / Board Features
# -------------------------------

def fen_to_move_only_features(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    features = {}

    # Turn / phase
    features["white_to_move"] = int(board.turn == chess.WHITE)
    features["black_to_move"] = int(board.turn == chess.BLACK)
    features["has_en_passant"] = int(board.ep_square is not None)
    features["halfmove_clock"] = board.halfmove_clock
    features["fullmove_number"] = board.fullmove_number

    # Castling rights are derived from move history
    features["white_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.WHITE))
    features["white_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.WHITE))
    features["black_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.BLACK))
    features["black_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.BLACK))
    features["white_castling_rights_count"] = (
        features["white_can_castle_kingside"] + features["white_can_castle_queenside"]
    )
    features["black_castling_rights_count"] = (
        features["black_can_castle_kingside"] + features["black_can_castle_queenside"]
    )
    features["castling_rights_diff"] = (
        features["white_castling_rights_count"] - features["black_castling_rights_count"]
    )

    # Tactical / positional formations
    features["white_rook_batteries"] = count_rook_batteries(board, chess.WHITE)
    features["black_rook_batteries"] = count_rook_batteries(board, chess.BLACK)
    features["rook_battery_diff"] = features["white_rook_batteries"] - features["black_rook_batteries"]

    features["white_qr_batteries"] = count_queen_rook_batteries(board, chess.WHITE)
    features["black_qr_batteries"] = count_queen_rook_batteries(board, chess.BLACK)
    features["qr_battery_diff"] = features["white_qr_batteries"] - features["black_qr_batteries"]

    features["white_bq_batteries"] = count_bishop_queen_batteries(board, chess.WHITE)
    features["black_bq_batteries"] = count_bishop_queen_batteries(board, chess.BLACK)
    features["bq_battery_diff"] = features["white_bq_batteries"] - features["black_bq_batteries"]

    features["white_alekhines_gun"] = int(has_alekhines_gun(board, chess.WHITE))
    features["black_alekhines_gun"] = int(has_alekhines_gun(board, chess.BLACK))
    features["alekhines_gun_diff"] = features["white_alekhines_gun"] - features["black_alekhines_gun"]

    features["white_central_knights"] = count_central_knights(board, chess.WHITE)
    features["black_central_knights"] = count_central_knights(board, chess.BLACK)
    features["central_knight_diff"] = features["white_central_knights"] - features["black_central_knights"]

    # Check / mobility
    features["side_to_move_in_check"] = int(board.is_check())
    features["num_legal_moves"] = len(list(board.legal_moves))

    white_mobility = 0
    black_mobility = 0

    turn_backup = board.turn

    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves))

    board.turn = turn_backup

    features["white_mobility"] = white_mobility
    features["black_mobility"] = black_mobility
    features["mobility_diff"] = white_mobility - black_mobility

    # Material / piece counts
    white_material = 0
    black_material = 0

    for piece_type, value in PIECE_VALUES.items():
        piece_name = chess.piece_name(piece_type)

        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))

        features[f"white_{piece_name}s"] = white_count
        features[f"black_{piece_name}s"] = black_count
        features[f"{piece_name}_diff"] = white_count - black_count

        white_material += white_count * value
        black_material += black_count * value

    features["white_material"] = white_material
    features["black_material"] = black_material
    features["material_diff"] = white_material - black_material
    features["material_abs_diff"] = abs(features["material_diff"])

    features["white_total_pieces"] = sum(1 for p in piece_map.values() if p.color == chess.WHITE)
    features["black_total_pieces"] = sum(1 for p in piece_map.values() if p.color == chess.BLACK)
    features["piece_count_diff"] = features["white_total_pieces"] - features["black_total_pieces"]

    # Center control
    white_center_control = 0
    black_center_control = 0

    white_extended_center_control = 0
    black_extended_center_control = 0

    for sq in CENTER_SQUARES:
        white_center_control += len(board.attackers(chess.WHITE, sq))
        black_center_control += len(board.attackers(chess.BLACK, sq))

    for sq in EXTENDED_CENTER:
        white_extended_center_control += len(board.attackers(chess.WHITE, sq))
        black_extended_center_control += len(board.attackers(chess.BLACK, sq))

    features["white_center_control"] = white_center_control
    features["black_center_control"] = black_center_control
    features["center_control_diff"] = white_center_control - black_center_control

    features["white_extended_center_control"] = white_extended_center_control
    features["black_extended_center_control"] = black_extended_center_control
    features["extended_center_control_diff"] = (
        white_extended_center_control - black_extended_center_control
    )

    # King safety
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    features["white_king_attackers"] = (
        len(board.attackers(chess.BLACK, white_king_sq))
        if white_king_sq is not None else 0
    )
    features["black_king_attackers"] = (
        len(board.attackers(chess.WHITE, black_king_sq))
        if black_king_sq is not None else 0
    )
    features["king_attackers_diff"] = (
        features["black_king_attackers"] - features["white_king_attackers"]
    )

    features["white_king_zone_attackers"] = count_king_zone_attackers(board, chess.WHITE)
    features["black_king_zone_attackers"] = count_king_zone_attackers(board, chess.BLACK)
    features["king_zone_attackers_diff"] = (
        features["black_king_zone_attackers"] - features["white_king_zone_attackers"]
    )

    # Pawn structure
    features["white_doubled_pawns"] = count_doubled_pawns(board, chess.WHITE)
    features["black_doubled_pawns"] = count_doubled_pawns(board, chess.BLACK)
    features["doubled_pawn_diff"] = features["white_doubled_pawns"] - features["black_doubled_pawns"]

    features["white_isolated_pawns"] = count_isolated_pawns(board, chess.WHITE)
    features["black_isolated_pawns"] = count_isolated_pawns(board, chess.BLACK)
    features["isolated_pawn_diff"] = features["white_isolated_pawns"] - features["black_isolated_pawns"]

    features["white_passed_pawns"] = count_passed_pawns(board, chess.WHITE)
    features["black_passed_pawns"] = count_passed_pawns(board, chess.BLACK)
    features["passed_pawn_diff"] = features["white_passed_pawns"] - features["black_passed_pawns"]

    # Attacked / hanging pieces
    white_attacked_count, white_attacked_value = count_attacked_pieces(board, chess.WHITE)
    black_attacked_count, black_attacked_value = count_attacked_pieces(board, chess.BLACK)

    features["white_attacked_pieces"] = white_attacked_count
    features["black_attacked_pieces"] = black_attacked_count
    features["attacked_piece_diff"] = black_attacked_count - white_attacked_count

    features["white_attacked_piece_value"] = white_attacked_value
    features["black_attacked_piece_value"] = black_attacked_value
    features["attacked_piece_value_diff"] = black_attacked_value - white_attacked_value

    white_hanging_count, white_hanging_value = count_hanging_pieces(board, chess.WHITE)
    black_hanging_count, black_hanging_value = count_hanging_pieces(board, chess.BLACK)

    features["white_hanging_pieces"] = white_hanging_count
    features["black_hanging_pieces"] = black_hanging_count
    features["hanging_piece_diff"] = black_hanging_count - white_hanging_count

    features["white_hanging_piece_value"] = white_hanging_value
    features["black_hanging_piece_value"] = black_hanging_value
    features["hanging_piece_value_diff"] = black_hanging_value - white_hanging_value

    return features


# -------------------------------
# 7. Move-String Dynamic Features
# -------------------------------

def extract_san_move_features(move_string, move_cutoff=None, recent_window=10):
    moves = str(move_string).split()

    if move_cutoff is not None:
        moves = moves[:min(move_cutoff, len(moves))]

    recent_moves = moves[-recent_window:]

    features = {}

    features["san_total_moves"] = len(moves)

    # Global SAN statistics
    features["capture_count"] = sum("x" in m for m in moves)
    features["check_count"] = sum("+" in m or "#" in m for m in moves)
    features["castle_kingside_count"] = sum(m.startswith("O-O") and not m.startswith("O-O-O") for m in moves)
    features["castle_queenside_count"] = sum(m.startswith("O-O-O") for m in moves)
    features["promotion_count"] = sum("=" in m for m in moves)

    # Recent SAN statistics
    features["recent_capture_count"] = sum("x" in m for m in recent_moves)
    features["recent_check_count"] = sum("+" in m or "#" in m for m in recent_moves)
    features["recent_promotion_count"] = sum("=" in m for m in recent_moves)

    # Piece move counts
    piece_letters = {
        "N": "knight",
        "B": "bishop",
        "R": "rook",
        "Q": "queen",
        "K": "king",
    }

    for letter, name in piece_letters.items():
        features[f"{name}_move_count"] = sum(m.startswith(letter) for m in moves)
        features[f"recent_{name}_move_count"] = sum(m.startswith(letter) for m in recent_moves)

    pawn_moves = [
        m for m in moves
        if len(m) > 0 and m[0] not in piece_letters and not m.startswith("O-O")
    ]

    recent_pawn_moves = [
        m for m in recent_moves
        if len(m) > 0 and m[0] not in piece_letters and not m.startswith("O-O")
    ]

    features["pawn_move_count"] = len(pawn_moves)
    features["recent_pawn_move_count"] = len(recent_pawn_moves)

    # Ratios
    denom = max(len(moves), 1)
    recent_denom = max(len(recent_moves), 1)

    features["capture_rate"] = features["capture_count"] / denom
    features["check_rate"] = features["check_count"] / denom
    features["pawn_move_rate"] = features["pawn_move_count"] / denom
    features["piece_move_rate"] = 1.0 - features["pawn_move_rate"]

    features["recent_capture_rate"] = features["recent_capture_count"] / recent_denom
    features["recent_check_rate"] = features["recent_check_count"] / recent_denom
    features["recent_pawn_move_rate"] = features["recent_pawn_move_count"] / recent_denom
    features["recent_piece_move_rate"] = 1.0 - features["recent_pawn_move_rate"]

    # White/black move imbalance features
    white_moves = moves[0::2]
    black_moves = moves[1::2]

    features["white_capture_count"] = sum("x" in m for m in white_moves)
    features["black_capture_count"] = sum("x" in m for m in black_moves)
    features["capture_count_diff"] = features["white_capture_count"] - features["black_capture_count"]

    features["white_check_count"] = sum("+" in m or "#" in m for m in white_moves)
    features["black_check_count"] = sum("+" in m or "#" in m for m in black_moves)
    features["check_count_diff"] = features["white_check_count"] - features["black_check_count"]

    return features


# -------------------------------
# 8. Strategy History Features
# -------------------------------

def current_strategy_state(board, color):
    return {
        "rook_battery": int(count_rook_batteries(board, color) > 0),
        "qr_battery": int(count_queen_rook_batteries(board, color) > 0),
        "bq_battery": int(count_bishop_queen_batteries(board, color) > 0),
        "alekhines_gun": int(has_alekhines_gun(board, color)),
        "strong_center_knights": int(count_central_knights(board, color) >= 2),
    }


def compute_ever_strategy_features_by_moves(move_string, move_cutoff):
    moves = str(move_string).split()
    cutoff = min(move_cutoff, len(moves))

    history = {
        "white_ever_rook_battery": 0,
        "black_ever_rook_battery": 0,
        "white_ever_qr_battery": 0,
        "black_ever_qr_battery": 0,
        "white_ever_bq_battery": 0,
        "black_ever_bq_battery": 0,
        "white_ever_alekhines_gun": 0,
        "black_ever_alekhines_gun": 0,
        "white_ever_strong_center_knights": 0,
        "black_ever_strong_center_knights": 0,
    }

    board = chess.Board()

    try:
        for i in range(cutoff):
            board.push_san(moves[i])

            white_state = current_strategy_state(board, chess.WHITE)
            black_state = current_strategy_state(board, chess.BLACK)

            for key in white_state:
                if white_state[key]:
                    history[f"white_ever_{key}"] = 1
                if black_state[key]:
                    history[f"black_ever_{key}"] = 1

    except Exception:
        pass

    return history


def compute_strategy_history_features_by_moves(move_string, move_cutoff, window=10):
    moves = str(move_string).split()
    cutoff = min(move_cutoff, len(moves))
    start = max(0, cutoff - window)

    history = {
        "white_recent_rook_battery": 0,
        "black_recent_rook_battery": 0,
        "white_recent_qr_battery": 0,
        "black_recent_qr_battery": 0,
        "white_recent_bq_battery": 0,
        "black_recent_bq_battery": 0,
        "white_recent_alekhines_gun": 0,
        "black_recent_alekhines_gun": 0,
        "white_recent_strong_center_knights": 0,
        "black_recent_strong_center_knights": 0,

        "white_recent_rook_battery_count": 0,
        "black_recent_rook_battery_count": 0,
        "white_recent_qr_battery_count": 0,
        "black_recent_qr_battery_count": 0,
        "white_recent_bq_battery_count": 0,
        "black_recent_bq_battery_count": 0,
        "white_recent_alekhines_gun_count": 0,
        "black_recent_alekhines_gun_count": 0,
        "white_recent_strong_center_knight_count": 0,
        "black_recent_strong_center_knight_count": 0,
    }

    board = chess.Board()

    try:
        for i in range(start):
            board.push_san(moves[i])

        prev_white = current_strategy_state(board, chess.WHITE)
        prev_black = current_strategy_state(board, chess.BLACK)

        for i in range(start, cutoff):
            board.push_san(moves[i])

            curr_white = current_strategy_state(board, chess.WHITE)
            curr_black = current_strategy_state(board, chess.BLACK)

            for key in curr_white:
                if curr_white[key]:
                    history[f"white_recent_{key}"] = 1

                if curr_black[key]:
                    history[f"black_recent_{key}"] = 1

                white_count_name = (
                    f"white_recent_{key}_count"
                    if key != "strong_center_knights"
                    else "white_recent_strong_center_knight_count"
                )

                black_count_name = (
                    f"black_recent_{key}_count"
                    if key != "strong_center_knights"
                    else "black_recent_strong_center_knight_count"
                )

                if curr_white[key] and not prev_white[key]:
                    history[white_count_name] += 1

                if curr_black[key] and not prev_black[key]:
                    history[black_count_name] += 1

            prev_white = curr_white
            prev_black = curr_black

    except Exception:
        pass

    return history


def add_all_move_only_features(df, recent_window=10):
    df = df.copy().reset_index(drop=True)

    fen_features = pd.DataFrame([
        fen_to_move_only_features(fen)
        for fen in df["fen"]
    ])

    san_features = pd.DataFrame([
        extract_san_move_features(
            row["moves_clean"],
            move_cutoff=row["num_moves_used"],
            recent_window=recent_window
        )
        for _, row in df.iterrows()
    ])

    recent_strategy_features = pd.DataFrame([
        compute_strategy_history_features_by_moves(
            row["moves_clean"],
            row["num_moves_used"],
            window=recent_window
        )
        for _, row in df.iterrows()
    ])

    ever_strategy_features = pd.DataFrame([
        compute_ever_strategy_features_by_moves(
            row["moves_clean"],
            row["num_moves_used"]
        )
        for _, row in df.iterrows()
    ])

    return pd.concat(
        [
            df,
            fen_features.reset_index(drop=True),
            san_features.reset_index(drop=True),
            recent_strategy_features.reset_index(drop=True),
            ever_strategy_features.reset_index(drop=True),
        ],
        axis=1
    )


def build_live_feature_row(move_list, recent_window=10):
    board = chess.Board()
    valid_moves = []

    for san in move_list:
        try:
            board.push_san(san)
            valid_moves.append(san)
        except Exception:
            break

    moves_clean = " ".join(valid_moves)
    num_moves_used = len(valid_moves)

    live_df = pd.DataFrame([{
        "moves_clean": moves_clean,
        "moves_truncated": moves_clean,
        "fen": board.fen(),
        "num_moves_used": num_moves_used,
        "total_game_moves": max(num_moves_used, 1),
        "move_progress_fraction": 1.0,
        "terminal_backed_up": 0,
    }])

    return add_all_move_only_features(live_df, recent_window=recent_window)