import math
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from stockfish import Stockfish
from typing import Optional, Dict, List, Tuple
from utils import AIUtilities, NotationGenerator
from database.database import database

ai = AIUtilities()
notation = NotationGenerator()

class ReviewSystem:
    def __init__(self):
        self.stockfish = self.start_stockfish()
        self.curr_eval: Dict[str, int|str|float] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-eval")
        self._lock = Lock()
        self._pending_future: Optional[Future] = None
        self._evaluation_requested: bool = False
        self._synced_history_len: int = 0

    def start_stockfish(self):
        stockfish = Stockfish(ai.resource_path("stockfish/stockfish-windows-x86-64-avx2.exe"))
        return stockfish

    @staticmethod
    def _starting_fen() -> str:
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def _reset_review_engine(self) -> None:
        self.stockfish.set_fen_position(self._starting_fen())
        self._synced_history_len = 0

    def _sync_review_engine_to_history(self) -> None:
        """Always rebuild review engine from game history for consistency."""
        history = list(database.game_history)
        self._reset_review_engine()
        if history:
            self.stockfish.make_moves_from_current_position(history)
        self._synced_history_len = len(history)

    @staticmethod
    def _first_move_book_name(first_move: str) -> str:
        book_map = {
            "e2e4": "King Pawn Opening",
            "d2d4": "Queen Pawn Opening",
            "c2c4": "English Opening",
            "g1f3": "Reti Opening",
            "b1c3": "Van Geet Opening",
            "b2b3": "Larsen Opening",
            "g2g3": "King Fianchetto Opening",
            "f2f4": "Bird Opening",
            "e2e3": "Van't Kruijs Opening",
            "d2d3": "Mieses Opening",
            "b2b4": "Polish Opening",
            "g2g4": "Grob Opening",
            "a2a3": "Anderssen Opening",
            "h2h3": "Clemenz Opening",
            "a2a4": "Ware Opening",
            "h2h4": "Kadas Opening",
            "c2c3": "Saragossa Opening",
            "f2f3": "Barnes Opening",
            "g1h3": "Amar Opening",
            "b1a3": "Dunst Opening",
        }
        return book_map.get(first_move, "First Move")

    def starting_evaluation(self):
        self._reset_review_engine()
        
        evaluation = self.stockfish.get_evaluation()
        
        if evaluation["type"] == "cp":
            cp = evaluation["value"]
            mate = ""
        else:
            cp = ""
            mate = evaluation["value"]
        
        self.curr_eval = {"color": "white", "cp": cp, "mate": mate, "type": "", "accuracy": ""}

        self.update_evaluation_history()
    
    def update_evaluation_history(self):
        db_eval = database.evaluation_history
        curr_eval = dict(self.curr_eval)
        if not db_eval:
            db_eval.append(curr_eval)
            return
        
        if curr_eval == db_eval[-1]:
            return
        
        db_eval.append(curr_eval)
    
    def get_opening_name(self, fen: str, play: str="") -> str|None:
        try:
            response = requests.get(
                "https://explorer.lichess.ovh/lichess",
                params={"fen": fen, 'play': play},
                timeout=3.0
            )
            
            if response.status_code == 200:
                data = response.json()
                opening = data.get("opening")
                if isinstance(opening, dict):
                    return opening.get("name")
                return None
            return None
        except Exception:
            return None
    
    def calculate_expected_points(self, centipawns: int|float) -> float:
        e = math.e
        return 1 / (1 + e**(-0.0035 * centipawns))
    
    def update_last_move(self) -> None:
        """Synchronize review engine with current game history."""
        if self.stockfish:
            self._sync_review_engine_to_history()
    
    def calculate_accuracy(self, ep_loss: int|float) -> float:
        accuracy = 103.16 * math.e**(-4 * ep_loss) - 3.17
        return round(max(0.0, min(100.0, accuracy)), 2)
    
    
    def calculate_ep_loss(self, before_ep: float, after_ep: float) -> float:
        return max(0.0, before_ep - (1 - after_ep))
    
    def calculate_centipawn(self):
        evaluation = self.stockfish.get_evaluation()
        
        if evaluation["type"] == "cp":
            cp = evaluation["value"]
            mate = ""
        else:
            cp = ""
            mate = evaluation["value"]

        return cp, mate
    
    def evaluate_last_move(self):
        if not database.game_history:
            if not database.evaluation_history:
                self.starting_evaluation()
            return

        self.update_last_move()
        
        cp, mate = self.calculate_centipawn()
        color = "white" if database.current_turn == "black" else "black"
        last_pgn = database.game_pgn[-1] if database.game_pgn else ""
        fen = notation.generate_fen(
            board=database.matrix,
            side_to_move=database.current_turn[0],  # type: ignore[arg-type]
            fullmove_number=database.fullmove,
        )
        opening = self.get_opening_name(fen)
        move_count = len(database.game_history)

        if database.last_forced:
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": "forced", "accuracy": ""}

        elif move_count == 1:
            first_move = database.game_history[0] if database.game_history else ""
            opening_name = opening if opening else self._first_move_book_name(first_move)
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": "book", "opening": opening_name, "accuracy": ""}

        elif opening is not None and move_count <= 14:
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": "book", "opening": opening, "accuracy": ""}
        elif (
            move_count <= 13
            and "x" not in last_pgn
            and isinstance(cp, (int, float))
            and abs(cp) <= 130
        ):
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": "book", "opening": "Unknown Opening", "accuracy": ""}
        elif len(database.get_legal_moves(database.current_turn)) == 0:
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": "best", "accuracy": 100}
        
        else:
            if not database.evaluation_history:
                self.starting_evaluation()
            last_eval = database.evaluation_history[-1]
            cp_after = cp
            cp_before = last_eval["cp"]
            mate_after = mate
            mate_before = last_eval["mate"]
            ep_before = self._eval_to_expected_points(cp_before, mate_before)
            ep_after = self._eval_to_expected_points(cp_after, mate_after)

            ep_loss = self.calculate_ep_loss(ep_before, ep_after)
            accuracy = self.calculate_accuracy(ep_loss)

            move_type = None

            if ep_loss >=0.35:
                move_type = "blunder"
            elif 0.14 <= ep_loss < 0.35:
                move_type = "mistake"
            elif 0.02 <= ep_loss < 0.14:
                move_type = "inaccuracy"
            elif 0.013 <= ep_loss < 0.02:
                move_type = "good"
            elif 0.006 <= ep_loss < 0.013:
                move_type = "excellent"
            else:
                move_type = self.classify_top(cp_before, cp_after, ep_loss)
            
            self.curr_eval = {"color": color, "cp": cp, "mate": mate, "type": move_type, "accuracy": accuracy}    
        
        self.update_evaluation_history()

    def evaluate_last_move_async(self) -> None:
        """Queue non-blocking move evaluation and coalesce bursts safely."""
        with self._lock:
            self._evaluation_requested = True
            if self._pending_future is None or self._pending_future.done():
                self._pending_future = self._executor.submit(self._drain_evaluation_queue)

    @staticmethod
    def _eval_to_expected_points(cp_value, mate_value) -> float:
        if isinstance(cp_value, (int, float)):
            return 1 / (1 + math.e ** (-0.0035 * cp_value))

        if isinstance(mate_value, (int, float)):
            return 1.0 if mate_value > 0 else 0.0

        return 0.5

    def _drain_evaluation_queue(self) -> None:
        while True:
            with self._lock:
                if not self._evaluation_requested:
                    self._pending_future = None
                    return
                self._evaluation_requested = False
            self._evaluate_guarded()

    def _evaluate_guarded(self) -> None:
        try:
            self.evaluate_last_move()
            if database.evaluation_history:
                database.gamelogger.move(f"Review : {database.evaluation_history[-1]}")
        except Exception as e:
            database.gamelogger.error(f"Review evaluation failed: {e}")

    def shutdown(self) -> None:
        with self._lock:
            self._evaluation_requested = False
            self._executor.shutdown(wait=False, cancel_futures=True)
    
    @staticmethod
    def _uci_to_square(square: str) -> Tuple[int, int]:
        return 8 - int(square[1]), ord(square[0]) - ord("a")

    def _parse_uci_move(self, move: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        if len(move) < 4:
            return None
        from_square = self._uci_to_square(move[:2])
        to_square = self._uci_to_square(move[2:4])
        promotion = move[4].lower() if len(move) >= 5 else ""
        return from_square, to_square, promotion

    @staticmethod
    def _piece_value(piece: str) -> int:
        values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
        if not piece or piece == ".":
            return 0
        return values.get(piece.lower(), 0)

    @staticmethod
    def _matrix_piece_to_fen(piece) -> str:
        if piece == 0 or piece == "":
            return "."
        piece_str = str(piece)
        base = piece_str.lstrip("-")[0].lower()
        return base.upper() if "-" not in piece_str else base

    def _board_from_matrix(self) -> List[List[str]]:
        board: List[List[str]] = []
        for row in database.matrix:
            board.append([self._matrix_piece_to_fen(piece) for piece in row])
        return board

    @staticmethod
    def _board_from_fen(fen: str) -> List[List[str]]:
        board_state = fen.split(" ")[0]
        rows = board_state.split("/")
        board: List[List[str]] = []

        for row in rows:
            parsed_row: List[str] = []
            for char in row:
                if char.isdigit():
                    parsed_row.extend(["."] * int(char))
                else:
                    parsed_row.append(char)
            board.append(parsed_row)

        return board

    @staticmethod
    def _inside(row: int, col: int) -> bool:
        return 0 <= row <= 7 and 0 <= col <= 7

    @staticmethod
    def _piece_color(piece: str) -> Optional[str]:
        if not piece or piece == ".":
            return None
        return "white" if piece.isupper() else "black"

    @staticmethod
    def _piece_id_color(piece_id: str) -> str:
        return "black" if str(piece_id).startswith("-") else "white"

    def _find_king(self, board: List[List[str]], color: str) -> Optional[Tuple[int, int]]:
        king = "K" if color == "white" else "k"
        for r in range(8):
            for c in range(8):
                if board[r][c] == king:
                    return (r, c)
        return None

    def _attackers_to_square(
        self,
        board: List[List[str]],
        square: Tuple[int, int],
        attacker_color: str,
    ) -> List[str]:
        row, col = square
        attackers: List[str] = []

        # Pawn attacks
        if attacker_color == "white":
            pawn_sources = [(row + 1, col - 1), (row + 1, col + 1)]
            pawn_char = "P"
        else:
            pawn_sources = [(row - 1, col - 1), (row - 1, col + 1)]
            pawn_char = "p"

        for r, c in pawn_sources:
            if self._inside(r, c) and board[r][c] == pawn_char:
                attackers.append(board[r][c])

        # Knight attacks
        knight_offsets = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        knight_char = "N" if attacker_color == "white" else "n"
        for dr, dc in knight_offsets:
            r, c = row + dr, col + dc
            if self._inside(r, c) and board[r][c] == knight_char:
                attackers.append(board[r][c])

        # Bishop/queen diagonals
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            r, c = row + dr, col + dc
            while self._inside(r, c):
                piece = board[r][c]
                if piece != ".":
                    if self._piece_color(piece) == attacker_color and piece.lower() in ("b", "q"):
                        attackers.append(piece)
                    break
                r += dr
                c += dc

        # Rook/queen straights
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            r, c = row + dr, col + dc
            while self._inside(r, c):
                piece = board[r][c]
                if piece != ".":
                    if self._piece_color(piece) == attacker_color and piece.lower() in ("r", "q"):
                        attackers.append(piece)
                    break
                r += dr
                c += dc

        # King attacks
        king_char = "K" if attacker_color == "white" else "k"
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self._inside(r, c) and board[r][c] == king_char:
                    attackers.append(board[r][c])

        return attackers

    @staticmethod
    def _stockfish_score(move_info: dict) -> int:
        mate = move_info.get("Mate")
        cp = move_info.get("Centipawn")
        if isinstance(mate, int):
            return 100000 - abs(mate) if mate > 0 else -100000 + abs(mate)
        if isinstance(cp, int):
            return cp
        return -99999

    def _rebuild_before_position_data(self) -> Tuple[Optional[List[List[str]]], List[dict]]:
        if not database.game_history:
            return None, []
        history_snapshot = list(database.game_history)

        try:
            self.stockfish.set_fen_position(self._starting_fen())
            if len(history_snapshot) > 1:
                self.stockfish.make_moves_from_current_position(history_snapshot[:-1])

            fen_before = self.stockfish.get_fen_position()
            top_moves = self.stockfish.get_top_moves(3) or []
            before_board = self._board_from_fen(fen_before)

            return before_board, top_moves
        except Exception:
            return None, []
        finally:
            try:
                self._sync_review_engine_to_history()
            except Exception:
                # Keep evaluation resilient; error is reported by caller path if needed.
                pass

    def _is_top_or_close_second(self, move: str, top_moves: List[dict]) -> bool:
        if not top_moves:
            return False

        if top_moves[0].get("Move") == move:
            return True

        if len(top_moves) > 1 and top_moves[1].get("Move") == move:
            gap = abs(self._stockfish_score(top_moves[0]) - self._stockfish_score(top_moves[1]))
            return gap <= 35

        return False

    def _is_only_advantage_move_in_losing_position(self, move: str, top_moves: List[dict], cp_before) -> bool:
        if not isinstance(cp_before, (int, float)) or cp_before >= -300:
            return False

        if not top_moves or top_moves[0].get("Move") != move:
            return False

        best_score = self._stockfish_score(top_moves[0])
        if best_score <= -80:
            return False

        if len(top_moves) == 1:
            return True

        second_score = self._stockfish_score(top_moves[1])
        return (best_score - second_score) >= 120

    def classify_top(self, cp_before, cp_after, ep_loss: Optional[float] = None):
        if ep_loss is None:
            if isinstance(cp_before, (int, float)) and isinstance(cp_after, (int, float)):
                ep_before = self.calculate_expected_points(cp_before)
                ep_after = self.calculate_expected_points(cp_after)
                ep_loss = self.calculate_ep_loss(ep_before, ep_after)
            else:
                ep_loss = 0.0

        last_move = database.game_history[-1] if database.game_history else ""
        last_pgn = database.game_pgn[-1] if database.game_pgn else ""

        # Section 2.3: checkmate moves are directly Best (forced/book handled upstream).
        if "#" in last_pgn:
            return "best"

        if ep_loss >= 0.045:
            return "excellent"

        before_board, top_moves = self._rebuild_before_position_data()
        is_engine_top = bool(top_moves and top_moves[0].get("Move") == last_move)
        top_gap = None
        if len(top_moves) > 1:
            top_gap = abs(self._stockfish_score(top_moves[0]) - self._stockfish_score(top_moves[1]))
        base_class = "best"

        parsed_move = self._parse_uci_move(last_move)
        if not parsed_move:
            return base_class

        from_square, to_square, promotion = parsed_move
        mover_color = "white" if database.current_turn == "black" else "black"
        opponent_color = database.current_turn

        already_lost = isinstance(cp_before, (int, float)) and cp_before < -300
        if not already_lost and database.evaluation_history:
            mate_before = database.evaluation_history[-1].get("mate")
            if isinstance(mate_before, (int, float)) and mate_before < 0:
                already_lost = True

        was_in_check = False
        if before_board is not None:
            king_square = self._find_king(before_board, mover_color)
            if king_square is not None:
                was_in_check = len(self._attackers_to_square(before_board, king_square, opponent_color)) > 0
        elif len(database.game_pgn) > 1:
            was_in_check = "+" in database.game_pgn[-2]

        prelim_ok = (
            ep_loss < 0.008
            and promotion != "q"
            and not was_in_check
            and not already_lost
        )
        if not prelim_ok or ep_loss >= 0.006:
            return base_class

        after_board = self._board_from_matrix()
        to_row, to_col = to_square
        from_row, from_col = from_square

        moved_piece = after_board[to_row][to_col]
        if moved_piece == "." and before_board is not None:
            moved_piece = before_board[from_row][from_col]

        moved_value = self._piece_value(moved_piece)
        captured_piece = "."
        if before_board is not None:
            captured_piece = before_board[to_row][to_col]

        move_is_capture = captured_piece != "." or ("x" in last_pgn)

        defenders_before = self._attackers_to_square(before_board, to_square, opponent_color) if before_board else []
        attackers_after = self._attackers_to_square(after_board, to_square, opponent_color)

        moved_to_attacked_square = len(attackers_after) > 0
        attacked_by_lower_piece = any(self._piece_value(piece) < moved_value for piece in attackers_after)

        is_simple_recapture = False
        if move_is_capture and len(database.game_history) > 1 and len(database.game_pgn) > 1:
            prev_move = database.game_history[-2]
            is_simple_recapture = ("x" in database.game_pgn[-2]) and (prev_move[2:4] == last_move[2:4])

        queen_took_undefended_piece = (
            moved_piece.lower() == "q"
            and move_is_capture
            and len(defenders_before) == 0
        )

        full_sacrifice_detected = (
            moved_to_attacked_square
            and attacked_by_lower_piece
            and not is_simple_recapture
            and not queen_took_undefended_piece
        )

        check_given = "+" in last_pgn or "#" in last_pgn
        pinned_opponent_piece = any(self._piece_id_color(piece_id) == opponent_color for piece_id in database.pins)

        opponent_king_square = self._find_king(after_board, opponent_color)
        king_pressure = 0
        if opponent_king_square is not None:
            king_pressure = len(self._attackers_to_square(after_board, opponent_king_square, mover_color))

        forcing_continuation = check_given or pinned_opponent_piece
        compensation_ok = forcing_continuation or king_pressure >= 2

        from_attackers_before = self._attackers_to_square(before_board, from_square, opponent_color) if before_board else []
        removes_piece_from_danger = len(from_attackers_before) > 0 and not moved_to_attacked_square
        creates_new_threats = forcing_continuation or king_pressure > 0
        opponent_can_ignore = not moved_to_attacked_square and not creates_new_threats
        simplification_reject = removes_piece_from_danger and (not creates_new_threats) and opponent_can_ignore

        captured_value = self._piece_value(captured_piece)
        capture_defended_eq_higher = (
            move_is_capture
            and captured_piece != "."
            and len(defenders_before) > 0
            and captured_value >= moved_value
        )

        only_advantage_move = self._is_only_advantage_move_in_losing_position(last_move, top_moves, cp_before)

        strict_great = (
            ep_loss < 0.003
            and is_engine_top
            and (top_gap is None or top_gap >= 55)
            and not move_is_capture
            and "+" not in last_pgn
            and "#" not in last_pgn
            and not was_in_check
            and not already_lost
        )

        strict_brilliant_sac = (
            full_sacrifice_detected
            and compensation_ok
            and not simplification_reject
            and is_engine_top
            and ep_loss < 0.006
        )
        strict_brilliant_advantage = (
            only_advantage_move
            and is_engine_top
            and ep_loss < 0.004
            and not simplification_reject
        )
        strict_brilliant_capture = (
            capture_defended_eq_higher
            and attacked_by_lower_piece
            and compensation_ok
            and is_engine_top
            and ep_loss < 0.004
            and not simplification_reject
        )

        if strict_brilliant_sac or strict_brilliant_advantage or strict_brilliant_capture:
            return "brilliant"

        if strict_great:
            return "great"

        return base_class
