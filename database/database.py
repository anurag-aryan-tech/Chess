"""
Chess Game Database Module
Manages game state, board position, and piece tracking for a chess game.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, List
from dataclasses import dataclass, field
from stockfish import Stockfish

# ==================== MOVE RESULT DATACLASS ====================

@dataclass
class MoveResult:
    """Result of executing a game state transition"""
    # Move information
    from_square: Tuple[int, int]
    to_square: Tuple[int, int]
    piece: str

    # Move type flags
    is_capture: bool
    is_castling: bool
    is_en_passant: bool
    is_promotion: bool
    promotion_piece: Optional[str] = None  # The piece chosen for promotion (q, r, b, n)

    # State changes
    turn_changed: bool = True
    fullmove_incremented: bool = False

    # For notation generation
    chess_notation: Optional[str] = None
    stockfish_notation: Optional[str] = None

    disambiguation: Optional[str] = ""


# ==================== GAME STATE MANAGER ====================

@dataclass
class GameStateManager:
    """
    Centralized manager for all game state transitions.
    Ensures atomic, consistent updates to the game state.
    All move validation and state updates happen here.
    """
    def __init__(self, db: 'Database') -> None:
            self.database = db

    def execute_move(
        self,
        from_square: Tuple[int, int],
        to_square: Tuple[int, int],
        promotion_piece: Optional[str] = None
    ) -> Optional[MoveResult]:
        """
        Execute a complete game state transition.

        Args:
            from_square: Starting position (row, col)
            to_square: Target position (row, col)
            promotion_piece: For pawn promotion (q, r, b, n), None otherwise

        Returns:
            MoveResult with all state changes, or None if move is invalid
        """
        piece = self.database.matrix[from_square[0], from_square[1]]

        if piece == 0 or not isinstance(piece, str):
            return None

        base_piece = piece[0] if "-" not in piece else piece[1]

        # Capture the captured piece BEFORE modifying the board
        captured_piece = self.database.matrix[to_square[0], to_square[1]] if self.database.matrix[to_square[0], to_square[1]] != 0 else None

        # Determine move type and validate
        is_castling = self._is_castling(piece, from_square, to_square)
        is_en_passant = self._is_en_passant(piece, from_square, to_square)
        is_promotion = self._is_pawn_promotion(piece, to_square, promotion_piece)
        is_capture = self._is_capture(to_square, is_en_passant)

        ep_captured_pawn: Optional[str] = None
        if is_en_passant:
            ep_val = self.database.matrix[from_square[0], to_square[1]]
            ep_captured_pawn = ep_val if ep_val != 0 and isinstance(ep_val, str) else None

        disambiguation = self._get_disambiguation(to_square, from_square, base_piece)

        # Create result object
        result = MoveResult(
            from_square=from_square,
            to_square=to_square,
            piece=piece,
            is_capture=is_capture,
            is_castling=is_castling,
            is_en_passant=is_en_passant,
            is_promotion=is_promotion,
            promotion_piece=promotion_piece,
            disambiguation=disambiguation
        )

        # Execute the move atomically
        self._update_board(from_square, to_square, result)
        self._update_castling_rights(piece, from_square)
        self._update_en_passant(piece, from_square, to_square, is_capture)
        self._update_piece_lists(piece, to_square, promotion_piece, is_capture, is_en_passant, captured_piece, ep_captured_pawn)
        self._update_turn_and_fullmove(result)

        return result

    def _is_castling(self, piece: str, from_square: Tuple[int, int], to_square: Tuple[int, int]) -> bool:
        """Check if move is castling"""
        return "k1" in piece and abs(from_square[1] - to_square[1]) == 2

    def _is_en_passant(self, piece: str, from_square: Tuple[int, int], to_square: Tuple[int, int]) -> bool:
        """Check if move is en passant capture"""
        if "p" not in piece:
            return False

        # Check if it's a diagonal pawn move
        if abs(from_square[1] - to_square[1]) != 1:
            return False

        # Check if target square is en passant square and is empty
        target_notation = self._matrix_to_chess(to_square)
        return (
            self.database.en_passant == target_notation and
            self.database.matrix[to_square[0], to_square[1]] == 0
        )

    def _is_pawn_promotion(self, piece: str, to_square: Tuple[int, int], promotion_piece: Optional[str]) -> bool:
        """Check if pawn reaches promotion rank"""
        return "p" in piece and to_square[0] in [0, 7] and promotion_piece is not None

    def _is_capture(self, to_square: Tuple[int, int], is_en_passant: bool) -> bool:
        """Check if move is a capture"""
        if is_en_passant:
            return True
        target = self.database.matrix[to_square[0], to_square[1]]
        return target != 0 and isinstance(target, str)

    def _get_disambiguation(
        self,
        to_square: Tuple[int, int],
        from_square: Tuple[int, int],
        base_piece: Optional[str]
    ) -> str:
        """
        Return the minimal disambiguation string per PGN standard:
        - ""        if no other piece of same type can reach to_square
        - file      if pieces are on different files
        - rank      if pieces are on the same file
        - full sq   if neither file nor rank alone disambiguates
        """
        if not base_piece:
            return ""

        legal_moves = (
            self.database.white_legal_moves
            if self.database.current_turn == "white"
            else self.database.black_legal_moves
        )

        # Collect from_squares of all same-type pieces that can reach to_square
        ambiguous_from_squares: List[Tuple[int, int]] = []

        for piece, moves in legal_moves.items():
            if not isinstance(piece, str):
                continue
            piece_type = piece[1] if piece.startswith("-") else piece[0]
            if piece_type.lower() != base_piece.lower():
                continue
            if any(np.array_equal(to_square, move) for move in moves):
                try:
                    pos = self.database.white_legal_moves  # just to find position
                    # find the actual square this piece is on
                    result = np.where(self.database.matrix == piece)
                    if len(result[0]) > 0:
                        ambiguous_from_squares.append(
                            (int(result[0][0]), int(result[1][0]))
                        )
                except Exception:
                    continue

        # Only one piece can reach to_square — no disambiguation needed
        if len(ambiguous_from_squares) <= 1:
            return ""

        from_file = from_square[1]
        from_rank = from_square[0]

        files = [sq[1] for sq in ambiguous_from_squares]
        ranks = [sq[0] for sq in ambiguous_from_squares]

        # Files are all different — use file
        if files.count(from_file) == 1:
            return chr(ord('a') + from_file)

        # Files not unique but ranks are — use rank
        if ranks.count(from_rank) == 1:
            return str(8 - from_rank)

        # Neither unique — use full square
        return chr(ord('a') + from_file) + str(8 - from_rank)

    def _update_board(self, from_square: Tuple[int, int], to_square: Tuple[int, int], result: MoveResult) -> None:
        """Update board matrix for the move"""
        piece = result.piece

        # Handle castling specially (includes rook movement)
        if result.is_castling:
            self._execute_castling(from_square, to_square)
        # Handle en passant specially (remove captured pawn)
        elif result.is_en_passant:
            self.database.matrix[from_square[0], to_square[1]] = 0  # Remove captured pawn
            self.database.matrix[from_square[0], from_square[1]] = 0
            self.database.matrix[to_square[0], to_square[1]] = piece
        # Handle promotion (add promoted piece, remove pawn)
        elif result.is_promotion:
            self.database.matrix[from_square[0], from_square[1]] = 0
            # Piece will be set to promoted piece in _update_piece_lists
            self.database.matrix[to_square[0], to_square[1]] = piece  # Temporarily set pawn
        # Normal move
        else:
            self.database.matrix[from_square[0], from_square[1]] = 0
            self.database.matrix[to_square[0], to_square[1]] = piece

    def _execute_castling(self, from_square: Tuple[int, int], to_square: Tuple[int, int]) -> None:
        """Execute castling move (king and rook)"""
        piece = self.database.matrix[from_square[0], from_square[1]]

        self.database.matrix[from_square[0], from_square[1]] = 0

        if to_square[1] == 6:  # Kingside castling
            rook = piece.replace("k1", "r2")
            self.database.matrix[to_square[0], 7] = 0  # Remove rook from h-file
            self.database.matrix[to_square[0], to_square[1]] = piece  # King to g-file
            self.database.matrix[to_square[0], to_square[1]-1] = rook  # Rook to f-file
        else:  # Queenside castling (to_square[1] == 2)
            rook = piece.replace("k1", "r1")
            self.database.matrix[to_square[0], 0] = 0  # Remove rook from a-file
            self.database.matrix[to_square[0], to_square[1]] = piece  # King to c-file
            self.database.matrix[to_square[0], to_square[1]+1] = rook  # Rook to d-file

    def _update_castling_rights(self, piece: str, from_square: Tuple[int, int]) -> None:
        """Update castling rights based on piece movement"""
        if "k1" in piece:
            if "-" in piece:
                self.database.k1_black_moved = True
            else:
                self.database.k1_moved = True
        elif "r1" in piece:
            if "-" in piece:
                self.database.r1_black_moved = True
            else:
                self.database.r1_moved = True
        elif "r2" in piece:
            if "-" in piece:
                self.database.r2_black_moved = True
            else:
                self.database.r2_moved = True

    def _update_en_passant(
        self,
        piece: str,
        from_square: Tuple[int, int],
        to_square: Tuple[int, int],
        is_capture: bool
    ) -> None:
        """Update en passant state"""
        if "p" not in piece:
            # Non-pawn moves clear en passant
            self.database.en_passant = ""
            if "-" in piece:
                self.database.black_last_pawn = None
            else:
                self.database.white_last_pawn = None
            return

        # Pawn move
        distance = abs(from_square[0] - to_square[0])

        if distance == 2:
            # Double pawn push - set en passant target
            if "-" in piece:
                en_passant_square = (to_square[0] - 1, to_square[1])
                self.database.black_last_pawn = to_square
            else:
                en_passant_square = (to_square[0] + 1, to_square[1])
                self.database.white_last_pawn = to_square

            self.database.en_passant = self._matrix_to_chess(en_passant_square)
        else:
            # Single move or capture - clear en passant
            self.database.en_passant = ""
            if "-" in piece:
                self.database.black_last_pawn = None
            else:
                self.database.white_last_pawn = None

    def _update_piece_lists(
        self,
        piece: str,
        to_square: Tuple[int, int],
        promotion_piece: Optional[str],
        is_capture: bool,
        is_en_passant: bool,
        captured_piece: Optional[str] = None,
        ep_captured_pawn: Optional[str] = None,   # NEW
    ) -> None:
        is_white = "-" not in piece

        if is_capture:
            if is_en_passant:
                # Use the exact piece identifier we snapshotted before board modification
                if ep_captured_pawn:
                    if is_white:
                        self.database.black_pieces = self.database.black_pieces[
                            self.database.black_pieces != ep_captured_pawn
                        ]
                    else:
                        self.database.white_pieces = self.database.white_pieces[
                            self.database.white_pieces != ep_captured_pawn
                        ]
            else:
                if captured_piece:
                    if is_white:
                        self.database.black_pieces = self.database.black_pieces[
                            self.database.black_pieces != captured_piece
                        ]
                    else:
                        self.database.white_pieces = self.database.white_pieces[
                            self.database.white_pieces != captured_piece
                        ]

        # Handle promotion
        if promotion_piece:
            # Remove pawn from piece list
            if is_white:
                self.database.white_pieces = self.database.white_pieces[self.database.white_pieces != piece]
            else:
                self.database.black_pieces = self.database.black_pieces[self.database.black_pieces != piece]

            # Add promoted piece
            promoted = self._get_promoted_piece(promotion_piece, "black" if "-" in piece else "white")
            if is_white:
                self.database.white_pieces = np.append(self.database.white_pieces, promoted)
            else:
                self.database.black_pieces = np.append(self.database.black_pieces, promoted)

            # Update board
            self.database.matrix[to_square[0], to_square[1]] = promoted

    def _get_promoted_piece(self, base: str, color: Literal["white", "black"]) -> str:
        """
        Generate next piece identifier for promotion.

        Args:
            base: Piece type ('q', 'r', 'b', 'n')
            color: Piece color

        Returns:
            Next available piece identifier (e.g., 'q3', '-b2')
        """
        pieces = database.white_pieces if color == "white" else database.black_pieces
        prefix = "-" if color == "black" else ""

        # Find highest number for this piece type
        max_num = 0
        for piece in pieces.flatten():
            if base in piece:
                try:
                    num = int(piece[-1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue

        return f"{prefix}{base}{max_num + 1}"

    def _update_turn_and_fullmove(self, result: MoveResult) -> None:
        """Update current turn and fullmove counter"""
        # Toggle turn
        self.database.current_turn = "black" if self.database.current_turn == "white" else "white"
        result.turn_changed = True

        # Increment fullmove after black's move
        if self.database.current_turn == "white":
            self.database.fullmove += 1
            result.fullmove_incremented = True

    def _matrix_to_chess(self, square: Tuple[int, int]) -> str:
        """Convert (row, col) to chess notation (e.g., 'e4')"""
        row, col = square
        return chr(ord('a') + col) + str(8 - row)


# ==================== CONSTANTS ====================

INITIAL_WHITE_PIECES = np.array([
    [f"p{i}" for i in range(1, 9)],
    ['r1', 'n1', 'b1', 'q1', 'k1', 'b2', 'n2', 'r2'],
], dtype=object)

INITIAL_BLACK_PIECES = np.array([
    ['-r1', '-n1', '-b1', '-q1', '-k1', '-b2', '-n2', '-r2'],
    [f"-p{i}" for i in range(1, 9)]
], dtype=object)


# ==================== GAME STATE ====================

@dataclass
class CastlingRights:
    """Track which pieces have moved (affects castling)"""
    white_king_moved: bool = False
    white_rook_kingside_moved: bool = False
    white_rook_queenside_moved: bool = False
    black_king_moved: bool = False
    black_rook_kingside_moved: bool = False
    black_rook_queenside_moved: bool = False

    def can_castle_white_kingside(self) -> bool:
        return not self.white_king_moved and not self.white_rook_kingside_moved

    def can_castle_white_queenside(self) -> bool:
        return not self.white_king_moved and not self.white_rook_queenside_moved

    def can_castle_black_kingside(self) -> bool:
        return not self.black_king_moved and not self.black_rook_kingside_moved

    def can_castle_black_queenside(self) -> bool:
        return not self.black_king_moved and not self.black_rook_queenside_moved

# =================== GAMELOGGER CLASS ===================

class GameLogger:
    """Concise, informative game logging"""

    ENABLED = True
    _last_type: str = ""  # Track last log type for spacing

    @staticmethod
    def _print(log_type: str, message: str) -> None:
        """Core print method with automatic spacing between different types"""
        if not GameLogger.ENABLED:
            return

        # Add blank line when type changes
        if GameLogger._last_type and GameLogger._last_type != log_type:
            print()

        print(f"[{log_type}] {message}")
        GameLogger._last_type = log_type

    @staticmethod
    def init(message: str) -> None:
        GameLogger._print("INIT", message)

    @staticmethod
    def game(message: str) -> None:
        GameLogger._print("GAME", message)

    @staticmethod
    def ai(message: str) -> None:
        GameLogger._print("AI  ", message)

    @staticmethod
    def move(message: str) -> None:
        GameLogger._print("MOVE", message)

    @staticmethod
    def error(message: str) -> None:
        GameLogger._print("ERR ", message)

    @staticmethod
    def warn(message: str) -> None:
        GameLogger._print("WARN", message)

gamelogger = GameLogger()

# ==================== DATABASE CLASS ====================

class Database:
    """
    Central game state manager.
    """

    def __init__(self, save_path: Optional[Path] = None) -> None:
        # File management
        self.matrix_path = save_path or Path("database/matrix.json")

        self.gamelogger = gamelogger

        # Game state manager - centralized move execution
        self.state_manager = GameStateManager(self)

        # Game state
        self.current_turn: Literal["white", "black"] = "white"
        self.en_passant: str = ""
        self.fullmove: int = 1

        # Castling tracking
        self.r1_moved: bool = False
        self.r2_moved: bool = False
        self.k1_moved: bool = False
        self.r1_black_moved: bool = False
        self.r2_black_moved: bool = False
        self.k1_black_moved: bool = False

        # En passant tracking
        self.black_last_pawn: Optional[Tuple[int, int]] = None
        self.white_last_pawn: Optional[Tuple[int, int]] = None

        # Piece tracking (CHANGED: Using module constants for clarity)
        self.white_pieces: np.ndarray = INITIAL_WHITE_PIECES.copy()
        self.black_pieces: np.ndarray = INITIAL_BLACK_PIECES.copy()

        # Board state
        self.matrix: np.ndarray = np.zeros((8, 8), dtype=object)

        # Legal moves cache
        self.white_legal_moves: Dict[str, np.ndarray] = {}
        self.black_legal_moves: Dict[str, np.ndarray] = {}

        # Pin tracking
        self.pins: Dict[str, Tuple[int, int]] = {}

        # Stockfish
        self.stockfish: Optional[Stockfish] = None
        self.game_history: List[str] = []
        self.stockfish_move_time: Optional[int] = None
        self.game_pgn: List[str] = []
        self.fen_history: List[str] = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        # Evaluation
        self.evaluation_history: list = []
        self.last_forced: bool = False

        self.ai_total_moves: int = 0
        self.ai_blunder_moves: int = 0

        # PGN Metadata
        self.pgn_event: str = "-"
        self.pgn_date: str = "-"
        self.pgn_round: str = "-"
        self.pgn_white: str = "-"
        self.pgn_black: str = "-"
        self.pgn_result: str = "*"
        self.pgn_white_elo: str = "-"
        self.pgn_black_elo: str = "-"
        self.pgn_time_control: str = "-"
        self.pgn_termination: str = "-"  # How the game ended

        # Initialize board
        self.initialize_matrix()
        self.save_matrix()
        self.initialize_legal_moves()

    def initialize_matrix(self) -> None:
        """Set up initial board position"""
        self.matrix[:2, :] = self.black_pieces
        self.matrix[6:, :] = self.white_pieces
        gamelogger.init("Board ready")

    def save_matrix(self) -> None:
        """Save current board state to JSON file"""
        # CHANGED: Better error handling
        try:
            self.matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_data = json.dumps({'matrix': self.matrix.tolist()})
            self.matrix_path.write_text(matrix_data)

        except Exception as e:
            gamelogger.error("Error saving game: " + str(e))

    def import_matrix(self) -> None:
        """Load board state from JSON file"""
        # CHANGED: Better error handling
        try:
            if not self.matrix_path.exists():
                gamelogger.warn("No saved game found")
                return

            matrix_data = json.loads(self.matrix_path.read_text())
            self.matrix = np.array(matrix_data['matrix'], dtype=object)
            gamelogger.init("Game loaded successfully")
        except Exception as e:
            gamelogger.error("Error loading game: " + str(e))

    def initialize_legal_moves(self) -> None:
        """Initialize empty legal moves dictionaries"""
        self.white_legal_moves = {
            piece: np.array([])
            for piece in self.white_pieces.flatten()
        }
        self.black_legal_moves = {
            piece: np.array([])
            for piece in self.black_pieces.flatten()
        }

    def get_legal_moves(self, color: Literal["white", "black"]) -> Dict[str, np.ndarray]:
        """Get legal moves for a specific color"""
        return self.white_legal_moves if color == "white" else self.black_legal_moves

    def reset(self) -> None:
        """Reset database to initial state"""
        # CHANGED: Explicit reset instead of __init__ call (clearer intent)
        self.current_turn = "white"
        self.en_passant = ""
        self.fullmove = 1

        # Reinitialize state manager
        self.state_manager = GameStateManager(self)

        # Reset castling flags
        self.r1_moved = False
        self.r2_moved = False
        self.k1_moved = False
        self.r1_black_moved = False
        self.r2_black_moved = False
        self.k1_black_moved = False

        # Reset pawn tracking
        self.black_last_pawn = None
        self.white_last_pawn = None

        # Reset pieces (use copies to avoid mutation)
        self.white_pieces = INITIAL_WHITE_PIECES.copy()
        self.black_pieces = INITIAL_BLACK_PIECES.copy()

        # Reset board
        self.matrix = np.zeros((8, 8), dtype=object)

        # Reset Game History
        self.game_history = []
        self.game_pgn = []
        self.fen_history = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

        # Reset evaluation history
        self.evaluation_history: list = []

        # Clear caches
        self.white_legal_moves.clear()
        self.black_legal_moves.clear()
        self.pins.clear()

        # Reinitialize
        self.initialize_matrix()
        self.save_matrix()
        self.initialize_legal_moves()

        self.ai_blunder_moves = 0
        self.ai_total_moves = 0

        gamelogger.init("Game reset successfully")

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"Database(turn={self.current_turn}, "
            f"move={self.fullmove}, "
            f"en_passant={self.en_passant or '-'})"
        )


# Singleton instance
database = Database()
