"""
Chess Game Review System
========================

A comprehensive post-game analysis system that evaluates chess moves using Stockfish,
classifies them into categories (Brilliant, Great, Best, Excellent, Good, Miss, 
Inaccuracy, Mistake, Blunder, Book, Forced), and calculates accuracy percentages.

Author: Anurag Aryan
Version: 2.0
Date: February 19, 2026

Key Features:
-------------
- Chess.com-style move classification
- Expected points model for accuracy calculation
- Opening detection via Lichess API
- Brilliant move detection with sacrifice analysis
- Asynchronous evaluation (non-blocking UI)
- Mate score handling with distance awareness

Design Philosophy:
------------------
This system follows a strict classification priority:
1. Forced (only legal move)
2. Book (opening theory)
3. Checkmate (game-ending move)
4. Evaluation-based (using expected points loss)
5. Special upgrades (Brilliant, Great)

All calculations use the expected points model where a position's score
is converted to a probability-like value (0.0 to 1.0) representing winning chances.
"""

import math
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from stockfish import Stockfish
from typing import Optional, Dict, List, Tuple, Literal
from dataclasses import dataclass

from utils import AIUtilities, NotationGenerator
from database.database import database


# ==================== CONSTANTS ====================

# Expected points formula constant
EP_SIGMOID_CONSTANT = 0.0035

# Accuracy formula constants
ACCURACY_MULTIPLIER = 103.16
ACCURACY_EXPONENT = -4
ACCURACY_OFFSET = -3.17

# Mate score decay (how much EP drops per mate move)
MATE_DISTANCE_DECAY = 0.005
MAX_MATE_DISTANCE_CONSIDERED = 10

# Classification thresholds (expected points loss)
THRESHOLD_BLUNDER = 0.35
THRESHOLD_MISTAKE = 0.22
THRESHOLD_INACCURACY = 0.15
THRESHOLD_MISS = 0.12
THRESHOLD_GOOD = 0.08
THRESHOLD_EXCELLENT = 0.045
THRESHOLD_BEST = 0.025
THRESHOLD_GREAT = 0.01
THRESHOLD_BRILLIANT = 0.01

# Book move detection
BOOK_MAX_MOVE = 14  # Check opening API until this move
BOOK_HEURISTIC_MAX_MOVE = 12  # Fallback heuristic until this move
BOOK_HEURISTIC_MAX_CP = 150  # Consider quiet positions as book

# Brilliant detection
BRILLIANT_GREAT_GAP_THRESHOLD = 30  # Centipawns gap for "Great" moves
BRILLIANT_SACRIFICE_KING_PRESSURE = 2  # Attackers needed for king pressure

# API configuration
LICHESS_API_URL = "https://explorer.lichess.ovh/lichess"
LICHESS_API_TIMEOUT = 3.0

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ==================== DATA CLASSES ====================

@dataclass
class EvaluationData:
    """
    Represents a position evaluation.
    
    Attributes:
        color: Player who just moved ('white' or 'black')
        cp: Centipawn evaluation (int/float) or empty string
        mate: Mate score (positive = winning, negative = losing) or empty string
        move_type: Classification (brilliant, great, best, excellent, good, miss, 
                   inaccuracy, mistake, blunder, book, forced)
        accuracy: Move accuracy percentage (0-100) or empty string
        opening: Opening name (only for book moves)
    """
    color: str
    cp: int | float | str
    mate: int | float | str
    move_type: str
    accuracy: float | str = ""
    opening: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        result = {
            "color": self.color,
            "cp": self.cp,
            "mate": self.mate,
            "type": self.move_type,
            "accuracy": self.accuracy
        }
        if self.opening:
            result["opening"] = self.opening
        return result


# ==================== OPENING BOOK UTILITIES ====================

class OpeningBook:
    """
    Manages opening detection using Lichess API and fallback heuristics.
    """
    
    # First move opening names (fallback when API unavailable)
    FIRST_MOVE_NAMES = {
        "e2e4": "King's Pawn Opening",
        "d2d4": "Queen's Pawn Opening",
        "c2c4": "English Opening",
        "g1f3": "Réti Opening",
        "b1c3": "Van't Kruijs Opening",
        "b2b3": "Larsen's Opening",
        "g2g3": "King's Fianchetto Opening",
        "f2f4": "Bird's Opening",
        "e2e3": "Van't Kruijs Opening",
        "d2d3": "Mieses Opening",
        "b2b4": "Polish Opening",
        "g2g4": "Grob's Attack",
        "a2a3": "Anderssen's Opening",
        "h2h3": "Clemenz Opening",
        "a2a4": "Ware Opening",
        "h2h4": "Kadas Opening",
        "c2c3": "Saragossa Opening",
        "f2f3": "Barnes Opening",
        "g1h3": "Amar Opening",
        "b1a3": "Sodium Attack",
    }
    
    @staticmethod
    def get_opening_from_api(fen: str, play: str = "") -> Optional[str]:
        """
        Query Lichess API for opening name.
        
        Args:
            fen: Position in FEN format
            play: Moves from starting position (optional)
            
        Returns:
            Opening name or None if not found/error
        """
        try:
            response = requests.get(
                LICHESS_API_URL,
                params={"fen": fen, "play": play},
                timeout=LICHESS_API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                opening = data.get("opening")
                if isinstance(opening, dict):
                    return opening.get("name")
            return None
        except Exception as e:
            # Silently fail - don't spam logs with API errors
            return None
    
    @classmethod
    def get_first_move_name(cls, move: str) -> str:
        """
        Get opening name for first move (fallback).
        
        Args:
            move: First move in UCI format (e.g., 'e2e4')
            
        Returns:
            Opening name or generic "First Move"
        """
        return cls.FIRST_MOVE_NAMES.get(move, "First Move")


# ==================== EXPECTED POINTS CALCULATOR ====================

class ExpectedPointsCalculator:
    """
    Calculates expected points (winning probability) from evaluations.
    
    Expected points range from 0.0 (certain loss) to 1.0 (certain win).
    Uses sigmoid function for centipawn scores and mate distance for mate scores.
    """
    
    @staticmethod
    def from_centipawns(cp: int | float) -> float:
        """
        Convert centipawn evaluation to expected points.
        
        Formula: EP = 1 / (1 + e^(-0.0035 * cp))
        
        Args:
            cp: Centipawn evaluation (positive = advantage)
            
        Returns:
            Expected points (0.0 to 1.0)
            
        Example:
            >>> from_centipawns(0)      # Even position
            0.5
            >>> from_centipawns(100)    # Small advantage
            0.59
            >>> from_centipawns(300)    # Winning advantage
            0.74
        """
        return 1.0 / (1.0 + math.e ** (-EP_SIGMOID_CONSTANT * cp))
    
    @staticmethod
    def from_mate(mate: int | float) -> float:
        """
        Convert mate score to expected points with distance awareness.
        
        Closer mates are better than distant mates.
        Distant lost mates are better than immediate lost mates.
        
        Args:
            mate: Mate in N moves (positive = winning, negative = losing)
            
        Returns:
            Expected points (0.0 to 1.0)
            
        Example:
            >>> from_mate(1)      # Mate in 1 (winning)
            0.995
            >>> from_mate(10)     # Mate in 10 (winning)
            0.95
            >>> from_mate(-1)     # Mated in 1 (losing)
            0.005
            >>> from_mate(-10)    # Mated in 10 (losing)
            0.05
        """
        if mate > 0:
            # Winning mate: closer is better
            # M1 = 0.995, M2 = 0.99, M10 = 0.95
            distance = min(abs(mate), MAX_MATE_DISTANCE_CONSIDERED)
            return 1.0 - (MATE_DISTANCE_DECAY * distance)
        else:
            # Losing mate: farther is better
            # M-1 = 0.005, M-2 = 0.01, M-10 = 0.05
            distance = min(abs(mate), MAX_MATE_DISTANCE_CONSIDERED)
            return 0.0 + (MATE_DISTANCE_DECAY * distance)
    
    @classmethod
    def from_evaluation(cls, cp_value, mate_value) -> float:
        """
        Convert any evaluation to expected points.
        
        Args:
            cp_value: Centipawn score (int/float) or empty string
            mate_value: Mate score (int/float) or empty string
            
        Returns:
            Expected points (0.0 to 1.0)
        """
        # Prioritize mate score
        if isinstance(mate_value, (int, float)):
            return cls.from_mate(mate_value)
        
        # Use centipawn score
        if isinstance(cp_value, (int, float)):
            return cls.from_centipawns(cp_value)
        
        # Fallback: equal position
        return 0.5
    
    @staticmethod
    def calculate_loss(before_ep: float, after_ep: float) -> float:
        """
        Calculate expected points loss from mover's perspective.
        
        CRITICAL: After the move, it's opponent's turn, so we flip their EP.
        
        Args:
            before_ep: Expected points before move (mover's perspective)
            after_ep: Expected points after move (opponent's perspective)
            
        Returns:
            Expected points loss (clamped to 0.0 minimum)
            
        Example:
            Before: +50 cp → EP = 0.56 (White's turn)
            After:  -30 cp → EP = 0.41 (Black's perspective)
                           → EP = 0.59 (White's perspective after flip)
            Loss: 0.56 - 0.59 = -0.03 → clamped to 0.0 (improved!)
        """
        # Flip opponent's EP to get mover's EP after move
        loss = before_ep - (1.0 - after_ep)
        return max(0.0, loss)


# ==================== ACCURACY CALCULATOR ====================

class AccuracyCalculator:
    """
    Calculates move accuracy percentage from expected points loss.
    """
    
    @staticmethod
    def calculate(ep_loss: float) -> float:
        """
        Calculate move accuracy percentage.
        
        Formula: Accuracy = 103.16 * e^(-4 * ep_loss) - 3.17
        
        Args:
            ep_loss: Expected points loss (0.0 to 1.0)
            
        Returns:
            Accuracy percentage (0.0 to 100.0)
            
        Example:
            >>> calculate(0.0)     # Perfect move
            100.0
            >>> calculate(0.05)    # Small mistake
            84.3
            >>> calculate(0.35)    # Blunder
            23.8
        """
        accuracy = ACCURACY_MULTIPLIER * math.e ** (ACCURACY_EXPONENT * ep_loss) + ACCURACY_OFFSET
        return round(max(0.0, min(100.0, accuracy)), 2)


# ==================== MOVE CLASSIFIER ====================

class MoveClassifier:
    """
    Classifies moves into categories based on expected points loss and position analysis.
    
    Classification Priority (checked in order):
    1. Forced (only legal move)
    2. Book (opening theory)
    3. Checkmate (game-ending move)
    4. Base classification from EP loss thresholds
    5. Special upgrades (Brilliant, Great)
    """
    
    def __init__(self, board_analyzer: 'BoardAnalyzer', opening_book: OpeningBook):
        """
        Initialize classifier.
        
        Args:
            board_analyzer: Board analysis utilities
            opening_book: Opening detection system
        """
        self.board_analyzer = board_analyzer
        self.opening_book = opening_book
    
    def classify_from_ep_loss(self, ep_loss: float) -> str:
        """
        Get base classification from expected points loss.
        
        Args:
            ep_loss: Expected points loss (0.0 to 1.0)
            
        Returns:
            Move classification string
        """
        if ep_loss >= THRESHOLD_BLUNDER:
            return "blunder"
        elif ep_loss >= THRESHOLD_MISTAKE:
            return "mistake"
        elif ep_loss >= THRESHOLD_INACCURACY:
            return "inaccuracy"
        elif ep_loss >= THRESHOLD_MISS:
            return "miss"
        elif ep_loss >= THRESHOLD_GOOD:
            return "good"
        elif ep_loss >= THRESHOLD_EXCELLENT:
            return "excellent"
        else:
            return "best"
    
    def upgrade_to_great(
        self,
        ep_loss: float,
        last_move: str,
        last_pgn: str,
        top_moves: List[dict],
        was_in_check: bool,
        already_lost: bool
    ) -> bool:
        """
        Check if move should be upgraded to "Great".
        
        Great moves are near-perfect quiet moves in complex positions where
        the alternative moves are significantly worse.
        
        Args:
            ep_loss: Expected points loss
            last_move: UCI move (e.g., 'e2e4')
            last_pgn: Chess notation (e.g., 'Nf3')
            top_moves: Top engine moves with scores
            was_in_check: Whether player was in check
            already_lost: Whether position was already lost
            
        Returns:
            True if should be Great, False otherwise
        """
        # Must be near-perfect
        if ep_loss >= THRESHOLD_GREAT:
            return False
        
        # Must be engine's top choice
        if not top_moves or top_moves[0].get("Move") != last_move:
            return False
        
        # Must have significant gap to 2nd best (complex decision)
        if len(top_moves) > 1:
            top_score = self.board_analyzer._get_move_score(top_moves[0])
            second_score = self.board_analyzer._get_move_score(top_moves[1])
            gap = abs(top_score - second_score)
            if gap < BRILLIANT_GREAT_GAP_THRESHOLD:
                return False
        
        # Should be a quiet move (not capture, check, or checkmate)
        if "x" in last_pgn or "+" in last_pgn or "#" in last_pgn:
            return False
        
        # Not made while in check (defensive moves)
        if was_in_check:
            return False
        
        # Not in a lost position (desperation moves)
        if already_lost:
            return False
        
        return True
    
    def upgrade_to_brilliant(
        self,
        ep_loss: float,
        last_move: str,
        last_pgn: str,
        parsed_move: Tuple[Tuple[int, int], Tuple[int, int], str],
        top_moves: List[dict],
        before_board: List[List[str]],
        after_board: List[List[str]],
        mover_color: str,
        opponent_color: str,
        was_in_check: bool,
        already_lost: bool,
        cp_before
    ) -> bool:
        """
        Check if move should be upgraded to "Brilliant".
        
        Brilliant moves are spectacular moves involving calculated risk:
        - Sacrifices with compensation
        - Only moves giving advantage in lost positions
        - Equal trades with tactical pressure
        
        Args:
            ep_loss: Expected points loss
            last_move: UCI move
            last_pgn: Chess notation
            parsed_move: ((from_row, from_col), (to_row, to_col), promotion)
            top_moves: Top engine moves
            before_board: Board state before move
            after_board: Board state after move
            mover_color: Color that just moved
            opponent_color: Opponent color
            was_in_check: Whether in check
            already_lost: Whether position lost
            cp_before: Centipawn eval before move
            
        Returns:
            True if should be Brilliant, False otherwise
        """
        # Must be near-perfect (allow Best or Great as base)
        if ep_loss >= THRESHOLD_BRILLIANT:
            return False
        
        # Must be engine's top choice
        if not top_moves or top_moves[0].get("Move") != last_move:
            return False
        
        # Preliminary gates (reject if any fail)
        if not self._passes_brilliant_gates(
            last_move, last_pgn, was_in_check, already_lost
        ):
            return False
        
        # Try three brilliant paths
        from_square, to_square, promotion = parsed_move
        
        # Path 1: Brilliant sacrifice
        if self._is_brilliant_sacrifice(
            last_move, last_pgn, from_square, to_square,
            before_board, after_board, mover_color, opponent_color
        ):
            return True
        
        # Path 2: Only advantage move in losing position
        if self._is_brilliant_only_advantage(
            last_move, top_moves, already_lost, cp_before
        ):
            return True
        
        # Path 3: Brilliant equal trade with pressure
        if self._is_brilliant_equal_trade(
            last_move, last_pgn, to_square,
            before_board, after_board, mover_color, opponent_color
        ):
            return True
        
        return False
    
    def _passes_brilliant_gates(
        self,
        last_move: str,
        last_pgn: str,
        was_in_check: bool,
        already_lost: bool
    ) -> bool:
        """Check if move passes preliminary brilliant gates"""
        # Not a queen promotion (too obvious)
        if len(last_move) == 5 and last_move[-1].lower() == 'q':
            return False
        
        # Not made while in check (defensive forced moves)
        if was_in_check:
            return False
        
        # Not in already lost position (less than -300 cp)
        if already_lost:
            return False
        
        return True
    
    def _is_brilliant_sacrifice(
        self,
        last_move: str,
        last_pgn: str,
        from_square: Tuple[int, int],
        to_square: Tuple[int, int],
        before_board: List[List[str]],
        after_board: List[List[str]],
        mover_color: str,
        opponent_color: str
    ) -> bool:
        """Check if move is a brilliant sacrifice"""
        # Detect sacrifice
        moved_piece = after_board[to_square[0]][to_square[1]]
        if moved_piece == ".":
            moved_piece = before_board[from_square[0]][from_square[1]]
        
        moved_value = self.board_analyzer._piece_value(moved_piece)
        captured_piece = before_board[to_square[0]][to_square[1]]
        
        # Check if moved to attacked square
        attackers_after = self.board_analyzer._attackers_to_square(
            after_board, to_square, opponent_color
        )
        moved_to_attacked = len(attackers_after) > 0
        attacked_by_lower = any(
            self.board_analyzer._piece_value(piece) < moved_value
            for piece in attackers_after
        )
        
        if not (moved_to_attacked and attacked_by_lower):
            return False
        
        # Not a simple recapture
        if "x" in last_pgn and len(database.game_history) > 1:
            prev_move = database.game_history[-2]
            if prev_move[2:4] == last_move[2:4]:  # Same target square
                return False
        
        # Not queen taking undefended piece
        defenders_before = self.board_analyzer._attackers_to_square(
            before_board, to_square, opponent_color
        )
        if moved_piece.lower() == 'q' and len(defenders_before) == 0:
            return False
        
        # Check compensation
        check_given = "+" in last_pgn or "#" in last_pgn
        pinned_opponent = any(
            self.board_analyzer._piece_color(str(piece_id)) == opponent_color
            for piece_id in database.pins
        )
        
        opponent_king_square = self.board_analyzer._find_king(after_board, opponent_color)
        king_pressure = 0
        if opponent_king_square:
            king_attackers = self.board_analyzer._attackers_to_square(
                after_board, opponent_king_square, mover_color
            )
            king_pressure = len(king_attackers)
        
        compensation = (
            check_given or 
            pinned_opponent or 
            king_pressure >= BRILLIANT_SACRIFICE_KING_PRESSURE
        )
        
        if not compensation:
            return False
        
        # Check not simplification (escaping danger without creating threats)
        from_attackers = self.board_analyzer._attackers_to_square(
            before_board, from_square, opponent_color
        )
        removes_from_danger = len(from_attackers) > 0 and not moved_to_attacked
        creates_threats = check_given or pinned_opponent or king_pressure > 0
        
        if removes_from_danger and not creates_threats:
            return False
        
        return True
    
    def _is_brilliant_only_advantage(
        self,
        last_move: str,
        top_moves: List[dict],
        already_lost: bool,
        cp_before
    ) -> bool:
        """Check if move is only move giving advantage in losing position"""
        # Must be in losing position (but not completely lost)
        if not isinstance(cp_before, (int, float)):
            return False
        
        if cp_before >= -300:
            return False
        
        # Must be best move
        if not top_moves or top_moves[0].get("Move") != last_move:
            return False
        
        # Best move must give positive evaluation
        best_score = self.board_analyzer._get_move_score(top_moves[0])
        if best_score <= -80:
            return False
        
        # Check if 2nd best is significantly worse
        if len(top_moves) == 1:
            return True
        
        second_score = self.board_analyzer._get_move_score(top_moves[1])
        gap = best_score - second_score
        
        return gap >= 120
    
    def _is_brilliant_equal_trade(
        self,
        last_move: str,
        last_pgn: str,
        to_square: Tuple[int, int],
        before_board: List[List[str]],
        after_board: List[List[str]],
        mover_color: str,
        opponent_color: str
    ) -> bool:
        """Check if move is brilliant equal trade with pressure"""
        # Must be a capture
        if "x" not in last_pgn:
            return False
        
        captured_piece = before_board[to_square[0]][to_square[1]]
        if captured_piece == ".":
            return False
        
        moved_piece = after_board[to_square[0]][to_square[1]]
        moved_value = self.board_analyzer._piece_value(moved_piece)
        captured_value = self.board_analyzer._piece_value(captured_piece)
        
        # Must be equal or favorable trade
        if captured_value < moved_value:
            return False
        
        # Target must have been defended
        defenders = self.board_analyzer._attackers_to_square(
            before_board, to_square, opponent_color
        )
        if len(defenders) == 0:
            return False
        
        # Must be attacked by lower-value piece
        attackers_after = self.board_analyzer._attackers_to_square(
            after_board, to_square, opponent_color
        )
        attacked_by_lower = any(
            self.board_analyzer._piece_value(piece) < moved_value
            for piece in attackers_after
        )
        
        if not attacked_by_lower:
            return False
        
        # Check compensation (king pressure, pins, checks)
        check_given = "+" in last_pgn or "#" in last_pgn
        pinned_opponent = any(
            self.board_analyzer._piece_color(str(piece_id)) == opponent_color
            for piece_id in database.pins
        )
        
        opponent_king_square = self.board_analyzer._find_king(after_board, opponent_color)
        king_pressure = 0
        if opponent_king_square:
            king_attackers = self.board_analyzer._attackers_to_square(
                after_board, opponent_king_square, mover_color
            )
            king_pressure = len(king_attackers)
        
        compensation = (
            check_given or 
            pinned_opponent or 
            king_pressure >= BRILLIANT_SACRIFICE_KING_PRESSURE
        )
        
        return compensation


# ==================== BOARD ANALYZER ====================

class BoardAnalyzer:
    """
    Analyzes board positions for tactical features.
    
    Used by MoveClassifier to detect sacrifices, king pressure, and piece safety.
    """
    
    @staticmethod
    def _uci_to_square(square: str) -> Tuple[int, int]:
        """Convert UCI square (e.g., 'e4') to matrix coordinates"""
        return 8 - int(square[1]), ord(square[0]) - ord("a")
    
    @staticmethod
    def _piece_value(piece: str) -> int:
        """Get material value of a piece"""
        values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
        if not piece or piece == ".":
            return 0
        return values.get(piece.lower(), 0)
    
    @staticmethod
    def _piece_color(piece: str) -> Optional[str]:
        """Get color of a piece"""
        if not piece or piece == ".":
            return None
        return "white" if piece.isupper() else "black"
    
    @staticmethod
    def _inside(row: int, col: int) -> bool:
        """Check if square is inside board"""
        return 0 <= row <= 7 and 0 <= col <= 7
    
    @staticmethod
    def _matrix_piece_to_fen(piece) -> str:
        """Convert matrix piece to FEN character"""
        if piece == 0 or piece == "":
            return "."
        piece_str = str(piece)
        base = piece_str.lstrip("-")[0].lower()
        return base.upper() if "-" not in piece_str else base
    
    @classmethod
    def _board_from_matrix(cls) -> List[List[str]]:
        """Convert database matrix to FEN-style board"""
        board: List[List[str]] = []
        for row in database.matrix:
            board.append([cls._matrix_piece_to_fen(piece) for piece in row])
        return board
    
    @staticmethod
    def _board_from_fen(fen: str) -> List[List[str]]:
        """Parse FEN string to board array"""
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
    
    @classmethod
    def _find_king(cls, board: List[List[str]], color: str) -> Optional[Tuple[int, int]]:
        """Find king position on board"""
        king = "K" if color == "white" else "k"
        for r in range(8):
            for c in range(8):
                if board[r][c] == king:
                    return (r, c)
        return None
    
    @classmethod
    def _attackers_to_square(
        cls,
        board: List[List[str]],
        square: Tuple[int, int],
        attacker_color: str
    ) -> List[str]:
        """
        Find all pieces of a color that attack a square.
        
        Args:
            board: Board state
            square: Target square (row, col)
            attacker_color: Color of attacking pieces
            
        Returns:
            List of piece characters attacking the square
        """
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
            if cls._inside(r, c) and board[r][c] == pawn_char:
                attackers.append(board[r][c])
        
        # Knight attacks
        knight_offsets = [
            (1, 2), (1, -2), (-1, 2), (-1, -2),
            (2, 1), (2, -1), (-2, 1), (-2, -1)
        ]
        knight_char = "N" if attacker_color == "white" else "n"
        for dr, dc in knight_offsets:
            r, c = row + dr, col + dc
            if cls._inside(r, c) and board[r][c] == knight_char:
                attackers.append(board[r][c])
        
        # Bishop/queen diagonals
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            r, c = row + dr, col + dc
            while cls._inside(r, c):
                piece = board[r][c]
                if piece != ".":
                    if cls._piece_color(piece) == attacker_color and piece.lower() in ("b", "q"):
                        attackers.append(piece)
                    break
                r += dr
                c += dc
        
        # Rook/queen straights
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            r, c = row + dr, col + dc
            while cls._inside(r, c):
                piece = board[r][c]
                if piece != ".":
                    if cls._piece_color(piece) == attacker_color and piece.lower() in ("r", "q"):
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
                if cls._inside(r, c) and board[r][c] == king_char:
                    attackers.append(board[r][c])
        
        return attackers
    
    @staticmethod
    def _get_move_score(move_info: dict) -> int:
        """
        Get numerical score from Stockfish move info.
        
        Args:
            move_info: Stockfish move dictionary
            
        Returns:
            Score in centipawns (mate converted to large values)
        """
        mate = move_info.get("Mate")
        cp = move_info.get("Centipawn")
        
        if isinstance(mate, int):
            # Convert mate to centipawn equivalent
            return 100000 - abs(mate) if mate > 0 else -100000 + abs(mate)
        
        if isinstance(cp, int):
            return cp
        
        return -99999


# ==================== MAIN REVIEW SYSTEM ====================

class ReviewSystem:
    """
    Main chess game review system.
    
    Evaluates moves asynchronously, classifies them, and stores evaluation history.
    
    Usage:
        review = ReviewSystem()
        review.starting_evaluation()  # Evaluate initial position
        
        # After each move:
        review.evaluate_last_move_async()
        
        # Shutdown when done:
        review.shutdown()
    
    Thread Safety:
        All evaluation happens in a background thread pool.
        UI remains responsive during analysis.
    """
    
    def __init__(self):
        """Initialize review system with all components"""
        # Stockfish engine
        self.stockfish = self._initialize_stockfish()
        
        # Current evaluation storage
        self.curr_eval: Dict[str, int | str | float] = {}
        
        # Threading for async evaluation
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-eval")
        self._lock = Lock()
        self._pending_future: Optional[Future] = None
        self._evaluation_requested: bool = False
        
        # Sync tracking
        self._synced_history_len: int = 0
        
        # Utility components
        self.notation_gen = NotationGenerator()
        self.opening_book = OpeningBook()
        self.ep_calc = ExpectedPointsCalculator()
        self.accuracy_calc = AccuracyCalculator()
        self.board_analyzer = BoardAnalyzer()
        self.move_classifier = MoveClassifier(self.board_analyzer, self.opening_book)
        
        database.gamelogger.init("Review system initialized")
    
    # ==================== STOCKFISH MANAGEMENT ====================
    
    def _initialize_stockfish(self) -> Stockfish:
        """Initialize Stockfish engine"""
        ai = AIUtilities()
        path = ai.resource_path("stockfish/stockfish-windows-x86-64-avx2.exe")
        return Stockfish(path)
    
    def _reset_review_engine(self) -> None:
        """Reset Stockfish to starting position"""
        self.stockfish.set_fen_position(STARTING_FEN)
        self._synced_history_len = 0
    
    def _sync_review_engine_to_history(self) -> None:
        """
        Synchronize Stockfish with game history.
        
        Always rebuilds from scratch to avoid drift.
        """
        history = list(database.game_history)
        self._reset_review_engine()
        
        if history:
            self.stockfish.make_moves_from_current_position(history)
        
        self._synced_history_len = len(history)
    
    # ==================== EVALUATION METHODS ====================
    
    def starting_evaluation(self) -> None:
        """
        Evaluate the starting position.
        
        Should be called once at game start.
        """
        self._reset_review_engine()
        
        evaluation = self.stockfish.get_evaluation()
        
        if evaluation["type"] == "cp":
            cp = evaluation["value"]
            mate = ""
        else:
            cp = ""
            mate = evaluation["value"]
        
        eval_data = EvaluationData(
            color="white",
            cp=cp,
            mate=mate,
            move_type="",
            accuracy=""
        )
        
        self.curr_eval = eval_data.to_dict()
        self._update_evaluation_history()
        
        database.gamelogger.init(f"Starting position: CP={cp}, Mate={mate}")
    
    def evaluate_last_move(self) -> None:
        """
        Evaluate the most recent move.
        
        This is the core evaluation logic that:
        1. Checks for forced/book/checkmate
        2. Calculates expected points loss
        3. Classifies the move
        4. Calculates accuracy
        """
        # Handle starting position
        if not database.game_history:
            if not database.evaluation_history:
                self.starting_evaluation()
            return
        
        # Sync Stockfish
        self._sync_review_engine_to_history()
        
        # Get current evaluation
        cp, mate = self._get_current_evaluation()
        
        # Determine who moved
        color = "white" if database.current_turn == "black" else "black"
        
        # Get move context
        move_count = len(database.game_history)
        last_move = database.game_history[-1] if database.game_history else ""
        last_pgn = database.game_pgn[-1] if database.game_pgn else ""
        
        # Generate FEN for opening lookup
        fen = self.notation_gen.generate_fen(
            board=database.matrix,
            side_to_move=database.current_turn[0],  # type: ignore
            fullmove_number=database.fullmove
        )
        
        # ===== CLASSIFICATION PRIORITY =====
        
        # Priority 1: FORCED (only legal move)
        if database.last_forced:
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type="forced",
                accuracy=""
            )
            self.curr_eval = eval_data.to_dict()
            self._update_evaluation_history()
            return
        
        # Priority 2: BOOK (opening theory)
        book_result = self._check_book_move(move_count, last_move, last_pgn, color, fen, cp, mate)
        if book_result:
            self.curr_eval = book_result
            self._update_evaluation_history()
            return
        
        # Priority 3: CHECKMATE (game-ending move)
        if len(database.get_legal_moves(database.current_turn)) == 0:
            # Determine if forced checkmate
            move_type = "forced" if database.last_forced else "best"
            
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type=move_type,
                accuracy=100.0
            )
            self.curr_eval = eval_data.to_dict()
            self._update_evaluation_history()
            return
        
        # Priority 4 & 5: EVALUATION-BASED + SPECIAL UPGRADES
        self._evaluate_with_engine(color, cp, mate, last_move, last_pgn)
    
    def _check_book_move(
        self,
        move_count: int,
        last_move: str,
        last_pgn: str,
        color: str,
        fen: str,
        cp,
        mate
    ) -> Optional[Dict]:
        """
        Check if move is in opening book.
        
        Returns:
            Evaluation dict if book move, None otherwise
        """
        # First move is always book (with fallback name)
        if move_count == 1:
            opening = self.opening_book.get_opening_from_api(fen)
            if not opening:
                opening = self.opening_book.get_first_move_name(last_move)
            
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type="book",
                accuracy="",
                opening=opening
            )
            return eval_data.to_dict()
        
        # Check API until move 14
        if move_count <= BOOK_MAX_MOVE:
            opening = self.opening_book.get_opening_from_api(fen)
            if opening:
                eval_data = EvaluationData(
                    color=color,
                    cp=cp,
                    mate=mate,
                    move_type="book",
                    accuracy="",
                    opening=opening
                )
                return eval_data.to_dict()
        
        # Fallback heuristic: quiet positions until move 12
        if move_count <= BOOK_HEURISTIC_MAX_MOVE:
            # Must be quiet (no captures)
            if "x" not in last_pgn:
                # Must be balanced position
                if isinstance(cp, (int, float)) and abs(cp) <= BOOK_HEURISTIC_MAX_CP:
                    eval_data = EvaluationData(
                        color=color,
                        cp=cp,
                        mate=mate,
                        move_type="book",
                        accuracy="",
                        opening="Opening"
                    )
                    return eval_data.to_dict()
        
        return None
    
    def _evaluate_with_engine(
        self,
        color: str,
        cp,
        mate,
        last_move: str,
        last_pgn: str
    ) -> None:
        """
        Evaluate move using engine comparison.
        
        Calculates EP loss, base classification, and checks for upgrades.
        """
        # Get previous evaluation
        if not database.evaluation_history:
            self.starting_evaluation()
        
        last_eval = database.evaluation_history[-1]
        
        # Extract evaluations
        cp_before = last_eval["cp"]
        mate_before = last_eval["mate"]
        cp_after = cp
        mate_after = mate
        
        # Calculate expected points
        ep_before = self.ep_calc.from_evaluation(cp_before, mate_before)
        ep_after = self.ep_calc.from_evaluation(cp_after, mate_after)
        
        # Calculate loss
        ep_loss = self.ep_calc.calculate_loss(ep_before, ep_after)
        
        # Calculate accuracy
        accuracy = self.accuracy_calc.calculate(ep_loss)
        
        # Get base classification
        move_type = self.move_classifier.classify_from_ep_loss(ep_loss)
        
        # Check for upgrades (Brilliant, Great)
        # Only upgrade if base class is good enough
        if ep_loss < THRESHOLD_EXCELLENT:
            move_type = self._check_special_upgrades(
                move_type, ep_loss, last_move, last_pgn,
                color, cp_before, mate_before
            )
        
        # Store result
        eval_data = EvaluationData(
            color=color,
            cp=cp_after,
            mate=mate_after,
            move_type=move_type,
            accuracy=accuracy
        )
        
        self.curr_eval = eval_data.to_dict()
        self._update_evaluation_history()
    
    def _check_special_upgrades(
        self,
        base_type: str,
        ep_loss: float,
        last_move: str,
        last_pgn: str,
        color: str,
        cp_before,
        mate_before
    ) -> str:
        """
        Check if move should be upgraded to Brilliant or Great.
        
        Args:
            base_type: Current classification
            ep_loss: Expected points loss
            last_move: UCI move
            last_pgn: Chess notation
            color: Player color
            cp_before: CP before move
            mate_before: Mate before move
            
        Returns:
            Upgraded classification or base_type
        """
        # Rebuild board state before move
        before_board, top_moves = self._rebuild_before_position_data()
        
        if not before_board or not top_moves:
            return base_type
        
        # Get board after move
        after_board = self.board_analyzer._board_from_matrix()
        
        # Parse move
        parsed_move = self._parse_uci_move(last_move)
        if not parsed_move:
            return base_type
        
        # Get colors
        mover_color = color
        opponent_color = "black" if color == "white" else "white"
        
        # Check if in check before move
        was_in_check = self._was_in_check_before(before_board, mover_color, opponent_color)
        
        # Check if position was lost
        already_lost = self._position_was_lost(cp_before, mate_before)
        
        # Check Brilliant upgrade
        if self.move_classifier.upgrade_to_brilliant(
            ep_loss, last_move, last_pgn, parsed_move, top_moves,
            before_board, after_board, mover_color, opponent_color,
            was_in_check, already_lost, cp_before
        ):
            return "brilliant"
        
        # Check Great upgrade (only if not Brilliant)
        if self.move_classifier.upgrade_to_great(
            ep_loss, last_move, last_pgn, top_moves,
            was_in_check, already_lost
        ):
            return "great"
        
        return base_type
    
    # ==================== HELPER METHODS ====================
    
    def _get_current_evaluation(self) -> Tuple[int | float | str, int | float | str]:
        """
        Get current position evaluation from Stockfish.
        
        Returns:
            (cp, mate) tuple
        """
        evaluation = self.stockfish.get_evaluation()
        
        if evaluation["type"] == "cp":
            cp = evaluation["value"]
            mate = ""
        else:
            cp = ""
            mate = evaluation["value"]
        
        return cp, mate
    
    def _parse_uci_move(self, move: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Parse UCI move to from/to squares and promotion"""
        if len(move) < 4:
            return None
        
        from_square = self.board_analyzer._uci_to_square(move[:2])
        to_square = self.board_analyzer._uci_to_square(move[2:4])
        promotion = move[4].lower() if len(move) >= 5 else ""
        
        return from_square, to_square, promotion
    
    def _was_in_check_before(
        self,
        before_board: List[List[str]],
        mover_color: str,
        opponent_color: str
    ) -> bool:
        """Check if king was in check before the move"""
        king_square = self.board_analyzer._find_king(before_board, mover_color)
        if not king_square:
            # Fallback: check previous PGN for check symbol
            if len(database.game_pgn) > 1:
                return "+" in database.game_pgn[-2]
            return False
        
        attackers = self.board_analyzer._attackers_to_square(
            before_board, king_square, opponent_color
        )
        return len(attackers) > 0
    
    def _position_was_lost(self, cp_before, mate_before) -> bool:
        """Check if position was already lost before move"""
        # Check centipawn evaluation
        if isinstance(cp_before, (int, float)):
            if cp_before < -300:
                return True
        
        # Check mate evaluation
        if isinstance(mate_before, (int, float)):
            if mate_before < 0:
                return True
        
        return False
    
    def _rebuild_before_position_data(self) -> Tuple[Optional[List[List[str]]], List[dict]]:
        """
        Rebuild board state before last move and get engine analysis.
        
        Returns:
            (before_board, top_moves) tuple
        """
        if not database.game_history:
            return None, []
        
        history_snapshot = list(database.game_history)
        
        try:
            # Reset and replay all moves except last
            self.stockfish.set_fen_position(STARTING_FEN)
            if len(history_snapshot) > 1:
                self.stockfish.make_moves_from_current_position(history_snapshot[:-1])
            
            # Get FEN and top moves
            fen_before = self.stockfish.get_fen_position()
            top_moves = self.stockfish.get_top_moves(3) or []
            
            # Parse board from FEN
            before_board = self.board_analyzer._board_from_fen(fen_before)
            
            return before_board, top_moves
        
        except Exception as e:
            database.gamelogger.error(f"Failed to rebuild position: {e}")
            return None, []
        
        finally:
            # Re-sync to current position
            try:
                self._sync_review_engine_to_history()
            except Exception:
                pass
    
    def _update_evaluation_history(self) -> None:
        """
        Update database evaluation history with current evaluation.
        
        Avoids duplicates.
        """
        db_eval = database.evaluation_history
        curr_eval = dict(self.curr_eval)
        
        # Initialize history if empty
        if not db_eval:
            db_eval.append(curr_eval)
            return
        
        # Don't add duplicates
        if curr_eval == db_eval[-1]:
            return
        
        db_eval.append(curr_eval)
    
    # ==================== ASYNC EVALUATION ====================
    
    def evaluate_last_move_async(self) -> None:
        """
        Queue non-blocking move evaluation.
        
        Coalesces multiple rapid requests into single evaluation.
        Thread-safe.
        """
        with self._lock:
            self._evaluation_requested = True
            
            # Start worker if not already running
            if self._pending_future is None or self._pending_future.done():
                self._pending_future = self._executor.submit(self._drain_evaluation_queue)
    
    def _drain_evaluation_queue(self) -> None:
        """
        Worker thread that drains evaluation queue.
        
        Continues evaluating while requests are pending.
        """
        while True:
            with self._lock:
                if not self._evaluation_requested:
                    self._pending_future = None
                    return
                self._evaluation_requested = False
            
            # Evaluate outside lock
            self._evaluate_guarded()
    
    def _evaluate_guarded(self) -> None:
        """
        Evaluate with exception handling.
        
        Logs errors but doesn't crash.
        """
        try:
            self.evaluate_last_move()
            
            # Log result
            if database.evaluation_history:
                latest = database.evaluation_history[-1]
                database.gamelogger.move(f"Review: {latest}")
        
        except Exception as e:
            database.gamelogger.error(f"Review evaluation failed: {e}")
    
    def shutdown(self) -> None:
        """
        Shutdown review system.
        
        Cancels pending evaluations and stops thread pool.
        Should be called before closing game.
        """
        with self._lock:
            self._evaluation_requested = False
            self._executor.shutdown(wait=False, cancel_futures=True)
        
        database.gamelogger.game("Review system shutdown")


# ==================== MODULE INITIALIZATION ====================

# Create shared utilities
ai = AIUtilities()
notation = NotationGenerator()


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    """
    Example usage of ReviewSystem.
    
    This demonstrates the typical workflow:
    1. Create system
    2. Evaluate starting position
    3. Evaluate each move after it's played
    4. Shutdown when done
    """
    
    # Initialize
    review = ReviewSystem()
    
    # Evaluate starting position
    review.starting_evaluation()
    print("Starting evaluation:", database.evaluation_history[0])
    
    # Simulate a game (you would actually play moves)
    # After each move:
    # review.evaluate_last_move_async()
    
    # Shutdown
    review.shutdown()
    
    print("Review system example complete")