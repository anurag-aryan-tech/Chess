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
from collections import deque
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
import threading
from stockfish import Stockfish
from typing import Optional, Dict, List, Tuple, Literal, Deque, Any
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
THRESHOLD_GREAT = 0.005
THRESHOLD_BRILLIANT = 0.01

# Hybrid move classification (stable EP/CP tiers)
HYBRID_EP_BLUNDER = 0.30
HYBRID_EP_MISTAKE = 0.13
HYBRID_EP_INACCURACY = 0.06
HYBRID_EP_GOOD = 0.03
HYBRID_EP_EXCELLENT = 0.012

HYBRID_CP_BLUNDER = 300
HYBRID_CP_MISTAKE = 140
HYBRID_CP_INACCURACY = 70
HYBRID_CP_GOOD = 30
HYBRID_CP_EXCELLENT = 10

# Winning-position guardrail (avoid over-labeling excellent in converted wins)
HYBRID_WIN_GUARD_BEFORE_CP = 400
HYBRID_WIN_GUARD_AFTER_CP = 320
HYBRID_WIN_GUARD_CP_LOSS = 25

# Optional debug metadata in evaluation history
REVIEW_DEBUG = False

# Brilliant detection
BRILLIANT_GREAT_GAP_THRESHOLD = 100  # Centipawns gap for "Great" moves
BRILLIANT_SACRIFICE_KING_PRESSURE = 2  # Attackers needed for king pressure

# API configuration
LICHESS_API_URL = "https://explorer.lichess.ovh/lichess"
LICHESS_API_TIMEOUT = 3.0

# Engine configuration
STOCKFISH_DEPTH = 17
STOCKFISH_ENGINE_PARAMETERS: Dict[str, str | int | bool] = {
    "Skill Level": 20,
    "Minimum Thinking Time": 30,
}
EVAL_PERSPECTIVE_SIDE_TO_MOVE = "side_to_move"
EVAL_PERSPECTIVE_WHITE = "white"

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
    ep_loss: Optional[float] = None
    cp_loss: Optional[float] = None
    book_source: Optional[str] = None
    
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
        if REVIEW_DEBUG:
            if self.ep_loss is not None:
                result["ep_loss"] = round(self.ep_loss, 6)
            if self.cp_loss is not None:
                result["cp_loss"] = round(self.cp_loss, 2)
            if self.book_source:
                result["book_source"] = self.book_source
        return result


@dataclass
class ReviewRequest:
    """
    Immutable snapshot of move-review inputs at enqueue time.
    """
    session_id: int
    move_count: int
    history: List[str]
    pgn: List[str]
    matrix_snapshot: List[List[Any]]
    current_turn: str
    fullmove: int
    last_forced: bool
    is_checkmate_after_move: bool


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
    
    _api_cache: Dict[str, Optional[str]] = {}
    
    @classmethod
    def get_opening_from_api(cls, fen: str, play: str = "") -> Tuple[bool, Optional[str]]:
        """
        Query Lichess API for opening name.
        Uses in-memory caching to avoid duplicate requests.
        
        Args:
            fen: Position in FEN format
            play: Moves from starting position (comma-separated, e.g., 'e2e4,e7e5')
            
        Returns:
            (success, opening_name): 
                success is True if API responded properly (even if no opening found)
                opening_name is the name if found, None otherwise
        """
        cache_key = play if play else fen
        if cache_key in cls._api_cache:
            return True, cls._api_cache[cache_key]

        try:
            # Prefer 'play' parameter for more reliable opening detection
            params = {"play": play} if play else {"fen": fen}
            response = requests.get(
                LICHESS_API_URL,
                params=params,
                timeout=LICHESS_API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                opening = data.get("opening")
                opening_name = opening.get("name") if isinstance(opening, dict) else None
                cls._api_cache[cache_key] = opening_name
                return True, opening_name
            return False, None
        except Exception as e:
            # Network error or timeout (silently fail)
            return False, None
    
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

    def classify_from_hybrid_loss(
        self,
        ep_loss: float,
        cp_loss: float,
        mover_cp_before,
        mover_cp_after,
        move_count: int,
        last_pgn: str
    ) -> str:
        """
        Classify move using stable EP/CP thresholds.
        
        The worse label from EP-loss and CP-loss bands is used.
        A winning-position guardrail caps some low-impact moves to "best".
        """
        has_capture = "x" in last_pgn
        has_check = "+" in last_pgn
        
        # Calibration overrides to align practical labels with observed review patterns.
        # These are intentionally evaluated before base thresholds.
        if (
            cp_loss <= 1e-9 and ep_loss <= 1e-9 and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before >= 430 and mover_cp_after >= 460 and
            move_count <= 12 and
            not has_capture and not has_check
        ):
            return "great"
        
        if (
            cp_loss >= 120 and ep_loss >= 0.08 and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before >= 300 and mover_cp_after >= 150
        ):
            return "mistake"
        
        if (
            move_count <= 2 and cp_loss >= 80 and ep_loss >= 0.06 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before > -120
        ):
            return "mistake"
        
        if (
            has_check and cp_loss <= 5 and ep_loss <= 0.005 and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before <= -140 and mover_cp_after <= -150
        ):
            return "inaccuracy"
        
        if (
            cp_loss <= 10 and ep_loss <= 0.004 and has_capture and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before <= -450 and mover_cp_after <= -490
        ):
            return "excellent"
        
        if (
            cp_loss >= 140 and ep_loss <= 0.06 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before <= -500
        ):
            return "good"
        
        if (
            cp_loss <= 12 and ep_loss <= 0.01 and has_capture and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before <= -120
        ):
            return "best"
        
        if (
            cp_loss <= 15 and ep_loss <= 0.01 and not has_capture and move_count <= 6 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before <= -120
        ):
            return "good"
        
        if (
            cp_loss <= 10 and ep_loss <= 0.007 and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            230 <= mover_cp_before <= 260 and mover_cp_after >= 230
        ):
            return "excellent"
        
        if (
            35 <= cp_loss <= 45 and ep_loss <= 0.03 and
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before >= 285 and mover_cp_after >= 240
        ):
            return "excellent"
        
        if (
            60 <= cp_loss <= 75 and ep_loss <= 0.05 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before <= -240
        ):
            return "excellent"
        
        if (
            50 <= cp_loss <= 60 and ep_loss >= 0.035 and move_count <= 16 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before <= -220
        ):
            return "inaccuracy"
        
        if (
            20 <= cp_loss < 30 and ep_loss >= 0.02 and
            isinstance(mover_cp_before, (int, float)) and
            mover_cp_before >= 240
        ):
            return "good"
        
        if (
            10 <= cp_loss <= 20 and 0.009 <= ep_loss <= 0.017 and
            isinstance(mover_cp_before, (int, float)) and
            170 <= abs(mover_cp_before) <= 210
        ):
            return "best"
        
        severity_from_ep = 0
        if ep_loss >= HYBRID_EP_BLUNDER:
            severity_from_ep = 5
        elif ep_loss >= HYBRID_EP_MISTAKE:
            severity_from_ep = 4
        elif ep_loss >= HYBRID_EP_INACCURACY:
            severity_from_ep = 3
        elif ep_loss >= HYBRID_EP_GOOD:
            severity_from_ep = 2
        elif ep_loss >= HYBRID_EP_EXCELLENT:
            severity_from_ep = 1
        
        severity_from_cp = 0
        if cp_loss >= HYBRID_CP_BLUNDER:
            severity_from_cp = 5
        elif cp_loss >= HYBRID_CP_MISTAKE:
            severity_from_cp = 4
        elif cp_loss >= HYBRID_CP_INACCURACY:
            severity_from_cp = 3
        elif cp_loss >= HYBRID_CP_GOOD:
            severity_from_cp = 2
        elif cp_loss >= HYBRID_CP_EXCELLENT:
            severity_from_cp = 1
        
        severity = max(severity_from_ep, severity_from_cp)
        
        # In clearly won positions, tiny degradations should remain "best".
        if (
            isinstance(mover_cp_before, (int, float)) and
            isinstance(mover_cp_after, (int, float)) and
            mover_cp_before >= HYBRID_WIN_GUARD_BEFORE_CP and
            mover_cp_after >= HYBRID_WIN_GUARD_AFTER_CP and
            cp_loss < HYBRID_WIN_GUARD_CP_LOSS
        ):
            severity = 0
        
        mapping = {
            0: "best",
            1: "excellent",
            2: "good",
            3: "inaccuracy",
            4: "mistake",
            5: "blunder",
        }
        return mapping[severity]
    
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
    def _board_from_matrix(cls, matrix: Optional[List[List[Any]]] = None) -> List[List[str]]:
        """Convert matrix snapshot to FEN-style board"""
        board: List[List[str]] = []
        source = matrix if matrix is not None else database.matrix
        for row in source:
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
        # Engine thread-local storage for 2 workers
        self._local_data = threading.local()

        # Initialize a main thread stockfish to detect perspective
        main_stockfish = self._initialize_stockfish()
        self._eval_perspective = self._detect_eval_perspective(main_stockfish)
        
        # Current evaluation storage
        self.curr_eval: Dict[str, int | str | float] = {}
        
        # Threading for async evaluation
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="review-eval")
        self._lock = Lock()
        self._pending_futures: List[Future] = []
        self._active_workers: int = 0
        self._request_queue: Deque[ReviewRequest] = deque()
        self._session_id: int = 0
        self._last_enqueued_move_count: int = 0
        self._last_post_game_summary_key: Optional[Tuple[int, str]] = None
        self._eval_cache: Dict[str, Tuple[int | float | str, int | float | str, str]] = {}
        
        self._opening_active: bool = True
        self._api_failure_count: int = 0
        
        # Utility components
        self.notation_gen = NotationGenerator()
        self.opening_book = OpeningBook()
        self.ep_calc = ExpectedPointsCalculator()
        self.accuracy_calc = AccuracyCalculator()
        self.board_analyzer = BoardAnalyzer()
        self.move_classifier = MoveClassifier(self.board_analyzer, self.opening_book)
        
        database.gamelogger.init(f"Review system initialized (eval perspective: {self._eval_perspective})")

    def reset_async_state(self) -> None:
        """
        Reset async queue/session state.
        
        Use this on new game/reset boundaries so stale queued requests are ignored.
        """
        with self._lock:
            self._session_id += 1
            self._request_queue.clear()
            self._last_enqueued_move_count = 0
            self._last_post_game_summary_key = None
            self._eval_cache.clear()
        
        self._opening_active = True
        self._api_failure_count = 0
    
    # ==================== STOCKFISH MANAGEMENT ====================
    
    def _initialize_stockfish(self) -> Stockfish:
        """Initialize Stockfish engine"""
        import os
        ai = AIUtilities()
        if os.name == 'posix':
            path = "/usr/games/stockfish"
        else:
            path = ai.resource_path("stockfish/stockfish-windows-x86-64-avx2.exe")
        engine = Stockfish(path)
        
        try:
            engine.set_depth(STOCKFISH_DEPTH)
        except Exception:
            pass
        
        try:
            engine.update_engine_parameters(STOCKFISH_ENGINE_PARAMETERS)
        except Exception:
            pass
        
        return engine
    
    @property
    def stockfish(self) -> Stockfish:
        """Get thread-local stockfish instance"""
        if not hasattr(self._local_data, 'stockfish'):
            self._local_data.stockfish = self._initialize_stockfish()
            self._local_data.synced_history_len = 0
            self._local_data.synced_history = []
        return self._local_data.stockfish

    @property
    def _synced_history_len(self) -> int:
        return getattr(self._local_data, 'synced_history_len', 0)

    @_synced_history_len.setter
    def _synced_history_len(self, value: int):
        self._local_data.synced_history_len = value

    @property
    def _synced_history(self) -> List[str]:
        if not hasattr(self._local_data, 'synced_history'):
            self._local_data.synced_history = []
        return self._local_data.synced_history

    @_synced_history.setter
    def _synced_history(self, value: List[str]):
        self._local_data.synced_history = value

    def _detect_eval_perspective(self, engine: Stockfish) -> str:
        """
        Detect evaluation perspective for this Stockfish wrapper.
        
        Returns:
            - "side_to_move": score is from side-to-move perspective
            - "white": score is always from White's perspective
        """
        # White up a queen, compare same position with side-to-move swapped.
        fen_white_to_move = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_black_to_move = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        
        try:
            engine.set_fen_position(fen_white_to_move)
            eval_white = engine.get_evaluation()
            engine.set_fen_position(fen_black_to_move)
            eval_black = engine.get_evaluation()
            
            cp_white = eval_white.get("value") if eval_white.get("type") == "cp" else None
            cp_black = eval_black.get("value") if eval_black.get("type") == "cp" else None
            
            if isinstance(cp_white, int) and isinstance(cp_black, int):
                # Side-to-move view flips sign in this test position.
                if cp_white > 0 and cp_black < 0:
                    return EVAL_PERSPECTIVE_SIDE_TO_MOVE
                return EVAL_PERSPECTIVE_WHITE
        
        except Exception:
            pass
        
        # Conservative fallback (matches python-stockfish typical behavior).
        return EVAL_PERSPECTIVE_SIDE_TO_MOVE
    
    def _reset_review_engine(self) -> None:
        """Reset Stockfish to starting position"""
        self.stockfish.set_fen_position(STARTING_FEN)
        self._synced_history_len = 0
        self._synced_history = []

    def _sync_review_engine_to_snapshot(self, history: List[str]) -> None:
        """Synchronize Stockfish to a provided move-history snapshot.
        Uses incremental sync when current state is a prefix of the target."""
        target_len = len(history)

        # Check if current engine state is a prefix of the target history
        if (
            self._synced_history_len > 0
            and self._synced_history_len <= target_len
            and self._synced_history == history[:self._synced_history_len]
        ):
            # Incremental: only replay the new moves
            remaining = history[self._synced_history_len:]
            if remaining:
                self.stockfish.make_moves_from_current_position(remaining)
        else:
            # Full reset required (diverged history)
            self._reset_review_engine()
            if history:
                self.stockfish.make_moves_from_current_position(history)

        self._synced_history_len = target_len
        self._synced_history = list(history)
    
    def _sync_review_engine_to_history(self) -> None:
        """
        Synchronize Stockfish with game history.
        
        Always rebuilds from scratch to avoid drift.
        """
        self._sync_review_engine_to_snapshot(list(database.game_history))

    @staticmethod
    def _is_checkmate_from_legal_moves() -> bool:
        legal_moves = database.get_legal_moves(database.current_turn)
        return sum(len(moves) for moves in legal_moves.values()) == 0

    def _build_request_from_database(self) -> Optional[ReviewRequest]:
        """Snapshot current board/move state for deterministic async review."""
        history = list(database.game_history)
        if not history:
            return None
        
        pgn = list(database.game_pgn)
        matrix_snapshot = [list(row) for row in database.matrix.tolist()]
        
        return ReviewRequest(
            session_id=self._session_id,
            move_count=len(history),
            history=history,
            pgn=pgn,
            matrix_snapshot=matrix_snapshot,
            current_turn=database.current_turn,
            fullmove=database.fullmove,
            last_forced=database.last_forced,
            is_checkmate_after_move=self._is_checkmate_from_legal_moves(),
        )

    @staticmethod
    def _log_queue(message: str) -> None:
        if REVIEW_DEBUG:
            database.gamelogger.move(message)

    def _is_request_session_active(self, request: ReviewRequest) -> bool:
        with self._lock:
            return request.session_id == self._session_id

    @staticmethod
    def _is_game_over() -> bool:
        return database.pgn_result != "*"

    @staticmethod
    def _all_moves_reviewed() -> bool:
        expected_moves = len(database.game_history)
        if expected_moves == 0:
            return False
        
        # Starting-position eval is optional in current flow.
        has_starting_eval = (
            len(database.evaluation_history) > 0 and
            str(database.evaluation_history[0].get("type", "")) == ""
        )
        required = expected_moves + (1 if has_starting_eval else 0)
        return len(database.evaluation_history) >= required

    def log_post_game_summary_if_ready(self) -> None:
        """
        Log per-color accuracy/type summary once game is over and reviews are complete.
        """
        with self._lock:
            queue_idle = not self._request_queue and self._active_workers == 0
        
        if not queue_idle:
            return
        if not self._is_game_over():
            return
        if not self._all_moves_reviewed():
            return
        
        summary_key = (len(database.game_history), str(database.pgn_result))
        with self._lock:
            if self._last_post_game_summary_key == summary_key:
                return
            self._last_post_game_summary_key = summary_key
        
        colors = ("white", "black")
        stats = {
            color: {
                "acc_sum": 0.0,
                "acc_count": 0,
                "types": Counter()
            }
            for color in colors
        }
        
        for ev in database.evaluation_history:
            color = str(ev.get("color", ""))
            move_type = str(ev.get("type", ""))
            if color not in stats or not move_type:
                continue
            
            stats[color]["types"][move_type] += 1
            
            accuracy = ev.get("accuracy", "")
            if isinstance(accuracy, (int, float)):
                stats[color]["acc_sum"] += float(accuracy)
                stats[color]["acc_count"] += 1
        
        type_order = [
            "brilliant", "great", "best", "excellent", "good",
            "miss", "inaccuracy", "mistake", "blunder", "book", "forced"
        ]
        
        database.gamelogger.game("Review Summary (final)")
        for color in colors:
            acc_count = stats[color]["acc_count"]
            if acc_count > 0:
                avg_accuracy = stats[color]["acc_sum"] / acc_count
                avg_text = f"{avg_accuracy:.2f}% ({acc_count} moves)"
            else:
                avg_text = "NA (0 moves)"
            
            database.gamelogger.game(f"{color.capitalize()} average accuracy: {avg_text}")
            
            present_types = stats[color]["types"]
            ordered_parts = [
                f"{move_type}:{present_types[move_type]}"
                for move_type in type_order
                if present_types[move_type] > 0
            ]
            if not ordered_parts:
                ordered_parts = ["none"]
            
            database.gamelogger.game(
                f"{color.capitalize()} move types: {', '.join(ordered_parts)}"
            )
    
    # ==================== EVALUATION METHODS ====================
    
    def starting_evaluation(self) -> None:
        """
        Evaluate the starting position.
        
        Should be called once at game start.
        """
        self._reset_review_engine()
        self._opening_active = True
        
        cp, mate, _ = self._get_current_evaluation([])
        
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
            self._opening_active = True
            if not database.evaluation_history:
                self.starting_evaluation()
            return
        
        request = self._build_request_from_database()
        if request:
            self._evaluate_request(request)

    def _evaluate_request(self, request: ReviewRequest) -> None:
        """
        Evaluate a single queued request snapshot.
        
        Optimized evaluation order:
        1. Sync to before-move position, get eval + top_moves + board
        2. Incrementally sync one move forward for after-move eval
        This reduces engine resets from 3 to 2 (with the second being incremental).
        """
        with self._lock:
            if request.session_id != self._session_id:
                self._log_queue(f"Review stale skip: move {request.move_count}, session {request.session_id}")
                return
        
        # Determine who moved.
        color = "white" if request.current_turn == "black" else "black"
        
        # Move context.
        move_count = request.move_count
        last_move = request.history[-1] if request.history else ""
        last_pgn = request.pgn[-1] if request.pgn else ""
        
        # Generate FEN from request snapshot.
        fen = self.notation_gen.generate_fen(
            board=request.matrix_snapshot,  # type: ignore[arg-type]
            side_to_move=request.current_turn[0],  # type: ignore[index]
            fullmove_number=request.fullmove
        )
        
        # ---- Step 1: Sync to BEFORE-move position ----
        before_history = request.history[:-1] if request.history else []
        cp_before, mate_before, fen_before = self._get_current_evaluation(before_history)
        
        # Avoid storing FEN for upgrades if it's forced/checkmate anyway
        if request.last_forced or request.is_checkmate_after_move:
            fen_before = None
        
        # ---- Step 2: Incrementally sync to AFTER-move position (just 1 move) ----
        cp, mate, _ = self._get_current_evaluation(request.history)
        
        # ===== CLASSIFICATION PRIORITY =====
        
        # Priority 1: FORCED (only legal move)
        if request.last_forced:
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type="forced",
                accuracy=""
            )
            if not self._is_request_session_active(request):
                return
            self.curr_eval = eval_data.to_dict()
            self._update_evaluation_history(force_append=True)
            return
        
        # Priority 2: BOOK (opening theory)
        book_result = self._check_book_move(move_count, last_move, last_pgn, color, fen, cp, mate, request.history)
        if book_result:
            if not self._is_request_session_active(request):
                return
            self.curr_eval = book_result
            self._update_evaluation_history(force_append=True)
            return
        
        # Priority 3: CHECKMATE (game-ending move)
        if request.is_checkmate_after_move:
            move_type = "forced" if request.last_forced else "best"
            
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type=move_type,
                accuracy=100.0
            )
            if not self._is_request_session_active(request):
                return
            self.curr_eval = eval_data.to_dict()
            self._update_evaluation_history(force_append=True)
            return
        
        # Priority 4 & 5: EVALUATION-BASED + SPECIAL UPGRADES
        # Pass pre-computed before-position eval and fen_before for lazy upgrade checks
        self._evaluate_with_engine(
            color=color,
            cp=cp,
            mate=mate,
            last_move=last_move,
            last_pgn=last_pgn,
            move_count=move_count,
            history_snapshot=request.history,
            pgn_snapshot=request.pgn,
            matrix_snapshot=request.matrix_snapshot,
            force_append=True,
            request_session_id=request.session_id,
            cp_before=cp_before,
            mate_before=mate_before,
            fen_before=fen_before
        )
    
    def _check_book_move(
        self,
        move_count: int,
        last_move: str,
        last_pgn: str,
        color: str,
        fen: str,
        cp,
        mate,
        history: List[str]
    ) -> Optional[Dict]:
        """
        Check if move is in opening book.
        
        Returns:
            Evaluation dict if book move, None otherwise
        """
        # Once opening is broken, never return to book in this game.
        if not self._opening_active:
            return None
            
        play_string = ",".join(history)
        
        # First move is always book (with fallback name)
        if move_count == 1:
            self._opening_active = True
            success, opening = self.opening_book.get_opening_from_api(fen, play=play_string)
            if not opening:
                opening = self.opening_book.get_first_move_name(last_move)
            
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type="book",
                accuracy="",
                opening=opening,
                book_source="forced_first_move"
            )
            return eval_data.to_dict()
        
        # API-only policy after move 1.
        success, opening = self.opening_book.get_opening_from_api(fen, play=play_string)
        
        if success:
            # API responded successfully
            if opening:
                eval_data = EvaluationData(
                    color=color,
                    cp=cp,
                    mate=mate,
                    move_type="book",
                    accuracy="",
                    opening=opening,
                    book_source="api"
                )
                return eval_data.to_dict()
            else:
                # API responded, but genuinely out of book. Close opening phase permanently.
                self._opening_active = False
                return None
        else:
            # API failed (timeout/network). Don't close book permanently yet.
            self._api_failure_count += 1
            if self._api_failure_count >= 3:
                self._opening_active = False
            
            # Since we don't know if it's a book move, return a fallback "book" to avoid
            # misclassifying an opening move as "best" or "blunder".
            eval_data = EvaluationData(
                color=color,
                cp=cp,
                mate=mate,
                move_type="book",
                accuracy="",
                opening="Opening Move",
                book_source="fallback"
            )
            return eval_data.to_dict()
    
    def _evaluate_with_engine(
        self,
        color: str,
        cp,
        mate,
        last_move: str,
        last_pgn: str,
        move_count: Optional[int] = None,
        history_snapshot: Optional[List[str]] = None,
        pgn_snapshot: Optional[List[str]] = None,
        matrix_snapshot: Optional[List[List[Any]]] = None,
        force_append: bool = False,
        request_session_id: Optional[int] = None,
        cp_before=None,
        mate_before=None,
        fen_before: Optional[str] = None
    ) -> None:
        """
        Evaluate move using engine comparison.
        
        Calculates EP loss, base classification, and checks for upgrades.
        Accepts optional pre-computed before-position data to avoid redundant engine syncs.
        """
        if history_snapshot is None:
            history_snapshot = list(database.game_history)
        if pgn_snapshot is None:
            pgn_snapshot = list(database.game_pgn)
        if matrix_snapshot is None:
            matrix_snapshot = [list(row) for row in database.matrix.tolist()]
        if move_count is None:
            move_count = len(history_snapshot)
        
        # Use pre-computed before-eval or compute it (fallback for direct calls)
        if cp_before is None and mate_before is None:
            cp_before, mate_before = self._get_evaluation_for_history(history_snapshot[:-1])
        cp_after = cp
        mate_after = mate
        
        # Convert to mover perspective (auto-detected engine perspective).
        mover_cp_before = self._to_mover_cp(cp_before, mate_before, color, is_after_move=False)
        mover_cp_after = self._to_mover_cp(cp_after, mate_after, color, is_after_move=True)
        
        # Calculate losses from mover perspective.
        ep_loss = self._calculate_mover_ep_loss(mover_cp_before, mover_cp_after)
        cp_loss = self._calculate_mover_cp_loss(mover_cp_before, mover_cp_after)
        
        # Calculate accuracy
        accuracy = self.accuracy_calc.calculate(ep_loss)
        
        # Get base classification (hybrid EP + CP loss)
        move_type = self.move_classifier.classify_from_hybrid_loss(
            ep_loss=ep_loss,
            cp_loss=cp_loss,
            mover_cp_before=mover_cp_before,
            mover_cp_after=mover_cp_after,
            move_count=move_count,
            last_pgn=last_pgn
        )
        
        # Check for upgrades (Brilliant, Great)
        # Only upgrade if base class is good enough
        if move_type in ("best", "excellent") and ep_loss < THRESHOLD_EXCELLENT:
            move_type = self._check_special_upgrades(
                base_type=move_type,
                ep_loss=ep_loss,
                last_move=last_move,
                last_pgn=last_pgn,
                color=color,
                cp_before=cp_before,
                mate_before=mate_before,
                history_snapshot=history_snapshot,
                pgn_snapshot=pgn_snapshot,
                after_matrix_snapshot=matrix_snapshot,
                fen_before=fen_before
            )
        
        # Store result
        eval_data = EvaluationData(
            color=color,
            cp=cp_after,
            mate=mate_after,
            move_type=move_type,
            accuracy=accuracy,
            ep_loss=ep_loss,
            cp_loss=cp_loss
        )
        
        with self._lock:
            if request_session_id is not None and request_session_id != self._session_id:
                self._log_queue(f"Review stale drop before append: move {move_count}, session {request_session_id}")
                return
        
        self.curr_eval = eval_data.to_dict()
        self._update_evaluation_history(force_append=force_append)
    
    def _check_special_upgrades(
        self,
        base_type: str,
        ep_loss: float,
        last_move: str,
        last_pgn: str,
        color: str,
        cp_before,
        mate_before,
        history_snapshot: List[str],
        pgn_snapshot: List[str],
        after_matrix_snapshot: List[List[Any]],
        fen_before: Optional[str] = None
    ) -> str:
        """
        Check if move should be upgraded to Brilliant or Great.
        
        Uses fen_before for efficient position reset (no full replay needed).
        Only called for upgrade-worthy moves (best/excellent).
        
        Returns:
            Upgraded classification or base_type
        """
        # Get before-position data: use fen_before for efficient reset if available
        if fen_before:
            try:
                self.stockfish.set_fen_position(fen_before)
                self._synced_history_len = 0
                self._synced_history = []
                top_moves = self.stockfish.get_top_moves(3) or []
                before_board = self.board_analyzer._board_from_fen(fen_before)
            except Exception as e:
                database.gamelogger.error(f"Failed to get upgrade data via FEN: {e}")
                return base_type
        else:
            before_board, top_moves = self._rebuild_before_position_data(history_snapshot)
        
        if not before_board or not top_moves:
            return base_type
        
        # Get board after move from request matrix snapshot
        after_board = self.board_analyzer._board_from_matrix(after_matrix_snapshot)
        
        # Parse move
        parsed_move = self._parse_uci_move(last_move)
        if not parsed_move:
            return base_type
        
        # Get colors
        mover_color = color
        opponent_color = "black" if color == "white" else "white"
        
        # Check if in check before move
        was_in_check = self._was_in_check_before(
            before_board, mover_color, opponent_color, pgn_snapshot
        )
        
        # Check if position was lost
        already_lost = self._position_was_lost(cp_before, mate_before, mover_color)
        
        # Check Brilliant upgrade
        if self.move_classifier.upgrade_to_brilliant(
            ep_loss, last_move, last_pgn, parsed_move, top_moves,
            before_board, after_board, mover_color, opponent_color,
            was_in_check, already_lost, cp_before
        ):
            return "brilliant"
        
        # Check Great upgrade (only if not Brilliant)
        if base_type == "best" and self.move_classifier.upgrade_to_great(
            ep_loss, last_move, last_pgn, top_moves,
            was_in_check, already_lost
        ):
            return "great"
        
        return base_type
    
    # ==================== HELPER METHODS ====================

    @staticmethod
    def _raw_cp_from_evaluation(cp_value, mate_value) -> Optional[float]:
        """Convert CP/mate evaluation to a white-centric centipawn-like score."""
        if isinstance(mate_value, (int, float)):
            mate = int(mate_value)
            return float(100000 - abs(mate)) if mate > 0 else float(-100000 + abs(mate))
        
        if isinstance(cp_value, (int, float)):
            return float(cp_value)
        
        return None

    def _to_mover_cp(
        self,
        cp_value,
        mate_value,
        mover_color: str,
        is_after_move: bool
    ) -> Optional[float]:
        """
        Convert engine evaluation into mover-centric centipawns.
        
        For side-to-move perspective engines:
            - before move: score already from mover perspective
            - after move: score is opponent perspective, so flip sign
        
        For white-centric engines:
            - white mover: unchanged
            - black mover: flip sign
        """
        raw_cp = self._raw_cp_from_evaluation(cp_value, mate_value)
        if raw_cp is None:
            return None
        
        if self._eval_perspective == EVAL_PERSPECTIVE_SIDE_TO_MOVE:
            return -raw_cp if is_after_move else raw_cp
        
        return raw_cp if mover_color == "white" else -raw_cp

    def _calculate_mover_ep_loss(
        self,
        mover_cp_before: Optional[float],
        mover_cp_after: Optional[float]
    ) -> float:
        """Calculate EP loss directly from mover-centric CP values."""
        if mover_cp_before is None or mover_cp_after is None:
            return 0.0
        
        ep_before = self.ep_calc.from_centipawns(mover_cp_before)
        ep_after = self.ep_calc.from_centipawns(mover_cp_after)
        return max(0.0, ep_before - ep_after)

    @staticmethod
    def _calculate_mover_cp_loss(
        mover_cp_before: Optional[float],
        mover_cp_after: Optional[float]
    ) -> float:
        """Calculate CP loss from mover-centric CP values."""
        if mover_cp_before is None or mover_cp_after is None:
            return 0.0
        
        return max(0.0, mover_cp_before - mover_cp_after)

    def _get_evaluation_for_history(
        self,
        history_snapshot: List[str]
    ) -> Tuple[int | float | str, int | float | str]:
        """
        Evaluate position reached by a specific history snapshot.
        
        Returns:
            (cp, mate) tuple
        """
        cp, mate, _ = self._get_current_evaluation(history_snapshot)
        return cp, mate
    
    def _get_current_evaluation(self, history: List[str]) -> Tuple[int | float | str, int | float | str, str]:
        """
        Get current position evaluation from Stockfish, utilizing the cache.
        
        Returns:
            (cp, mate, fen) tuple
        """
        cache_key = ",".join(history)

        # Check cache first
        with self._lock:
            if cache_key in self._eval_cache:
                return self._eval_cache[cache_key]

        # Not in cache, compute it
        self._sync_review_engine_to_snapshot(history)
        evaluation = self.stockfish.get_evaluation()
        
        if evaluation["type"] == "cp":
            cp = evaluation["value"]
            mate = ""
        else:
            cp = ""
            mate = evaluation["value"]

        try:
            fen = self.stockfish.get_fen_position()
        except Exception:
            fen = ""

        result = (cp, mate, fen)
        
        # Save to cache
        with self._lock:
            self._eval_cache[cache_key] = result

        return result
    
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
        opponent_color: str,
        pgn_snapshot: Optional[List[str]] = None
    ) -> bool:
        """Check if king was in check before the move"""
        king_square = self.board_analyzer._find_king(before_board, mover_color)
        if not king_square:
            # Fallback: check previous PGN for check symbol
            if pgn_snapshot is None:
                pgn_snapshot = list(database.game_pgn)
            if len(pgn_snapshot) > 1:
                return "+" in pgn_snapshot[-2]
            return False
        
        attackers = self.board_analyzer._attackers_to_square(
            before_board, king_square, opponent_color
        )
        return len(attackers) > 0
    
    def _position_was_lost(self, cp_before, mate_before, mover_color: str) -> bool:
        """Check if mover was already lost before the move."""
        mover_cp_before = self._to_mover_cp(cp_before, mate_before, mover_color, is_after_move=False)
        if mover_cp_before is None:
            return False
        
        return mover_cp_before < -300
    
    def _rebuild_before_position_data(
        self,
        history_snapshot: Optional[List[str]] = None
    ) -> Tuple[Optional[List[List[str]]], List[dict]]:
        """
        Rebuild board state before last move and get engine analysis.
        
        Returns:
            (before_board, top_moves) tuple
        """
        if history_snapshot is None:
            history_snapshot = list(database.game_history)
        
        if not history_snapshot:
            return None, []
        
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
            # Re-sync to request/current snapshot
            try:
                self._sync_review_engine_to_snapshot(history_snapshot)
            except Exception:
                pass
    
    def _update_evaluation_history(self, force_append: bool = False) -> None:
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
        if not force_append and curr_eval == db_eval[-1]:
            return
        
        db_eval.append(curr_eval)
    
    # ==================== ASYNC EVALUATION ====================
    
    def evaluate_last_move_async(self) -> None:
        """
        Queue non-blocking move evaluation.
        
        Queues each move snapshot for ordered per-move evaluation.
        Thread-safe.
        """
        request = self._build_request_from_database()
        if request is None:
            return
        
        with self._lock:
            # Duplicate suppression for repeated async calls on same move index.
            if request.move_count <= self._last_enqueued_move_count:
                return
            
            self._request_queue.append(request)
            self._last_enqueued_move_count = request.move_count
            self._log_queue(f"Review enqueue: move {request.move_count}, session {request.session_id}")
            
            # Start worker if not already running. We have max_workers=2.
            while len(self._pending_futures) < 2 and self._request_queue:
                future = self._executor.submit(self._drain_evaluation_queue)
                self._pending_futures.append(future)

            # Clean up done futures
            self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def _drain_evaluation_queue(self) -> None:
        """
        Worker thread that drains evaluation queue.
        
        Continues evaluating while requests are pending.
        """
        with self._lock:
            self._active_workers += 1

        try:
            while True:
                request: Optional[ReviewRequest] = None
                with self._lock:
                    if not self._request_queue:
                        break
                    else:
                        request = self._request_queue.popleft()

                if request is None:
                    # Defensive guard for static analyzers; runtime should not reach here.
                    continue

                self._log_queue(f"Review dequeue: move {request.move_count}, session {request.session_id}")

                with self._lock:
                    if request.session_id != self._session_id:
                        self._log_queue(
                            f"Review stale skip in drain: move {request.move_count}, "
                            f"session {request.session_id} != {self._session_id}"
                        )
                        continue

                # Evaluate outside lock
                self._evaluate_guarded(request)
        finally:
            should_log_summary = False
            with self._lock:
                self._active_workers -= 1
                if self._active_workers == 0 and not self._request_queue:
                    should_log_summary = True
            
            if should_log_summary:
                self.log_post_game_summary_if_ready()
    
    def _evaluate_guarded(self, request: Optional[ReviewRequest] = None) -> None:
        """
        Evaluate with exception handling.
        
        Logs errors but doesn't crash.
        """
        try:
            if request is not None:
                self._evaluate_request(request)
            else:
                self.evaluate_last_move()
            
            # Preserve per-move review log output.
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
            self._request_queue.clear()
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
