"""
Chess Game Utilities Module
Provides UI utilities, coordinate conversions, FEN generation, and legal moves calculation.
"""

import tkinter as tk
import customtkinter as ctk
import numpy as np
from typing import Tuple, Dict, List, Optional, Literal, Callable
from functools import lru_cache
from PIL import Image

from database.database import database


# ==================== CONSTANTS ====================

# Piece type mapping for FEN generation
PIECE_TO_FEN = {
    "p": "P", "r": "R", "n": "N", "b": "B", "q": "Q", "k": "K",
    "-p": "p", "-r": "r", "-n": "n", "-b": "b", "-q": "q", "-k": "k",
}

# Direction vectors for piece movement
STRAIGHT_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRECTIONS = STRAIGHT_DIRECTIONS + DIAGONAL_DIRECTIONS
KNIGHT_OFFSETS = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]

# Board boundaries
BOARD_MIN = 0
BOARD_MAX = 7


# ==================== COORDINATE UTILITIES ====================

class CoordinateConverter:
    """Handles conversion between matrix and chess notation"""
    
    @staticmethod
    def matrix_to_chess(square: Tuple[int, int]) -> str:
        """
        Convert matrix coordinates to chess notation.
        
        Args:
            square: (row, col) where matrix[0][0] = a8, matrix[7][7] = h1
            
        Returns:
            Chess notation (e.g., 'e4')
        """
        row, col = square
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank
    
    @staticmethod
    def chess_to_matrix(notation: str) -> Tuple[int, int]:
        """
        Convert chess notation to matrix coordinates.
        
        Args:
            notation: Chess notation (e.g., 'e4')
            
        Returns:
            (row, col) tuple
        """
        file, rank = notation[0], notation[1]
        col = ord(file) - ord('a')
        row = 8 - int(rank)
        return (row, col)


# ==================== FEN GENERATION ====================

class FENGenerator:
    """Handles FEN (Forsyth-Edwards Notation) string generation"""
    
    @staticmethod
    def generate(
        board: Optional[np.ndarray] = None,
        side_to_move: Literal["w", "b"] = "w",
        halfmove_clock: int = 0,
        fullmove_number: int = 1
    ) -> str:
        """
        Generate FEN string from board state.
        
        Args:
            board: 8x8 numpy array (uses database.matrix if None)
            side_to_move: Current player ('w' or 'b')
            halfmove_clock: Halfmove clock for fifty-move rule
            fullmove_number: Fullmove number
            
        Returns:
            Complete FEN string
        """
        if board is None:
            board = database.matrix
        
        # Build board position
        fen_rows = []
        for row in board:
            empty = 0
            fen_row = ""
            
            for cell in row:
                if cell == 0:
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    
                    piece_type = str(cell)[:-1].lower()
                    fen_piece = PIECE_TO_FEN.get(piece_type, "")
                    fen_row += fen_piece
            
            if empty > 0:
                fen_row += str(empty)
            
            fen_rows.append(fen_row)
        
        # Build castling rights
        castling = ""
        if not database.k1_moved:
            if not database.r2_moved:
                castling += "K"
            if not database.r1_moved:
                castling += "Q"
        if not database.k1_black_moved:
            if not database.r2_black_moved:
                castling += "k"
            if not database.r1_black_moved:
                castling += "q"
        castling = castling or "-"
        
        en_passant = database.en_passant or "-"
        board_fen = "/".join(fen_rows)
        
        return f"{board_fen} {side_to_move} {castling} {en_passant} {halfmove_clock} {fullmove_number}"


# ==================== UI UTILITIES ====================

class UIUtilities:
    """UI-related utility functions"""
    
    @staticmethod
    def fullscreen_window(window: tk.Tk | ctk.CTk) -> None:
        """Set window to fullscreen mode"""
        window.attributes("-fullscreen", True)
        window.update_idletasks()
    
    @staticmethod
    def fullscreen_toggle(window: tk.Tk | ctk.CTk) -> None:
        """Toggle fullscreen state"""
        current_state = window.attributes("-fullscreen")
        window.attributes("-fullscreen", not current_state)
        print(f"Fullscreen: {not current_state}\n")
    
    @staticmethod
    def calculate_centered_relx(rely: float, dimensions: Tuple[int, int]) -> float:
        """
        Calculate relative x position to center a square board.
        
        Args:
            rely: Relative y position (0.0 to 1.0)
            dimensions: (height, width) of window
            
        Returns:
            relx value to center the square board
        """
        height, width = dimensions
        available_height = height * (1 - 2 * rely)
        relx = (width - available_height) / (2 * width)
        return relx
    
    @staticmethod
    @lru_cache(maxsize=128)
    def create_image(path: str, size: Tuple[int, int] = (70, 70)) -> ctk.CTkImage:
        """
        Create CTkImage with caching for performance.
        
        CHANGED: Added LRU cache to avoid reloading same images
        
        Args:
            path: Path to image file
            size: (width, height) in pixels
            
        Returns:
            CTkImage object
        """
        img = Image.open(path)
        
        # Ensure RGBA for transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)


# ==================== PIECE UTILITIES ====================

class PieceUtilities:
    """Helper functions for piece management"""
    
    @staticmethod
    def get_color(piece: str) -> Literal["white", "black"]:
        """Get color of a piece"""
        return "black" if '-' in piece else "white"
    
    @staticmethod
    def get_type(piece: str) -> str:
        """Get type of piece (p, r, n, b, q, k)"""
        return piece.strip('-')[0]
    
    @staticmethod
    def generate_next_piece(base: str, color: Literal["white", "black"]) -> str:
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


# ==================== LEGAL MOVES ENGINE ====================

class LegalMovesEngine:
    """
    Chess legal move calculation engine.
    Handles all piece move generation, pin detection, and check validation.
    """
    
    def __init__(self) -> None:
        """Initialize the legal moves engine"""
        self._last_matrix_hash: Optional[int] = None
        self.coords = CoordinateConverter()
        self.pieces = PieceUtilities()
        self.update_legal_moves(database.matrix)
    
    # ==================== UTILITY METHODS ====================
    
    @staticmethod
    def search_piece(piece: str, matrix: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Find position of a piece on the board.
        
        Args:
            piece: Piece identifier (e.g., 'k1', '-r2')
            matrix: Board state (uses database.matrix if None)
            
        Returns:
            (row, col) position
            
        Raises:
            ValueError: If piece not found
        """
        if matrix is None:
            matrix = database.matrix
        
        pos = np.where(matrix == piece)
        if len(pos[0]) == 0:
            raise ValueError(f"{piece} not found on board")
        return int(pos[0][0]), int(pos[1][0])
    
    @staticmethod
    def is_valid_square(row: int, col: int) -> bool:
        """Check if coordinates are within board bounds"""
        return BOARD_MIN <= row <= BOARD_MAX and BOARD_MIN <= col <= BOARD_MAX
    
    @staticmethod
    def is_occupied(val) -> bool:
        """Check if square contains a piece"""
        return val != 0
    
    @staticmethod
    def is_enemy(val, color: Literal["white", "black"]) -> bool:
        """Check if piece belongs to opponent"""
        if val == 0:
            return False
        piece_is_black = '-' in str(val)
        return piece_is_black != (color == "black")
    
    # ==================== PIN DETECTION ====================
    
    def can_attack_in_direction(self, piece: str, direction: Tuple[int, int]) -> bool:
        """Check if a sliding piece can attack along a direction"""
        piece_type = self.pieces.get_type(piece)
        dr, dc = direction
        
        is_straight = (dr == 0 or dc == 0)
        is_diagonal = (abs(dr) == abs(dc) and dr != 0)
        
        if piece_type == 'r':
            return is_straight
        elif piece_type == 'b':
            return is_diagonal
        elif piece_type == 'q':
            return is_straight or is_diagonal
        return False
    
    def find_pins(self, color: Literal["white", "black"], matrix: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Find all pinned pieces for a color.
        
        Returns:
            Dictionary mapping pinned pieces to pin direction
        """
        pins = {}
        king = "k1" if color == "white" else "-k1"
        
        try:
            king_row, king_col = self.search_piece(king, matrix)
        except ValueError:
            return pins  # King not on board
        
        # Check all 8 directions
        for dr, dc in ALL_DIRECTIONS:
            friendly_piece = None
            row, col = king_row + dr, king_col + dc
            
            while self.is_valid_square(row, col):
                piece = matrix[row, col]
                
                if piece != 0:
                    if not self.is_enemy(piece, color):
                        # Friendly piece
                        if friendly_piece is None:
                            friendly_piece = (piece, row, col)
                        else:
                            # Second friendly piece blocks pin
                            break
                    else:
                        # Enemy piece
                        if friendly_piece is not None:
                            if self.can_attack_in_direction(piece, (dr, dc)):
                                pins[friendly_piece[0]] = (dr, dc)
                        break
                
                row += dr
                col += dc
        
        return pins
    
    # ==================== CHECK DETECTION ====================
    
    def opponent_legal_search(
        self,
        color: Literal["white", "black"],
        coordinates: Tuple[int, int],
        matrix: Optional[np.ndarray] = None,
        return_piece: bool = False
    ) -> bool | List[str]:
        """
        Check if opponent can attack a square.
        
        Args:
            color: Current player color
            coordinates: Square to check
            matrix: Board state
            return_piece: If True, return list of attacking pieces
            
        Returns:
            True if square is under attack, or list of attacking pieces
        """
        if matrix is None:
            matrix = database.matrix
        
        moves = database.white_legal_moves.items() if color == "black" else database.black_legal_moves.items()
        pieces = []
        
        for piece, move in moves:
            if len(move) == 0:
                continue
            
            if np.any(np.all(move == coordinates, axis=1)):
                if return_piece:
                    pieces.append(piece)
                else:
                    return True
        
        return pieces if return_piece else False
    
    def check_checker(self, color: Literal["white", "black"], matrix: Optional[np.ndarray] = None) -> bool:
        """Check if the king is in check"""
        if matrix is None:
            matrix = database.matrix
        
        king = "k1" if color == "white" else "-k1"
        try:
            king_pos = self.search_piece(king, matrix)
            return self.opponent_legal_search(color, king_pos, matrix)  # type: ignore
        except ValueError:
            return False
    
    def check_legal(self, color: Literal["white", "black"], matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]] | bool:
        """
        Get squares that can block or capture the checking piece.
        
        Returns:
            - True if not in check
            - Empty list if double check (only king can move)
            - List of blocking squares otherwise
        """
        if matrix is None:
            matrix = database.matrix
        
        king = "k1" if color == "white" else "-k1"
        try:
            king_coordinates = self.search_piece(king, matrix)
        except ValueError:
            return True
        
        pieces = self.opponent_legal_search(color, king_coordinates, matrix, return_piece=True)
        
        if not pieces or isinstance(pieces, bool):
            return True
        
        if len(pieces) >= 2:
            return []  # Double check
        
        piece = pieces[0]
        piece_pos = self.search_piece(piece, matrix)
        
        # Knights and pawns can only be captured
        if 'n' in piece or 'p' in piece:
            return [piece_pos]
        
        # Sliding pieces can be blocked
        return self._calculate_blocking_squares(piece, piece_pos, king_coordinates)
    
    def _calculate_blocking_squares(
        self,
        piece: str,
        piece_pos: Tuple[int, int],
        king_pos: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Calculate squares between attacker and king"""
        piece_type = self.pieces.get_type(piece)
        
        if piece_type == 'r':
            return self._rook_blocking_squares(piece_pos, king_pos)
        elif piece_type == 'b':
            return self._bishop_blocking_squares(piece_pos, king_pos)
        elif piece_type == 'q':
            # Queen attacks like rook or bishop
            if piece_pos[0] == king_pos[0] or piece_pos[1] == king_pos[1]:
                return self._rook_blocking_squares(piece_pos, king_pos)
            else:
                return self._bishop_blocking_squares(piece_pos, king_pos)
        
        return []
    
    def _rook_blocking_squares(self, rook_pos: Tuple[int, int], king_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get squares between rook and king"""
        if rook_pos[0] == king_pos[0]:  # Same row
            start = min(rook_pos[1], king_pos[1])
            end = max(rook_pos[1], king_pos[1])
            return [(rook_pos[0], col) for col in range(start, end + 1)]
        else:  # Same column
            start = min(rook_pos[0], king_pos[0])
            end = max(rook_pos[0], king_pos[0])
            return [(row, rook_pos[1]) for row in range(start, end + 1)]
    
    def _bishop_blocking_squares(self, bishop_pos: Tuple[int, int], king_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get squares between bishop and king"""
        distance = abs(bishop_pos[0] - king_pos[0])
        dr = 1 if bishop_pos[0] > king_pos[0] else -1
        dc = 1 if bishop_pos[1] > king_pos[1] else -1
        
        return [
            (king_pos[0] + dr * (i + 1), king_pos[1] + dc * (i + 1))
            for i in range(distance)
        ]
    
    def check_allowed_moves(
        self,
        moves: List[Tuple[int, int]],
        color: Literal["white", "black"],
        matrix: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int]]:
        """Filter moves to only those that resolve check"""
        if not self.check_checker(color, matrix):
            return moves
        
        allowed_moves = self.check_legal(color, matrix)
        
        if allowed_moves is True:
            return moves
        
        if not allowed_moves:
            return []
        
        return [move for move in moves if move in allowed_moves]
    
    # ==================== CASTLING ====================
    
    def can_castle(self, color: Literal["white", "black"], castle_range: range, matrix: Optional[np.ndarray] = None) -> bool:
        """Check if castling path is clear and safe"""
        if matrix is None:
            matrix = database.matrix
        
        opponent_color = "white" if color == "black" else "black"
        row = 0 if color == "black" else 7
        king_col = 4
        
        # Check if king is in check (use fresh calculation)
        if self._is_square_attacked_by_any_piece((row, king_col), opponent_color, matrix):
            return False
        
        # Check if path is clear and safe (use fresh calculation)
        for col in castle_range:
            if self.is_occupied(matrix[row, col]):
                return False
            if self._is_square_attacked_by_any_piece((row, col), opponent_color, matrix):
                return False
        
        return True
    
    # ==================== SLIDING PIECE MOVES ====================
    
    def _generate_sliding_moves(
        self,
        piece: str,
        position: Tuple[int, int],
        directions: List[Tuple[int, int]],
        matrix: np.ndarray,
        pin_direction: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int]]:
        """Generate moves for sliding pieces (rook, bishop, queen)"""
        color = self.pieces.get_color(piece)
        moves = []
        row, col = position
        
        for dr, dc in directions:
            # Skip direction if pinned in different direction
            if pin_direction:
                if (dr, dc) != pin_direction and (dr, dc) != (-pin_direction[0], -pin_direction[1]):
                    continue
            
            r, c = row + dr, col + dc
            
            while self.is_valid_square(r, c):
                if self.is_occupied(matrix[r, c]):
                    if self.is_enemy(matrix[r, c], color):
                        moves.append((r, c))
                    break
                moves.append((r, c))
                r += dr
                c += dc
        
        return moves
    
    def rook_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal rook moves"""
        if matrix is None:
            matrix = database.matrix
        
        # Check if pinned diagonally
        pin_direction = database.pins.get(piece)
        if pin_direction and pin_direction in DIAGONAL_DIRECTIONS:
            return []
        
        color = self.pieces.get_color(piece)
        position = self.search_piece(piece, matrix)
        
        moves = self._generate_sliding_moves(piece, position, STRAIGHT_DIRECTIONS, matrix, pin_direction)
        moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves
    
    def bishop_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal bishop moves"""
        if matrix is None:
            matrix = database.matrix
        
        # Check if pinned horizontally/vertically
        pin_direction = database.pins.get(piece)
        if pin_direction and pin_direction in STRAIGHT_DIRECTIONS:
            return []
        
        color = self.pieces.get_color(piece)
        position = self.search_piece(piece, matrix)
        
        moves = self._generate_sliding_moves(piece, position, DIAGONAL_DIRECTIONS, matrix, pin_direction)
        moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves
    
    def queen_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal queen moves (combination of rook and bishop)"""
        return self.rook_moves(piece, matrix) + self.bishop_moves(piece, matrix)
    
    # ==================== KNIGHT MOVES ====================
    
    def knight_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal knight moves"""
        if matrix is None:
            matrix = database.matrix
        
        # Knights cannot move if pinned
        if piece in database.pins:
            return []
        
        color = self.pieces.get_color(piece)
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        for dr, dc in KNIGHT_OFFSETS:
            r, c = row + dr, col + dc
            if self.is_valid_square(r, c):
                if not self.is_occupied(matrix[r, c]) or self.is_enemy(matrix[r, c], color):
                    moves.append((r, c))
        
        moves = self.check_allowed_moves(moves, color, matrix)
        return moves
    
    # ==================== PAWN MOVES ====================
    
    def pawn_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal pawn moves"""
        if matrix is None:
            matrix = database.matrix
        
        # Check pin
        pin_direction = database.pins.get(piece)
        if pin_direction and pin_direction[1] != 0 and pin_direction[0] == 0:
            return []  # Pinned horizontally
        
        color = self.pieces.get_color(piece)
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        # Determine direction and starting row
        if color == "white":
            direction = -1
            start_row = 6
            opponent_last_pawn = database.black_last_pawn
        else:
            direction = 1
            start_row = 1
            opponent_last_pawn = database.white_last_pawn
        
        # Forward moves
        if not pin_direction or pin_direction[1] == 0:
            # Single forward
            if self.is_valid_square(row + direction, col):
                if not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction, col))
                    
                    # Double forward from starting position
                    if row == start_row:
                        if not self.is_occupied(matrix[row + direction * 2, col]):
                            moves.append((row + direction * 2, col))
        
        # Diagonal captures
        if not pin_direction or pin_direction[1] != 0:
            for dc in [-1, 1]:
                r, c = row + direction, col + dc
                if self.is_valid_square(r, c):
                    if self.is_enemy(matrix[r, c], color):
                        moves.append((r, c))
            
            # En passant
            en_passant_row = 3 if color == "white" else 4
            if opponent_last_pawn and row == en_passant_row:
                opp_row, opp_col = opponent_last_pawn
                if opp_row == row and abs(opp_col - col) == 1:
                    target_square = (row + direction, opp_col)
                    moves.append(target_square)
        
        moves = self.check_allowed_moves(moves, color, matrix)
        return moves
    
    # ==================== KING MOVES ====================
    
    def king_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal king moves including castling"""
        if matrix is None:
            matrix = database.matrix
        
        color = self.pieces.get_color(piece)
        row, col = self.search_piece(piece, matrix)
        opponent_color = "black" if color == "white" else "white"
        
        # Find opponent king position
        opponent_king = "-k1" if color == "white" else "k1"
        try:
            opp_king_row, opp_king_col = self.search_piece(opponent_king, matrix)
        except ValueError:
            opp_king_row, opp_king_col = -10, -10
        
        moves = []
        
        # Normal king moves (8 adjacent squares)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                r, c = row + dr, col + dc
                if not self.is_valid_square(r, c):
                    continue
                
                # Can't move next to opponent king
                king_distance = max(abs(r - opp_king_row), abs(c - opp_king_col))
                if king_distance <= 1:
                    continue
                
                if not self.is_occupied(matrix[r, c]) or self.is_enemy(matrix[r, c], color):
                    moves.append((r, c))
        
        # Castling
        if color == "white":
            castle_checks = [(database.r1_moved, database.k1_moved), (database.r2_moved, database.k1_moved)]
            rook_pieces = ["r1", "r2"]
        else:
            castle_checks = [(database.r1_black_moved, database.k1_black_moved), (database.r2_black_moved, database.k1_black_moved)]
            rook_pieces = ["-r1", "-r2"]
        
        row_castle = 0 if color == "black" else 7
        castle_ranges = [range(1, 4), range(5, 7)]
        castle_cols = [2, 6]
        rook_cols = [0, 7]
        
        for idx, (castle_range, castle_col) in enumerate(zip(castle_ranges, castle_cols)):
            if not any(castle_checks[idx]):
                rook_piece = rook_pieces[idx]
                expected_rook_pos = (row_castle, rook_cols[idx])
                
                if matrix[expected_rook_pos] == rook_piece:
                    if self.can_castle(color, castle_range, matrix):
                        moves.append((row_castle, castle_col))
        
        # Filter using attack checking
        safe_moves = self._filter_king_moves_by_safety(moves, piece, row, col, opponent_color, matrix)
        return safe_moves
    
    def _filter_king_moves_by_safety(
        self,
        moves: List[Tuple[int, int]],
        piece: str,
        king_row: int,
        king_col: int,
        opponent_color: Literal["white", "black"],
        matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Filter king moves to only safe squares"""
        safe_moves = []
        
        for move in moves:
            # Simulate the move
            temp_matrix = matrix.copy()
            temp_matrix[king_row, king_col] = 0
            temp_matrix[move] = piece
            
            # Check if square is attacked
            is_safe = not self._is_square_attacked_by_any_piece(move, opponent_color, temp_matrix)
            
            if is_safe:
                safe_moves.append(move)
        
        return safe_moves
    
    def _is_square_attacked_by_any_piece(
        self,
        square: Tuple[int, int],
        by_color: Literal["white", "black"],
        matrix: np.ndarray
    ) -> bool:
        """Check if a square is attacked by any opponent piece (excluding kings)"""
        for r in range(8):
            for c in range(8):
                opponent_piece = matrix[r, c]
                if opponent_piece == 0:
                    continue
                
                # Skip opponent king
                if 'k' in str(opponent_piece).strip('-'):
                    continue
                
                # Check if right color
                piece_color = self.pieces.get_color(opponent_piece)
                if piece_color != by_color:
                    continue
                
                # Check if can attack
                if self._can_piece_attack(opponent_piece, (r, c), square, matrix):
                    return True
        
        return False
    
    def _can_piece_attack(self, piece: str, from_pos: Tuple[int, int], to_pos: Tuple[int, int], matrix: np.ndarray) -> bool:
        """Check if a piece can attack a square (non-recursive)"""
        piece_type = self.pieces.get_type(piece)
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Pawn attacks
        if piece_type == 'p':
            direction = 1 if '-' in piece else -1
            return (to_row == from_row + direction and abs(to_col - from_col) == 1)
        
        # Knight attacks
        elif piece_type == 'n':
            dr, dc = abs(to_row - from_row), abs(to_col - from_col)
            return (dr == 2 and dc == 1) or (dr == 1 and dc == 2)
        
        # Bishop/Queen diagonal attacks
        elif piece_type in ['b', 'q']:
            if abs(to_row - from_row) == abs(to_col - from_col):
                return self._is_clear_diagonal(from_pos, to_pos, matrix)
        
        # Rook/Queen straight attacks
        if piece_type in ['r', 'q']:
            if from_row == to_row or from_col == to_col:
                return self._is_clear_straight(from_pos, to_pos, matrix)
        
        return False
    
    def _is_clear_diagonal(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], matrix: np.ndarray) -> bool:
        """Check if diagonal path is clear"""
        fr, fc = from_pos
        tr, tc = to_pos
        
        dr = 1 if tr > fr else -1
        dc = 1 if tc > fc else -1
        
        r, c = fr + dr, fc + dc
        while (r, c) != (tr, tc):
            if matrix[r, c] != 0:
                return False
            r += dr
            c += dc
        
        return True
    
    def _is_clear_straight(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], matrix: np.ndarray) -> bool:
        """Check if straight path is clear"""
        fr, fc = from_pos
        tr, tc = to_pos
        
        if fr == tr:  # Horizontal
            start, end = (min(fc, tc), max(fc, tc))
            for c in range(start + 1, end):
                if matrix[fr, c] != 0:
                    return False
        else:  # Vertical
            start, end = (min(fr, tr), max(fr, tr))
            for r in range(start + 1, end):
                if matrix[r, fc] != 0:
                    return False
        
        return True
    
    # ==================== MAIN CALCULATION METHODS ====================
    
    def calculate_legal_moves(self, piece: str, matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Calculate legal moves for any piece"""
        piece_type = self.pieces.get_type(piece)
        
        move_functions: Dict[str, Callable] = {
            'p': self.pawn_moves,
            'r': self.rook_moves,
            'n': self.knight_moves,
            'b': self.bishop_moves,
            'q': self.queen_moves,
            'k': self.king_moves
        }
        
        generator = move_functions.get(piece_type)
        if generator:
            return generator(piece, matrix)
        return []
    
    def all_legal_moves(self, color: Literal["white", "black"], matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate legal moves for all pieces of a color"""
        if matrix is None:
            matrix = database.matrix
        
        legal = {}
        pieces = database.white_pieces if color == "white" else database.black_pieces
        
        for piece in pieces.flatten():
            try:
                moves = self.calculate_legal_moves(piece, matrix)
                legal[piece] = np.array(moves)
            except ValueError:
                # Piece was captured
                legal[piece] = np.array([])
        
        return legal
    
    def update_legal_moves(self, matrix: Optional[np.ndarray] = None) -> None:
        """Update all legal moves for both colors (only if matrix changed)"""
        if matrix is None:
            matrix = database.matrix
        
        # Check if matrix changed
        current_hash = hash(matrix.tobytes())
        if current_hash == self._last_matrix_hash:
            return
        
        self._last_matrix_hash = current_hash
        
        # Find pins for both colors
        white_pins = self.find_pins("white", matrix)
        black_pins = self.find_pins("black", matrix)
        database.pins = {**white_pins, **black_pins}
        
        # Initialize legal moves for all pieces (including promoted)
        all_white_pieces = set(database.white_pieces.flatten())
        all_black_pieces = set(database.black_pieces.flatten())
        
        database.white_legal_moves = {piece: np.array([]) for piece in all_white_pieces}
        database.black_legal_moves = {piece: np.array([]) for piece in all_black_pieces}
        
        # Calculate legal moves
        if database.current_turn == "white":
            database.black_legal_moves = self.all_legal_moves("black", matrix)
            database.white_legal_moves = self.all_legal_moves("white", matrix)
        else:
            database.white_legal_moves = self.all_legal_moves("white", matrix)
            database.black_legal_moves = self.all_legal_moves("black", matrix)
        
        print("Legal Moves Updated!\n")


# ==================== MAIN UTILITIES CLASS ====================

class Utilities:
    """
    Main utilities class combining all utility functions.
    
    CHANGED: Better organization with separate utility classes
    UNCHANGED: All original methods preserved with same signatures
    """
    
    def __init__(self) -> None:
        self.legal_moves = LegalMovesEngine()
        self.ui = UIUtilities()
        self.coords = CoordinateConverter()
        self.fen = FENGenerator()
        self.pieces = PieceUtilities()
    
    # ==================== CONVENIENCE METHODS ====================
    # These delegate to the utility classes for backward compatibility
    
    def fullscreen_window(self, window: tk.Tk | ctk.CTk) -> None:
        """Set window to fullscreen mode"""
        return self.ui.fullscreen_window(window)
    
    def fullscreen_toggle(self, window: tk.Tk | ctk.CTk) -> None:
        """Toggle fullscreen state"""
        return self.ui.fullscreen_toggle(window)
    
    def relative_dimensions(self, rely: float, dimensions: Tuple[int, int]) -> float:
        """Calculate relative x position to center a square board"""
        return self.ui.calculate_centered_relx(rely, dimensions)
    
    def ctkimage_generator(self, path: str, size: Tuple[int, int] = (70, 70)) -> ctk.CTkImage:
        """Generate CTkImage from path (with caching)"""
        return self.ui.create_image(path, size)
    
    def create_fen(self, **kwargs) -> str:
        """Generate FEN string from current position"""
        return self.fen.generate(**kwargs)
    
    def matrix_to_chess(self, square: Tuple[int, int]) -> str:
        """Convert matrix coordinates to chess notation"""
        return self.coords.matrix_to_chess(square)
    
    def next_piece(self, base: str, color: Literal["white", "black"]) -> str:
        """Generate next piece identifier for promotion"""
        return self.pieces.generate_next_piece(base, color)
    
    def reset(self) -> None:
        """Reset utilities"""
        self.legal_moves = LegalMovesEngine()

    def flip_legal(self, moves_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Flip legal moves for black player"""
        flipped = {}

        for piece, moves in moves_dict.items():
            if moves.size == 0:
                flipped[piece] = moves
                continue

            new_moves = []
            for move in moves:
                new_moves.append((7 - move[0], move[1]))
            flipped[piece] = np.array(new_moves)

        return flipped


# ==================== MODULE TEST ====================

if __name__ == "__main__":
    utils = Utilities()
    fen = utils.create_fen()
    print(f"Starting FEN: {fen}")