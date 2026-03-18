"""
Chess Game Utilities Module
Provides UI utilities, coordinate conversions, FEN generation, and legal moves calculation.
"""
import os
import sys
import warnings
import tkinter as tk
import customtkinter as ctk
import numpy as np
from typing import Tuple, Dict, List, Optional, Literal, Callable
from functools import lru_cache
from PIL import Image
from stockfish import Stockfish
from database.database import database

warnings.filterwarnings("ignore", category=UserWarning, module="stockfish")

# ==================== CONSTANTS ====================

# Piece type mapping for FEN generation
PIECE_TO_FEN = {
    "p": "P", "r": "R", "n": "N", "b": "B", "q": "Q", "k": "K",
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

class NotationGenerator:
    """Handles FEN (Forsyth-Edwards Notation) and standard chess move notation generation"""

    def __init__(self) -> None:
        """Initialize notation generator"""
        self.coords = CoordinateConverter()
        self.pieces = PieceUtilities()
    
    @staticmethod
    def generate_fen(
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
                if cell == 0 or cell == '':
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    
                    # Extract piece type correctly from piece identifier (e.g., 'p1', '-q2')
                    piece_str = str(cell)
                    piece_type = piece_str.lstrip('-')[0].lower()
                    fen_piece = PIECE_TO_FEN.get(piece_type, "")
                    if "-" in piece_str and fen_piece:
                        fen_piece = fen_piece.lower()
                    if fen_piece:
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
    
    @staticmethod
    def generate_matrix(fen: str) -> np.ndarray:
        """
        Generate matrix from FEN string.
        
        Args:
            fen: FEN string
            
        Returns:
            8x8 numpy array
        """
        board = np.zeros((8, 8), dtype=object)
        fen_parts = fen.split(" ")
        board_fen = fen_parts[0]
        
        fen_rows = board_fen.split("/")
        for row_idx, fen_row in enumerate(fen_rows):
            col_idx = 0
            for char in fen_row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    if char == char.lower():
                        char = "-" + char.lower()
                    else:
                        char = char.lower()
                    board[row_idx, col_idx] = char + "1"
                    col_idx += 1
        
        return board
    
    def chess_notation(self, piece: str, to_square: Tuple[int, int], promotion: Optional[str] = None, check: bool = False, checkmate: bool = False, castle: bool = False, capture: bool = False, from_square: Optional[Tuple[int, int]] = None, disambiguation: str = "") -> str:
        """
        Generate standard chess notation for a move.
        
        Args:
            piece: The piece being moved (e.g., 'p1', 'n1', '-p2', or just type 'p', 'n')
            to_square: Ending square (row, col)
            promotion: Optional promotion piece type ('q', 'r', 'b', 'n')
            check: Whether the move results in check
            checkmate: Whether the move results in checkmate
            castle: Whether the move is castling
            capture: Whether the move is a capture
            from_square: Starting square (row, col) - needed for pawn captures to include file of origin
            disambiguation: String for disambiguating moves (e.g., 'a', 'b1')
        Returns:
            Move in standard chess notation (e.g., 'e4', 'Nf3', 'O-O', 'Bxc5', 'exd5')
        """
        if castle:
            # Kingside castle (king to g-file)
            if to_square[1] == 6:
                notation = "O-O"
            # Queenside castle (king to c-file)
            else:
                notation = "O-O-O"
            
            if checkmate:
                notation += '#'
            elif check:
                notation += '+'
            return notation
        else:
            piece_type = self.pieces.get_type(piece)
            to_notation = self.coords.matrix_to_chess(to_square)
            
            # Build the main notation
            if piece_type == 'p':
                # Pawn notation
                if capture:
                    # For pawn captures, include the file of origin (e.g., 'exd5')
                    if from_square is not None:
                        from_file = chr(ord('a') + from_square[1])
                        notation = f"{from_file}x{to_notation}"
                    else:
                        notation = f"x{to_notation}"
                else:
                    notation = to_notation
            else:
                # Piece notation (N, B, R, Q, K)
                piece_letter = piece_type.upper()
                capture_marker = 'x' if capture else ''
                # Add disambiguation if multiple pieces of same type can move to same square
                notation = f"{piece_letter}{disambiguation}{capture_marker}{to_notation}"
            
            # Add promotion
            if promotion:
                notation += f"={promotion.upper()}"
            
            # Add check/checkmate
            if checkmate:
                notation += '#'
            elif check:
                notation += '+'
            
            return notation
        
    def stockfish_notation(self, from_square: Tuple[int, int], to_square: Tuple[int, int], promotion: Optional[str] = None) -> str:
        """
        Generate Stockfish-compatible move notation (e.g., 'e2e4', 'e7e8q').
        
        Args:
            from_square: Starting square (row, col)
            to_square: Ending square (row, col)
            promotion: Optional promotion piece type ('q', 'r', 'b', 'n')
        Returns:
            Move in Stockfish notation
        """
        from_notation = self.coords.matrix_to_chess(from_square)
        to_notation = self.coords.matrix_to_chess(to_square)
        promotion_str = promotion.lower() if promotion else ""
        return f"{from_notation}{to_notation}{promotion_str}"
    
    def format_pgn(self, moves: List[str]) -> str:
        """
        Format a list of moves into proper PGN format with move numbers.
        
        Args:
            moves: List of chess notation moves (e.g., ['e4', 'c5', 'Nf3', ...])
            
        Returns:
            Properly formatted PGN string (e.g., '1. e4 c5 2. Nf3 d6 ...')
        """
        if not moves:
            return ""
        
        pgn_parts = []
        for i, move in enumerate(moves):
            # Add move number before white's move (every 2 moves)
            if i % 2 == 0:
                move_num = (i // 2) + 1
                pgn_parts.append(f"{move_num}.")
            pgn_parts.append(move)
        
        return " ".join(pgn_parts)
    
    def generate_full_pgn(self) -> str:
        """
        Generate complete PGN with headers and moves.
        
        Returns:
            Full PGN string with headers and moves
        """
        headers = [
            f'[Event "{database.pgn_event}"]',
            f'[Date "{database.pgn_date}"]',
            f'[Round "{database.pgn_round}"]',
            f'[White "{database.pgn_white}"]',
            f'[Black "{database.pgn_black}"]',
            f'[Result "{database.pgn_result}"]',
            f'[WhiteElo "{database.pgn_white_elo}"]',
            f'[BlackElo "{database.pgn_black_elo}"]',
            f'[TimeControl "{database.pgn_time_control}"]',
            f'[Termination "{database.pgn_termination}"]',
        ]
        
        moves_string = self.format_pgn(database.game_pgn)
        
        # Add result suffix if there are moves
        if moves_string:
            moves_string += f" {database.pgn_result}"
        
        full_pgn = "\n".join(headers) + "\n\n" + moves_string
        return full_pgn
            

            
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
        database.gamelogger.game(f"Fullscreen: {not current_state}")
    
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


# ==================== AI UTILITIES CLASS ====================
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

@dataclass
class StockfishConfig:
    """Configuration for Stockfish strength at different ELO levels"""
    elo: int
    skill_level: int
    depth: int
    multi_pv: int
    blunder_chance: float  # 0.0 to 1.0
    move_weights: List[int]  # Weights for top N moves
    move_time_ms: Optional[int] = None  # Optional time limit
    
    @property
    def description(self) -> str:
        """Human-readable description"""
        if self.elo < 600:
            return "Absolute Beginner"
        elif self.elo < 800:
            return "Novice"
        elif self.elo < 1000:
            return "Beginner"
        elif self.elo < 1200:
            return "Casual Player"
        elif self.elo < 1400:
            return "Club Player"
        elif self.elo < 1600:
            return "Intermediate"
        elif self.elo < 1800:
            return "Advanced"
        elif self.elo < 2000:
            return "Strong Player"
        elif self.elo < 2200:
            return "Expert"
        elif self.elo < 2400:
            return "Master"
        elif self.elo < 2600:
            return "International Master"
        elif self.elo < 2800:
            return "Grandmaster"
        else:
            return "Super GM"


# ==================== ELO CONFIGURATIONS ====================

ELO_CONFIGS = {
    400: StockfishConfig(
        elo=400,
        skill_level=0,
        depth=1,
        multi_pv=10,
        blunder_chance=0.40,  # 40% chance of blunder
        move_weights=[20, 15, 15, 12, 10, 8, 7, 5, 4, 4],
        move_time_ms=50
    ),
    600: StockfishConfig(
        elo=600,
        skill_level=0,
        depth=2,
        multi_pv=8,
        blunder_chance=0.30,  # 30% chance of blunder
        move_weights=[25, 20, 15, 12, 10, 8, 5, 5],
        move_time_ms=100
    ),
    800: StockfishConfig(
        elo=800,
        skill_level=0,
        depth=3,
        multi_pv=5,
        blunder_chance=0.15,  # 15% chance of blunder
        move_weights=[40, 25, 15, 15, 5],
        move_time_ms=150
    ),
    1000: StockfishConfig(
        elo=1000,
        skill_level=0,
        depth=5,
        multi_pv=5,
        blunder_chance=0.10,  # 10% chance of blunder
        move_weights=[50, 25, 15, 7, 3]
    ),
    1200: StockfishConfig(
        elo=1200,
        skill_level=1,
        depth=8,
        multi_pv=4,
        blunder_chance=0.05,  # 5% chance of blunder
        move_weights=[60, 25, 10, 5]
    ),
    1400: StockfishConfig(
        elo=1400,
        skill_level=3,
        depth=10,
        multi_pv=3,
        blunder_chance=0.02,  # 2% chance of blunder
        move_weights=[70, 20, 10]
    ),
    1600: StockfishConfig(
        elo=1600,
        skill_level=5,
        depth=12,
        multi_pv=3,
        blunder_chance=0.01,  # 1% chance of blunder
        move_weights=[80, 15, 5]
    ),
    1800: StockfishConfig(
        elo=1800,
        skill_level=4,
        depth=15,
        multi_pv=2,
        blunder_chance=0.005,  # 0.5% chance of blunder
        move_weights=[90, 10]
    ),
    2000: StockfishConfig(
        elo=2000,
        skill_level=14,
        depth=8,
        multi_pv=2,
        blunder_chance=0.0,
        move_weights=[95, 5]
    ),
    2200: StockfishConfig(
        elo=2200,
        skill_level=12,
        depth=8,
        multi_pv=1,
        blunder_chance=0.0,
        move_weights=[100]
    ),
    2400: StockfishConfig(
        elo=2400,
        skill_level=15,
        depth=9,
        multi_pv=1,
        blunder_chance=0.0,
        move_weights=[100]
    ),
    2600: StockfishConfig(
        elo=2600,
        skill_level=18,
        depth=10,
        multi_pv=1,
        blunder_chance=0.0,
        move_weights=[100]
    ),
    2800: StockfishConfig(
        elo=2800,
        skill_level=20,
        depth=11,
        multi_pv=1,
        blunder_chance=0.0,
        move_weights=[100]
    ),
    3000: StockfishConfig(
        elo=3000,
        skill_level=20,
        depth=15,
        multi_pv=1,
        blunder_chance=0.0,
        move_weights=[100]
    ),
}


def get_config_for_elo(elo: int) -> StockfishConfig:
    """
    Get the appropriate configuration for a given ELO.
    
    Args:
        elo: Desired ELO rating
        
    Returns:
        StockfishConfig for that level
    """
    # Find closest predefined config
    available_elos = sorted(ELO_CONFIGS.keys())
    
    # Exact match
    if elo in ELO_CONFIGS:
        return ELO_CONFIGS[elo]
    
    # Find nearest config
    closest_elo = min(available_elos, key=lambda x: abs(x - elo))
    return ELO_CONFIGS[closest_elo]


class AIUtilities:
    """AI and Stockfish management utilities"""
    
    def __init__(self):
        self.current_config: Optional[StockfishConfig] = None
    
    def add_stockfish(self) -> None:
        """
        Initialize and load the Stockfish chess engine.
        """
        import os
        if os.name == 'posix':
            path = "/usr/games/stockfish"
            if os.path.exists(path):
                stockfish = Stockfish(path)
                database.stockfish = stockfish
                database.gamelogger.ai("Stockfish Added" if stockfish else "Failed to add Stockfish")
            else:
                database.gamelogger.error("Stockfish not found!")
        else:
            path = "stockfish/stockfish-windows-x86-64-avx2.exe"
            if os.path.exists(path):
                stockfish = Stockfish(self.resource_path(path))
                database.stockfish = stockfish
                database.gamelogger.ai("Stockfish Added" if stockfish else "Failed to add Stockfish")
            else:
                database.gamelogger.error("Stockfish not found!")
    
    def configure_strength(self, elo: int) -> None:
        """
        Configure Stockfish strength for the given ELO.
        
        Args:
            elo: Desired ELO rating (400-3000)
        """
        if not database.stockfish:
            database.gamelogger.ai("Stockfish not initialized!")
            return
        
        # Get configuration for this ELO
        self.current_config = get_config_for_elo(elo)
        config = self.current_config
        
        # Apply base configuration
        database.stockfish.set_skill_level(config.skill_level)
        database.stockfish.set_depth(config.depth)
        
        # Disable UCI_LimitStrength (we use Multi-PV instead)
        database.stockfish.update_engine_parameters({
            "UCI_LimitStrength": False,
            "MultiPV": 1  # Will be updated per move
        })
        
        database.gamelogger.ai(f"Configured: {config.description} ({config.elo} ELO) | Skill: {config.skill_level} | Depth: {config.depth} | Blunder: {config.blunder_chance*100:.0f}%")
    
    def get_ai_move(self) -> Optional[str]:
        """
        Get a move from Stockfish appropriate for the configured ELO.
        
        Uses Multi-PV and weighted randomness to simulate human-like play.
        
        Returns:
            Move in UCI notation or None
        """
        if not database.stockfish or not self.current_config:
            return None
        
        config = self.current_config
        
        try:
            database.ai_total_moves += 1

            # Check for blunder
            if config.blunder_chance > 0:
                if random.random() < config.blunder_chance:
                    database.ai_blunder_moves += 1

                    pct = (database.ai_blunder_moves / database.ai_total_moves) * 100
                    database.gamelogger.ai(f"Blundering! ({database.ai_blunder_moves}/{database.ai_total_moves} = {pct:.0f}%)")
                    return self._get_blunder_move()
            
            # Get move based on configuration
            if config.multi_pv == 1:
                # Strong players: Always best move
                return self._get_best_move(config)
            else:
                # Weaker players: Weighted random from top moves
                return self._get_weighted_move(config)
        
        except Exception as e:
            database.gamelogger.error(f"Stockfish error: {e}")
            return None
    
    def _get_best_move(self, config: StockfishConfig) -> Optional[str]:
        """
        Get the best move (for strong players).
        
        Args:
            config: Current Stockfish configuration
            
        Returns:
            Best move in UCI notation
        """
        if database.stockfish is None:
            return
        
        if config.move_time_ms:
            return database.stockfish.get_best_move_time(config.move_time_ms)
        else:
            return database.stockfish.get_best_move()
    
    def _get_weighted_move(self, config: StockfishConfig) -> Optional[str]:
        """
        Get a move from top N moves using weighted random selection.
        
        Args:
            config: Current Stockfish configuration
            
        Returns:
            Selected move in UCI notation
        """
        if database.stockfish is None:
            return
        
        # Enable Multi-PV
        database.stockfish.update_engine_parameters({"MultiPV": config.multi_pv})
        
        # Get top moves
        if config.move_time_ms:
            database.stockfish.get_best_move_time(config.move_time_ms)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                top_moves = database.stockfish.get_top_moves(config.multi_pv)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                top_moves = database.stockfish.get_top_moves(config.multi_pv)
        
        if not top_moves or len(top_moves) == 0:
            return database.stockfish.get_best_move()
        
        # Extract moves
        moves = [move_info['Move'] for move_info in top_moves]
        
        # Ensure weights match available moves
        weights = config.move_weights[:len(moves)]
        
        # Normalize weights if necessary
        weight_sum = sum(weights)
        if weight_sum != 100:
            weights = [w * 100 / weight_sum for w in weights]
        
        # Weighted random selection
        selected_move = random.choices(moves, weights=weights)[0]
        
        # Debug output
        if len(moves) > 1:
            database.gamelogger.ai(f"Pick #{moves.index(selected_move) + 1}/{len(moves)}")
        
        return str(selected_move) if selected_move else None
    
    def _get_blunder_move(self) -> Optional[str]:
        """
        Get a random legal move (simulates a blunder).
        
        Returns:
            Random move in UCI notation
        """
        if database.stockfish is None:
            return
        
        # Get many legal moves
        database.stockfish.update_engine_parameters({"MultiPV": 20})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            all_moves = database.stockfish.get_top_moves(20)
        
        if not all_moves or len(all_moves) < 5:
            # Not enough moves, just return best move
            return database.stockfish.get_best_move()
        
        # Pick a random move from the bottom half (bad moves)
        bad_moves_start = len(all_moves) // 2
        bad_moves = all_moves[bad_moves_start:]
        
        selected = random.choice(bad_moves)
        return str(selected['Move']) if selected['Move'] else None
    
    @staticmethod
    def update_start_fen() -> None:
        """Reset engine to starting position and apply all moves from game history."""
        if database.stockfish:
            database.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            if database.game_history:
                database.stockfish.make_moves_from_current_position(database.game_history)
    
    @staticmethod
    def update_last_move() -> None:
        """Apply the most recent move to the Stockfish engine."""
        if database.stockfish and database.game_history:
            database.stockfish.make_moves_from_current_position([database.game_history[-1]])
    
    @staticmethod
    def set_fen(fen: str) -> None:
        """Set the board position using a FEN string."""
        if database.stockfish and database.stockfish.is_fen_valid(fen):
            database.stockfish.set_fen_position(fen)
    
    @staticmethod
    def resource_path(relative_path):
        """Get the absolute path to a resource file."""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path) #type: ignore
        return os.path.join(os.path.abspath("."), relative_path)

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
        self.notations = NotationGenerator()
        self.pieces = PieceUtilities()
        self.ai = AIUtilities()
    
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
        return self.notations.generate_fen(**kwargs)
    
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

    def create_selector_frame(self, parent, label_text: str, placeholder_text: str, 
                         values: List, command=None, width: int = 400, height: int = 170):
        """
        Create a styled selector frame with label and combobox.
        
        Args:
            parent: The parent widget to place this frame in
            label_text: Text for the label (e.g., "Level:", "Character:", etc.)
            placeholder_text: Placeholder text for the combobox (e.g., "Select Difficulty")
            values: List of values for the combobox dropdown
            command: Optional callback function when selection changes
            width: Frame width (default 400)
            height: Frame height (default 150)
        
        Returns:
            tuple: (main_frame, combobox) - The frame and combobox widget for further customization
        """
        
        # Create main container frame with golden border
        main_frame = ctk.CTkFrame(
            parent,
            fg_color="#2a2420",
            border_color="#c9a961",
            border_width=3,
            corner_radius=8,
            width=width,
            height=height
        )
        main_frame.pack_propagate(False)
        
        # Add padding inside the frame
        content_frame = ctk.CTkFrame(
            main_frame,
            fg_color="transparent"
        )
        content_frame.pack(fill="both", expand=True, padx=int(width*0.05), pady=int(height*0.13))
        
        # Label with serif-like font
        label_font_size = max(16, min(int(height * 0.2), 28))
        label = ctk.CTkLabel(
            content_frame,
            text=label_text,
            font=("Georgia", label_font_size, "bold"),
            text_color="#c9a961",
            anchor="w"
        )
        label.pack(pady=(0, int(height*0.1)), anchor="w")
        
        # Selector frame (for the inner border effect)
        selector_frame = ctk.CTkFrame(
            content_frame,
            fg_color="#0a0806",
            border_color="#c9a961",
            border_width=2,
            corner_radius=5
        )
        selector_frame.pack(fill="both", expand=True)
        
        # Combobox for selection
        combo_font_size = max(12, min(int(height * 0.12), 16))
        combo_height = max(30, min(int(height * 0.3), 45))
        
        main_combo = ctk.CTkComboBox(
            selector_frame,
            values=values,
            state="readonly",
            height=combo_height,
            font=("Arial", combo_font_size),
            fg_color="#1a1410",
            button_color="#4a4440",
            button_hover_color="#5a5450",
            border_color="#c9a961",
            dropdown_fg_color="#2a2420",
            dropdown_hover_color="#3a3430",
            text_color="#a0a0a0",
            dropdown_text_color="#c9a961",
            corner_radius=5,
            justify="center",
        )
        
        # Now define the callback that uses main_combo
        def on_selection_change(choice):
            # Change text color when selected
            main_combo.configure(text_color="#c9a961")
            # Call user's callback if provided
            if command:
                command(choice)
        
        main_combo.configure(command=on_selection_change)
        main_combo.set(placeholder_text)
        main_combo.pack(fill="both", padx=3, pady=3)
        
        return main_frame, main_combo
    
    def get_formatted_pgn(self) -> str:
        """
        Get the current game as a formatted PGN string with move numbers.
        
        Returns:
            Formatted PGN string (e.g., '1. e4 c5 2. Nf3 d6 ...')
        """
        return self.notations.format_pgn(database.game_pgn)
    
    def get_full_pgn(self) -> str:
        """
        Get the complete PGN with headers and moves.
        
        Returns:
            Full PGN string with headers and moves
        """
        return self.notations.generate_full_pgn()
    
    def export_game_pgn(self, filename: Optional[str] = None) -> str:
        """
        Export the current game to a PGN file.
        
        Args:
            filename: Optional filename (default: chess_game_TIMESTAMP.pgn)
            
        Returns:
            Path to the created file
        """
        import datetime
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chess_game_{timestamp}.pgn"
        
        pgn_content = self.get_formatted_pgn()
        fen = self.create_fen()
        
        # Add PGN headers
        pgn_with_headers = f"""[Event "Chess Game"]
[Site "Local"]
[Date "{datetime.datetime.now().strftime('%Y.%m.%d')}"]
[Round "1"]
[White "Player"]
[Black "Player"]
[Result "*"]
[FEN "{fen}"]

{pgn_content}
"""
        
        with open(filename, 'w') as f:
            f.write(pgn_with_headers)
        
        database.gamelogger.game(f"Game exported to {filename}")
        return filename


# ==================== MODULE TEST ====================

if __name__ == "__main__":
    utils = Utilities()
    print(utils.notations.generate_matrix("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
