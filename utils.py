# utils.py - Complete refactored version

import tkinter as tk
import customtkinter as ctk
from typing import Tuple, Dict, List, Optional
import numpy as np
from database.database import database
from PIL import Image


class Utilities:
    """Utility class for window management and layout calculations"""

    def __init__(self):
        self.legal_moves = LegalMoves()

    def fullscreen_window(self, window: tk.Tk | ctk.CTk) -> None:
        """Set window to fullscreen mode"""
        window.attributes("-fullscreen", True)
        window.update_idletasks()

    def fullscreen_toggle(self, window: tk.Tk | ctk.CTk) -> None:
        """Toggle fullscreen state"""
        window.attributes("-fullscreen", not window.attributes("-fullscreen"))
        print(f"Fullscreen: {window.attributes('-fullscreen')}\n")

    def relative_dimensions(self, rely: float, dimensions: Tuple[int, int]) -> float:
        """
        Calculate relative x position to center a square board
        
        Args:
            rely: Relative y position (0.0 to 1.0)
            dimensions: Tuple of (height, width)
            
        Returns:
            relx value to center the square board
        """
        height, width = dimensions
        real_y = rely * height
        
        # Height available for the square board
        h = height - real_y * 2
        
        # Calculate relx to center the square
        relx = (width - h) / (2 * width)
        
        return relx

    def ctkimage_generator(self, path: str, size: Tuple[int, int] = (70, 70)) -> ctk.CTkImage:
        """
        Generate CTkImage from path with transparency support
        
        Args:
            path: Path to the image file
            size: Tuple of (width, height) in pixels
            
        Returns:
            CTkImage object
        """
        img = Image.open(path)
        
        # Ensure image has alpha channel for transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    
    import numpy as np

    def create_fen(self,
        board=database.matrix,
        side_to_move="w",
        halfmove_clock=0,
        fullmove_number=1
    ):
        """
        Converts an 8x8 NumPy ndarray into a FEN string.

        Board encoding:
        - 0        -> empty square
        - 'r1'     -> white rook
        - 'k2'     -> black king
        """

        piece_map = {
            "p": "P",
            "r": "R",
            "n": "N",
            "b": "B",
            "q": "Q",
            "k": "K",
            "-p": "p",
            "-r": "r",
            "-n": "n",
            "-b": "b",
            "-q": "q",
            "-k": "k",
        }

        castling=""
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

        en_passant = database.en_passant if database.en_passant else "-"

        fen_rows = []

        for row in board:
            empty = 0
            fen_row = ""

            for cell in row:
                # NumPy-safe empty check
                if cell == 0:
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0

                    cell = str(cell)  # safety for ndarray dtype
                    piece_type = cell[:-1].lower()

                    fen_piece = piece_map[piece_type]

                    fen_row += fen_piece

            if empty > 0:
                fen_row += str(empty)

            fen_rows.append(fen_row)

        board_fen = "/".join(fen_rows)
        return f"{board_fen} {side_to_move} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    
    def matrix_to_chess(self, square: Tuple[int, int]) -> str:
        """
        Converts matrix coordinates (row, col) to chess notation (e.g. e4).

        Assumes:
        - matrix[0][0] = a8
        - matrix[7][7] = h1
        """
        row, col = square

        file = chr(ord('a') + col)
        rank = str(8 - row)

        return file + rank
    
    def next_piece(self, base: str, color: str) -> str:
        """Returns the next piece for a given color and starting piece type."""
        maximum = 0
        prefix = "-" if color == "black" else ""

        pieces = database.white_pieces if color == "white" else database.black_pieces
        for piece in pieces.flatten():
            if base in piece:
                maximum = max(maximum, int(piece[-1]))
        
        return f"{prefix}{base}{maximum + 1}"
    
    def reset(self):
        self.legal_moves = LegalMoves()

class LegalMoves:
    """Class for calculating legal chess moves for all piece types"""
    
    def __init__(self):
        """Initialize and calculate legal moves for starting position"""

        self._last_matrix_hash = None  # NEW: Track when matrix changes
        self.update_legal_moves(database.matrix)

    # ==================== UTILITY METHODS ====================

    def search_piece(self, piece: str, matrix: np.ndarray = database.matrix) -> Tuple[int, int]:
        """
        Find position of a piece on the board
        
        Args:
            piece: Piece identifier (e.g., 'k1', '-r2')
            matrix: 8x8 numpy array representing the board
            
        Returns:
            Tuple of (row, col) position
            
        Raises:
            ValueError: If piece not found on board
        """
        pos = np.where(matrix == piece)
        if len(pos[0]) == 0:
            raise ValueError(f"{piece} not found on board")
        return int(pos[0][0]), int(pos[1][0])

    def is_occupied(self, val) -> bool:
        """Check if a board position contains a piece"""
        return val != 0
    
    def is_enemy(self, val, color: str) -> bool:
        """
        Check if a piece belongs to the opponent
        
        Args:
            val: Board value (piece identifier or 0)
            color: Player color ("white" or "black")
            
        Returns:
            True if piece belongs to opponent
        """
        if val == 0:
            return False
        piece_is_black = '-' in str(val)
        return piece_is_black != (color == "black")

    # ==================== ATTACK AND PIN DETECTION ====================

    def opponent_legal_search(
        self, 
        color: str, 
        coordinates: Tuple[int, int], 
        matrix: np.ndarray = database.matrix, 
        return_piece: bool = False
    ) -> bool | List[str]:
        """
        Check if opponent can attack a square
        
        Args:
            color: Current player color
            coordinates: Square to check (row, col)
            matrix: Current board state
            return_piece: If True, return list of attacking pieces
            
        Returns:
            True if square is under attack, or list of attacking pieces
        """
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

    def can_attack_in_direction(self, piece: str, direction: Tuple[int, int]) -> bool:
        """
        Check if a piece can attack along a direction
        
        Args:
            piece: Piece identifier
            direction: Direction vector (dr, dc)
            
        Returns:
            True if piece can attack in this direction
        """
        piece_type = piece.strip('-')[0]
        dr, dc = direction
        
        is_straight = (dr == 0 or dc == 0)
        is_diagonal = (abs(dr) == abs(dc) and dr != 0)
        
        if piece_type == 'r':
            return is_straight
        elif piece_type == 'b':
            return is_diagonal
        elif piece_type == 'q':
            return is_straight or is_diagonal
        else:
            return False

    def find_pins(self, color: str, matrix: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Find all pinned pieces for a color
        
        Args:
            color: Player color to check
            matrix: Current board state
            
        Returns:
            Dictionary mapping pinned pieces to pin direction
        """
        pins = {}
        king = "k1" if color == "white" else "-k1"
        king_row, king_col = self.search_piece(king, matrix)
        
        # 8 directions: 4 straight + 4 diagonal
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dr, dc in directions:
            friendly_piece = None
            row, col = king_row + dr, king_col + dc
            
            while 0 <= row < 8 and 0 <= col < 8:
                piece = matrix[row, col]
                
                if piece != 0:
                    if not self.is_enemy(piece, color):
                        if friendly_piece is None:
                            friendly_piece = (piece, row, col)
                        else:
                            break
                    else:
                        if friendly_piece is not None:
                            if self.can_attack_in_direction(piece, (dr, dc)):
                                pins[friendly_piece[0]] = (dr, dc)
                        break
                
                row += dr
                col += dc
        
        return pins

    # ==================== CHECK DETECTION ====================

    def check_checker(self, color: str, matrix: np.ndarray = database.matrix) -> bool:
        """
        Check if the king is in check
        
        Args:
            color: Player color to check
            matrix: Current board state
            
        Returns:
            True if king is in check
        """
        king = "k1" if color == "white" else "-k1"
        return self.opponent_legal_search(color, self.search_piece(king, matrix)) #type: ignore

    def check_legal(self, color: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]] | bool:
        """
        Get squares that can block or capture the checking piece
        
        Args:
            color: Player color in check
            matrix: Current board state
            
        Returns:
            List of squares that resolve check, or True if not in check
        """
        king = "k1" if color == "white" else "-k1"
        king_coordinates = self.search_piece(king, matrix)
        pieces = self.opponent_legal_search(color, king_coordinates, matrix, return_piece=True)

        if not pieces or isinstance(pieces, bool):
            return True
        
        if len(pieces) >= 2:
            return []  # Double check - only king can move
        
        piece = pieces[0]
        piece_pos = self.search_piece(piece, matrix)
        
        # For knights and pawns, only capturing them resolves check
        if 'n' in piece or 'p' in piece:
            return [piece_pos]
        
        # For sliding pieces, can block or capture
        return self._calculate_blocking_squares(piece, piece_pos, king_coordinates, matrix)

    def _calculate_blocking_squares(
        self, 
        piece: str, 
        piece_pos: Tuple[int, int], 
        king_pos: Tuple[int, int],
        matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Calculate squares that block a check from a sliding piece"""
        piece_type = piece.strip('-')[0]
        
        if piece_type == 'r':
            return self._rook_blocking_squares(piece_pos, king_pos)
        elif piece_type == 'b':
            return self._bishop_blocking_squares(piece_pos, king_pos)
        elif piece_type == 'q':
            # Queen can attack like rook or bishop
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
        color: str, 
        matrix: np.ndarray = database.matrix
    ) -> List[Tuple[int, int]]:
        """
        Filter moves to only those that resolve check
        
        Args:
            moves: List of pseudo-legal moves
            color: Player color
            matrix: Current board state
            
        Returns:
            List of legal moves that resolve check
        """
        if not self.check_checker(color, matrix):
            return moves
        
        allowed_moves = self.check_legal(color, matrix)
        
        if allowed_moves is True:
            return moves
        
        if not allowed_moves:
            return []
        
        return [move for move in moves if move in allowed_moves]

    # ==================== CASTLING ====================

    def can_castle(self, color: str, castle_range: range, matrix: np.ndarray = database.matrix) -> bool:
        """
        Check if castling is possible
        
        Args:
            color: Player color
            castle_range: Column range to check
            matrix: Current board state
            
        Returns:
            True if castling path is clear and safe
        """
        opponent_color = "white" if color == "black" else "black"
        row = 0 if color == "black" else 7
        
        for col in castle_range:
            if self.is_occupied(matrix[row, col]):
                return False
            if self.opponent_legal_search(opponent_color, (row, col)):
                return False
        
        return True

    # ==================== PIECE MOVE CALCULATION ====================

    def rook_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal rook moves"""
        # Check if pinned diagonally
        if piece in database.pins:
            direction = database.pins[piece]
            if direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                return []
        else:
            direction = None
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        # Horizontal moves
        if not direction or direction in [(0, -1), (0, 1)]:
            # Left
            for c in range(col - 1, -1, -1):
                if self.is_occupied(matrix[row, c]):
                    if self.is_enemy(matrix[row, c], color):
                        moves.append((row, c))
                    break
                moves.append((row, c))
            
            # Right
            for c in range(col + 1, 8):
                if self.is_occupied(matrix[row, c]):
                    if self.is_enemy(matrix[row, c], color):
                        moves.append((row, c))
                    break
                moves.append((row, c))
        
        # Vertical moves
        if not direction or direction in [(1, 0), (-1, 0)]:
            # Up
            for r in range(row - 1, -1, -1):
                if self.is_occupied(matrix[r, col]):
                    if self.is_enemy(matrix[r, col], color):
                        moves.append((r, col))
                    break
                moves.append((r, col))
            
            # Down
            for r in range(row + 1, 8):
                if self.is_occupied(matrix[r, col]):
                    if self.is_enemy(matrix[r, col], color):
                        moves.append((r, col))
                    break
                moves.append((r, col))
        
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves

    def bishop_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal bishop moves"""
        # Check if pinned horizontally or vertically
        if piece in database.pins:
            direction = database.pins[piece]
            if direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                return []
        else:
            direction = None
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        # Four diagonal directions
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in diagonals:
            # Skip if pinned in different diagonal
            if direction and direction != (dr, dc):
                # Allow opposite direction on same diagonal
                if direction != (-dr, -dc):
                    continue
            
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                if self.is_occupied(matrix[r, c]):
                    if self.is_enemy(matrix[r, c], color):
                        moves.append((r, c))
                    break
                moves.append((r, c))
                r += dr
                c += dc
        
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves

    def queen_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal queen moves (combination of rook and bishop)"""
        return self.bishop_moves(piece, matrix) + self.rook_moves(piece, matrix)

    def knight_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal knight moves"""
        # Knights cannot move if pinned
        if piece in database.pins:
            return []
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        # All 8 possible L-shaped moves
        knight_offsets = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        
        for dr, dc in knight_offsets:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                if not self.is_occupied(matrix[r, c]) or self.is_enemy(matrix[r, c], color):
                    moves.append((r, c))
        
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves

    # In utils.py, modify the king_moves method:

    def king_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal king moves including castling"""
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        # Find opponent king position
        opponent_king = "-k1" if color == "white" else "k1"
        try:
            opp_king_row, opp_king_col = self.search_piece(opponent_king, matrix)
        except ValueError:
            opp_king_row, opp_king_col = -10, -10  # King not on board (shouldn't happen)
        
        # 8 adjacent squares
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    # ✅ NEW: Check if square is adjacent to opponent king
                    king_distance = max(abs(r - opp_king_row), abs(c - opp_king_col))
                    if king_distance <= 1:
                        continue  # Can't move next to opponent king
                    
                    if not self.is_occupied(matrix[r, c]) or self.is_enemy(matrix[r, c], color):
                        moves.append((r, c))
        
        # Castling logic (keep as is)
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
                
                if matrix[expected_rook_pos[0], expected_rook_pos[1]] == rook_piece:
                    if self.can_castle(color, castle_range, matrix):
                        moves.append((row_castle, castle_col))
        
        # Filter using attack checking (existing code)
        safe_moves = []
        opponent_color = "black" if color == "white" else "white"
        
        for move in moves:
            # Simulate the move
            temp_matrix = matrix.copy()
            temp_matrix[row, col] = 0
            temp_matrix[move[0], move[1]] = piece
            
            # Check if square is attacked by opponent pieces (EXCLUDING kings)
            is_safe = True
            for r in range(8):
                for c in range(8):
                    opponent_piece = temp_matrix[r, c]
                    if opponent_piece == 0:
                        continue
                    
                    # Skip opponent king (kings can't give check)
                    if 'k' in str(opponent_piece).strip('-'):
                        continue
                    
                    # Check if this opponent piece is the right color
                    piece_color = "black" if '-' in str(opponent_piece) else "white"
                    if piece_color != opponent_color:
                        continue
                    
                    # Calculate pseudo-legal moves for this piece
                    if self._can_piece_attack(opponent_piece, (r, c), move, temp_matrix):
                        is_safe = False
                        break
                
                if not is_safe:
                    break
            
            if is_safe:
                safe_moves.append(move)
        
        return safe_moves

    def _can_piece_attack(self, piece: str, from_pos: Tuple[int, int], to_pos: Tuple[int, int], matrix: np.ndarray) -> bool:
        """Check if a piece can attack a square (simple, non-recursive)"""
        piece_type = piece.strip('-')[0]
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

    def pawn_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """Calculate legal pawn moves"""
        # Pawns can only move if not pinned horizontally
        if piece in database.pins:
            direction_ = database.pins[piece]
            if direction_[1] != 0:  # Pinned diagonally or horizontally
                return []
        else:
            direction_ = None
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece, matrix)
        moves = []
        
        if color == "white":
            start_row = 6
            direction = -1
            opponent_last_pawn = database.black_last_pawn
        else:
            start_row = 1
            direction = 1
            opponent_last_pawn = database.white_last_pawn
        
        # Forward moves
        if not direction_ or direction_[1] == 0:
            # Single forward
            if 0 <= row + direction < 8:
                if not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction, col))
                    
                    # Double forward from start
                    if row == start_row:
                        if not self.is_occupied(matrix[row + direction * 2, col]):
                            moves.append((row + direction * 2, col))
        
        # Diagonal captures
        if not direction_ or direction_[1] != 0:
            for dc in [-1, 1]:
                r, c = row + direction, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.is_enemy(matrix[r, c], color):
                        moves.append((r, c))
            
            # En passant
            en_passant_possible = False
            en_passant_row = 3 if color == "white" else 4
            if opponent_last_pawn and row == en_passant_row:
                row_l, col_l = opponent_last_pawn
                if row_l == row and abs(col_l - col) == 1:
                    en_passant_possible = True
                    moves.append((row + direction, col_l))
                    database.en_passant = utils.matrix_to_chess((row + direction, col_l))

            if not en_passant_possible:
                database.en_passant = ""
        
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        
        return moves

    # ==================== MAIN CALCULATION METHODS ====================

    def calculate_legal_moves(self, piece: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]]:
        """
        Calculate legal moves for any piece
        
        Args:
            piece: Piece identifier
            matrix: Current board state
            
        Returns:
            List of legal move coordinates
        """
        piece_type = piece.strip('-')[0]
        
        move_functions = {
            'p': self.pawn_moves,
            'r': self.rook_moves,
            'n': self.knight_moves,
            'b': self.bishop_moves,
            'q': self.queen_moves,
            'k': self.king_moves
        }
        
        return move_functions.get(piece_type, lambda p, m: [])(piece, matrix)

    def all_legal_moves(self, color: str, matrix: np.ndarray = database.matrix) -> Dict[str, np.ndarray]:
        """
        Calculate legal moves for all pieces of a color
        
        Args:
            color: Player color ("white" or "black")
            matrix: Current board state
            
        Returns:
            Dictionary mapping pieces to their legal moves
        """
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

    def update_legal_moves(self, matrix: np.ndarray = database.matrix) -> None:
        """
        Update all legal moves for both colors (only if matrix changed)
        """
        # Calculate hash of current matrix state
        current_hash = hash(matrix.tobytes())
        
        # Skip if matrix hasn't changed
        if current_hash == self._last_matrix_hash:
            return
        
        self._last_matrix_hash = current_hash
        
        # Find pins
        white_pins = self.find_pins("white", matrix)
        black_pins = self.find_pins("black", matrix)
        database.pins = {**white_pins, **black_pins}
        
        # FIX: Initialize ALL pieces including promoted ones
        # Get all unique pieces from the matrix
        all_white_pieces = set(database.white_pieces.flatten())
        all_black_pieces = set(database.black_pieces.flatten())
        
        database.white_legal_moves = {piece: np.array([]) for piece in all_white_pieces}
        database.black_legal_moves = {piece: np.array([]) for piece in all_black_pieces}
        
        # Calculate actual legal moves
        if database.current_turn == "white":
            database.black_legal_moves = self.all_legal_moves("black", matrix)
            database.white_legal_moves = self.all_legal_moves("white", matrix)
        else:
            database.white_legal_moves = self.all_legal_moves("white", matrix)
            database.black_legal_moves = self.all_legal_moves("black", matrix)
        
        print("Legal Moves Updated!\n")


if __name__ == "__main__":
    utils = Utilities()
    fen = utils.create_fen()
    print(fen)