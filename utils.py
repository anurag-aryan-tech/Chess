import tkinter as tk
import customtkinter as ctk
from typing import Tuple, Dict, List
import numpy as np
from database.database import database
from PIL import Image, ImageTk


class Utilities:
    """Utility class containing helper methods for window management and layout calculations."""

    def __init__(self):
        self.legal_moves = LegalMoves()
    def fullscreen_window(self, window: tk.Tk|ctk.CTk) -> None:
        """
        Sets the given window to fullscreen mode.
        
        Args:
            window: The tkinter or customtkinter window to make fullscreen.
        """
        window.attributes("-fullscreen", True)
        window.update_idletasks()

    def fullscreen_toggle(self, window: tk.Tk|ctk.CTk):
        """
        Toggles the fullscreen state of the given window.
        
        Args:
            window: The tkinter or customtkinter window to toggle fullscreen for.
        """
        window.attributes("-fullscreen", not window.attributes("-fullscreen"))
        print(f"Fullscreen: {True if window.attributes('-fullscreen') else False}\n")

    def relative_dimensions(self, rely: float, dimensions: Tuple[int, int]) -> float:
        """
        Calculates relative x position (relx) to center a square board based on given relative y position.
        
        Args:
            rely: Relative y position (0.0 to 1.0) for top/bottom margins.
            dimensions: Tuple of (height, width) of the window in pixels.
            
        Returns:
            relx value to center the square board horizontally.
        """
        height, width = dimensions
        real_y = rely * height
        
        # Height available for the square board
        h = height - real_y * 2
        
        # Width the square board will occupy
        square_width = h  # Since it's a square
        
        # Calculate relx to center the square
        relx = (width - square_width) / (2 * width)
        
        return relx

    def ctkimage_generator(self, path: str, size: Tuple[int, int] = (70, 70)) -> ctk.CTkImage:
        """
        Generate CTkImage from path with transparency support
        
        Args:
            path: Path to the image file.
            size: Tuple of (width, height) in pixels (optional).
            
        Returns:
            PhotoImage object.
        """
        from PIL import Image
        
        img = Image.open(path)
        # Ensure image has alpha channel for transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    

class LegalMoves:
    """Class containing methods to calculate legal chess moves for all piece types."""
    
    def __init__(self):
        self.update_legal_moves(database.matrix)
    def search_piece(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Finds the position of a specific piece on the chess board matrix.
        
        Args:
            piece: Piece identifier (e.g., 'k1', '-r2').
            matrix: 8x8 numpy array representing the chess board.
            
        Returns:
            Tuple of (row, col) position of the piece.
            
        Raises:
            ValueError: If piece not found on board.
        """
        pos = np.where(matrix == piece)
        if len(pos[0]) == 0:
            raise ValueError(f"{piece} not found on board")
        return int(pos[0][0]), int(pos[1][0])

    def is_occupied(self, val): 
        """Checks if a board position contains any piece (occupied)."""
        return val != 0
    
    def is_enemy(self, val, color: str):
        """
        Determines if a piece belongs to the enemy of the given color.
        
        Args:
            val: Board value (piece identifier or 0).
            color: Color of the player ("white" or "black").
            
        Returns:
            True if the piece belongs to the opponent.
        """
        if val == 0:
            return False
        piece_is_black = '-' in str(val)
        return piece_is_black != (color == "black")
    
    def opponent_legal_search(self, color: str, coordinates: Tuple[int, int], matrix: np.ndarray = database.matrix, return_piece: bool = False):
        """
        Checks if opponent can legally move to the given coordinates.
        
        Args:
            color: Color of the current player ("white" or "black").
            coordinates: Tuple of (row, col) to check.
            matrix: Current board state.
            
        Returns:
            True if opponent has a legal move to these coordinates.
        """
        
        moves = database.white_legal_moves.items() if color == "black" else database.black_legal_moves.items()
        pieces = []
        for piece, move in moves:
            # FIX: Handle empty arrays
            if len(move) == 0:  # Skip pieces with no moves
                continue
            # FIX: Use numpy comparison properly
            if np.any(np.all(move == coordinates, axis=1)):  # CHANGED THIS LINE
                if return_piece:
                    pieces.append(piece)
                else:
                    return True

        if return_piece:
            return pieces
        return False
    
    def can_castle(self, color: str, range: range, matrix: np.ndarray = database.matrix):
        """
        Checks if castling is possible through the given column range.
        
        Args:
            color: Player color attempting to castle ("white" or "black").
            range: Column range to check for clear path.
            matrix: Current board state.
            
        Returns:
            True if path is clear and not under attack.
        """
        opponent_color = "white" if color == "black" else "black"
        clear = True
        check = False
        row = 0 if color == "black" else 7
        for i in range:
            if self.is_occupied(matrix[row, i]):
                clear = False
                break
            if self.opponent_legal_search(opponent_color, (row, i)):
                check = True
                break

        return clear and not check
    
    def can_attack_in_direction(self, piece: str, direction: Tuple[int, int]) -> bool:
        """
        Check if a piece can attack along a given direction.
        
        Args:
            piece: The piece (e.g., "-r1", "q1", "-b2")
            direction: Direction vector (dr, dc)
            
        Returns:
            True if piece can attack in this direction
        """
        piece_type = piece.strip('-')[0]
        dr, dc = direction
        
        # Determine if direction is straight or diagonal
        is_straight = (dr == 0 or dc == 0)
        is_diagonal = (abs(dr) == abs(dc) and dr != 0)
        
        # Rook attacks on straight lines
        if piece_type == 'r':
            return is_straight
        
        # Bishop attacks on diagonals
        elif piece_type == 'b':
            return is_diagonal
        
        # Queen attacks on both
        elif piece_type == 'q':
            return is_straight or is_diagonal
        
        # Other pieces cannot pin
        else:
            return False
    
    def find_pins(self, color, matrix):
        """
        Finds all friendly pieces pinned by enemy pieces against the king.
        
        Args:
            color: Color of pieces to check for pins ("white" or "black").
            matrix: Current board state.
            
        Returns:
            Dictionary mapping pinned pieces to their pinning direction.
        """
        pins = {}  # {piece: allowed_directions}
        king = "k1" if color == "white" else "-k1"
        king_row, king_col = self.search_piece(king, matrix)
        
        # 8 directions: 4 straight + 4 diagonal
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),      # straight
            (1, 1), (1, -1), (-1, 1), (-1, -1)     # diagonal
        ]
        
        for dr, dc in directions:
            friendly_piece = None
            row, col = king_row + dr, king_col + dc
            
            # Scan outward from king
            while 0 <= row < 8 and 0 <= col < 8:
                piece = matrix[row, col]
                
                if piece != 0:
                    if not self.is_enemy(piece, color):
                        # Found friendly piece
                        if friendly_piece is None:
                            friendly_piece = (piece, row, col)
                        else:
                            break  # 2nd friendly piece, no pin possible
                    else:
                        # Found enemy piece
                        if friendly_piece is not None:
                            # Check if enemy can pin in this direction
                            if self.can_attack_in_direction(piece, (dr, dc)):
                                pins[friendly_piece[0]] = (dr, dc)
                        break
                
                row += dr
                col += dc
        
        return pins
    
    def check_checker(self, color: str, matrix: np.ndarray = database.matrix):
        """
        Checks if the given player is in check.
        
        Args:
            color: Player color to check ("white" or "black").
            matrix: Current board state.
            
        Returns:
            True if player is in check, False otherwise.
        """
        king = "k1" if color == "white" else "-k1"
        return self.opponent_legal_search(color, self.search_piece(king, matrix))
    
    def check_legal(self, color: str, matrix: np.ndarray = database.matrix) -> List[Tuple[int, int]|None]|bool:
        """
        Returns legal moves when the player is in check.
        
        Args:
            color: Player color to check ("white" or "black").
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        king = "k1" if color == "white" else "-k1"
        king_coordinates = self.search_piece(king, matrix)
        pieces = self.opponent_legal_search(color, king_coordinates, matrix, return_piece=True)

        if not pieces or isinstance(pieces, bool):
            return True
        elif len(pieces) >= 2:
            return []
        else:
            piece = pieces[0]
            if 'n' in piece or 'p' in piece:
                return [self.search_piece(piece, matrix)]
            elif 'r' in piece:
                rook_coordinates = self.search_piece(piece, matrix)
                if rook_coordinates[0] == king_coordinates[0]:
                    if rook_coordinates[1] > king_coordinates[1]:
                        return [(rook_coordinates[0], y) for y in range(king_coordinates[1] + 1, rook_coordinates[1]+1)]
                    else:
                        return [(rook_coordinates[0], y) for y in range(rook_coordinates[1], king_coordinates[1])]
                else:
                    if king_coordinates[0] > rook_coordinates[0]:
                        return [(x, rook_coordinates[1]) for x in range(rook_coordinates[0], king_coordinates[0])]
                    else:
                        return [(x, rook_coordinates[1]) for x in range(king_coordinates[0] + 1, rook_coordinates[0] + 1)]
            elif 'b' in piece:
                bishop_coordinates = self.search_piece(piece, matrix)
                distance = abs(bishop_coordinates[0] - king_coordinates[0])
                if bishop_coordinates[0] > king_coordinates[0] and bishop_coordinates[1] > king_coordinates[1]:
                    direction = (1, 1)
                elif bishop_coordinates[0] < king_coordinates[0] and bishop_coordinates[1] < king_coordinates[1]:
                    direction = (-1, -1)
                elif bishop_coordinates[0] > king_coordinates[0]:
                    direction = (1, -1)
                else:
                    direction = (-1, 1)
                return [(king_coordinates[0] + direction[0] * (i + 1), king_coordinates[1] + direction[1] * (i + 1)) for i in range(distance)]
            
            elif 'q' in piece:
                queen_coordinates = self.search_piece(piece, matrix)
                if any(queen_coordinates) == any(king_coordinates):
                    if queen_coordinates[0] == king_coordinates[0]:
                        if queen_coordinates[1] > king_coordinates[1]:
                            return [(queen_coordinates[0], y) for y in range(king_coordinates[1] + 1, queen_coordinates[1]+1)]
                        else:
                            return [(queen_coordinates[0], y) for y in range(queen_coordinates[1], king_coordinates[1])]
                    else:
                        if king_coordinates[0] > queen_coordinates[0]:
                            return [(x, queen_coordinates[1]) for x in range(queen_coordinates[0], king_coordinates[0])]
                        else:
                            return [(x, queen_coordinates[1]) for x in range(king_coordinates[0] + 1, queen_coordinates[0] + 1)]
                else:
                    distance = abs(queen_coordinates[0] - king_coordinates[0])
                    if queen_coordinates[0] > king_coordinates[0] and queen_coordinates[1] > king_coordinates[1]:
                        direction = (1, 1)
                    elif queen_coordinates[0] < king_coordinates[0] and queen_coordinates[1] < king_coordinates[1]:
                        direction = (-1, -1)
                    elif queen_coordinates[0] > king_coordinates[0]:
                        direction = (1, -1)
                    else:
                        direction = (-1, 1)
                    return [(king_coordinates[0] + direction[0] * (i + 1), king_coordinates[1] + direction[1] * (i + 1)) for i in range(distance)]
        return []
                
    def check_allowed_moves(self, moves: List[Tuple[int, int]], color: str, matrix: np.ndarray = database.matrix):
        final_moves = moves.copy()
        if self.check_checker(color, matrix):
            allowed_moves = self.check_legal(color, matrix)
            if allowed_moves == True:
                return final_moves
            elif not allowed_moves:
                return []
            
            for move in moves:
                if move not in allowed_moves:
                    final_moves.remove(move)
            return final_moves
        return final_moves

    def rook_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal moves for a rook, respecting pins.
        
        Args:
            piece: Rook piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        diagonal = (1, 1), (1, -1), (-1, 1), (-1, -1)

        is_pinned = piece in database.pins
        if is_pinned:
            direction = database.pins[piece]

            if direction in diagonal:
                return []
        else:
            direction = None
            
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        rook_row = matrix[row, :]
        rook_col = matrix[:, col]

        moves = []
        
        rook_pos_row = col
        rook_pos_col = row

        # Handle horizontal movement if allowed by pin
        if not direction or (direction and (direction == (0, -1) or direction == (0, 1))):
            # Left direction
            for idx, value in enumerate(rook_row[:rook_pos_row][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row, col - idx - 1))
                    break
                else:
                    moves.append((row, col - idx - 1))

            # Right direction
            for idx, value in enumerate(rook_row[rook_pos_row + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row, idx + col + 1))
                    break
                else:
                    moves.append((row, idx + col + 1))

        # Handle vertical movement if allowed by pin
        if not direction or (direction and (direction == (1, 0) or direction == (-1, 0))):
            # Up direction
            for idx, value in enumerate(rook_col[:rook_pos_col][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row - idx - 1, col))
                    break
                else:
                    moves.append((row - idx - 1, col))

            # Down direction
            for idx, value in enumerate(rook_col[rook_pos_col + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((idx + row + 1, col))
                    break
                else:
                    moves.append((idx + row + 1, col))

        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        return moves
    
    def bishop_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal moves for a bishop, respecting pins.
        
        Args:
            piece: Bishop piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        straight = (0, 1), (0, -1), (1, 0), (-1, 0), 

        if piece in database.pins:
            direction = database.pins[piece]

            if direction in straight:
                return []
        else:
            direction = None

        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        anti_matrix = np.fliplr(matrix)
        anti_row, anti_col = self.search_piece(piece, anti_matrix)

        moves = []

        # Extract diagonals using numpy diagonal
        diagonal = np.diagonal(matrix, offset=col - row)
        anti_diagonal = np.diagonal(anti_matrix, offset=anti_col - anti_row)

        diagonal_pos = np.where(diagonal == piece)[0][0]
        anti_diagonal_pos = np.where(anti_diagonal == piece)[0][0]

        # Main diagonal direction (1,1) or (-1,-1)
        if not direction or (direction and (direction == (1, 1) or direction == (-1, -1))):
            # Backward direction
            for idx, value in enumerate(diagonal[:diagonal_pos][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row - idx - 1, col - idx - 1))
                    break
                else:
                    moves.append((row - idx - 1, col - idx - 1))

            # Forward direction
            for idx, value in enumerate(diagonal[diagonal_pos + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row + idx + 1, col + idx + 1))
                    break
                else:
                    moves.append((row + idx + 1, col + idx + 1))
        
        # Anti-diagonal direction (1,-1) or (-1,1)
        if not direction or (direction and (direction == (1, -1) or direction == (-1, 1))):
            # Backward direction
            for idx, value in enumerate(anti_diagonal[:anti_diagonal_pos][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row - idx - 1, col + idx + 1))
                    break
                else:
                    moves.append((row - idx - 1, col + idx + 1))

            # Forward direction
            for idx, value in enumerate(anti_diagonal[anti_diagonal_pos + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row + idx + 1, col - idx - 1))
                    break
                else:
                    moves.append((row + idx + 1, col - idx - 1))
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        return moves

    def queen_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal moves for a queen by combining rook and bishop moves.
        
        Args:
            piece: Queen piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        return self.bishop_moves(piece, matrix) + self.rook_moves(piece, matrix)
    
    def knight_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal knight moves (L-shape). Knights ignore pins.
        
        Args:
            piece: Knight piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        if piece in database.pins.keys():
            return []
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        moves = []

        # All 8 possible knight moves
        for x, y in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
            expected_row, expected_col = row + x, col + y
            if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                if not self.is_occupied(matrix[expected_row, expected_col]):
                    moves.append((expected_row, expected_col))
                elif self.is_enemy(matrix[expected_row, expected_col], color):
                    moves.append((expected_row, expected_col))
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        return moves
    
    def king_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal king moves including castling.
        
        Args:
            piece: King piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        moves = []

        # 8 adjacent squares
        for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            expected_row, expected_col = row + x, col + y
            if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                if not self.is_occupied(matrix[expected_row, expected_col]):
                    moves.append((expected_row, expected_col))
                elif self.is_enemy(matrix[expected_row, expected_col], color):
                    moves.append((expected_row, expected_col))
        
        # Castling logic
        range_1, range_2 = range(1, 4), range(5, 7)

        row_ = 0 if color == "black" else 7
        # Check rook and king movement status for both sides
        checks = [(database.r1_moved, database.k1_moved), (database.r2_moved, database.k1_moved)] if color == "white" else [(database.r1_black_moved, database.k1_black_moved), (database.r2_black_moved, database.k1_black_moved)]
        for idx, range_ in enumerate([range_1, range_2]):
            if self.can_castle(color, range_, matrix):
                col_ = 1 if range_[0] == 1 else 6
                if not any(checks[idx]):  # Neither rook nor king has moved
                    moves.append((row_, col_))
        safe_moves = []
        for move in moves:
            # Simulate move and check if king would be attacked
            temp_matrix = matrix.copy()
            temp_matrix[row, col] = 0
            temp_matrix[move[0], move[1]] = piece
            
            if not self.opponent_legal_search(color, move, temp_matrix):
                safe_moves.append(move)
        
        return safe_moves
    
    def pawn_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        """
        Calculates all legal pawn moves including double move, captures, and en passant.
        
        Args:
            piece: Pawn piece identifier.
            matrix: Current board state.
            
        Returns:
            List of legal move coordinates (row, col).
        """
        is_pinned = piece in database.pins
        if is_pinned:
            direction_ = database.pins[piece]
            if direction_[1] != 0:  # If pinned horizontally or diagonally
                return []
        else:
            direction_ = None

        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        moves = []

        if color == "white":
            start_row = 6
            direction = -1
            opponent_last_pawn = database.black_last_pawn
        else:
            start_row = 1
            direction = 1
            opponent_last_pawn = database.white_last_pawn

        # Forward moves (including double pawn push from start)
        if not direction_ or direction_[1] == 0:  # Pinned vertically or not pinned
            if row == start_row:
                # Double move from starting position
                if not self.is_occupied(matrix[row + direction*2, col]) and not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction*2, col))
            if (color == "white" and row > 0) or (color == "black" and row < 7):
                # Single forward move
                if not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction, col))

        # Diagonal captures and en passant
        if not direction_ or direction_[1] != 0:  # Pinned diagonally or not pinned
            for x, y in [(direction, -1), (direction, 1)]:
                expected_row, expected_col = row + x, col + y
                if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                    if self.is_enemy(matrix[expected_row, expected_col], color):
                        moves.append((expected_row, expected_col))
            
            # En Passant capture
            en_passant_row = 3 if color == "white" else 4
            if opponent_last_pawn:
                row_l, col_l = opponent_last_pawn
                if row == en_passant_row and row_l == row and abs(col_l - col) == 1:
                    moves.append((row + direction, col_l))
        if self.check_checker(color, matrix):
            moves = self.check_allowed_moves(moves, color, matrix)
        return moves
    
    def calculate_legal_moves(self, piece: str):
        if 'p' in piece:
            return self.pawn_moves(piece)
        elif 'k' in piece:
            return self.king_moves(piece)
        elif 'q' in piece:
            return self.queen_moves(piece)
        elif 'r' in piece:
            return self.rook_moves(piece)
        elif 'b' in piece:
            return self.bishop_moves(piece)
        elif 'n' in piece:
            return self.knight_moves(piece)
    
    def all_legal_moves(self, color: str, matrix: np.ndarray = database.matrix) -> Dict[str, np.ndarray]:
        """
        Main method to calculate legal moves for all white pieces.
        
        Args:
            piece: Piece identifier.
            matrix: Current board state.
            
        Returns:
            Dictionary with piece identifiers as keys and lists of legal moves for the specified piece.
        """
        legal = {}
        pieces = database.white_pieces if color == "white" else database.black_pieces

        for piece in pieces.reshape(16, 1):
            piece = piece[0]
            legal[piece] = np.array(self.calculate_legal_moves(piece))

        return legal
    
    def update_legal_moves(self, matrix: np.ndarray = database.matrix):
        """
        Main method to calculate legal moves for all the pieces.
        
        Args:
            matrix: Current board state.
            
        Returns:
            None
            Updates database.white_legal_moves and database.black_legal_moves
        """

        database.pins = {}
        white_pins = self.find_pins("white", matrix)
        black_pins = self.find_pins("black", matrix)
        database.pins = {**white_pins, **black_pins}

        # First pass: populate with empty arrays to avoid KeyError
        database.white_legal_moves = {piece: np.array([]) for piece in database.white_pieces.flatten()}
        database.black_legal_moves = {piece: np.array([]) for piece in database.black_pieces.flatten()}

        # Second pass: calculate actual moves (now check_checker can work)
        database.white_legal_moves = self.all_legal_moves("white", matrix)
        database.black_legal_moves = self.all_legal_moves("black", matrix)
        
        print("Legal Moves Updated!\n")

if __name__ == "__main__":
    utils = Utilities()
    for piece, move in database.black_legal_moves.items():
        print(f"{piece}: {list(move)}")