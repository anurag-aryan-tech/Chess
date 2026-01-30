import tkinter as tk
import customtkinter as ctk
from typing import Tuple
import numpy as np
from database.database import database

class Utilities:
    def fullscreen_window(self, window: tk.Tk|ctk.CTk) -> None:
        window.attributes("-fullscreen", True)
        window.update_idletasks()

    def fullscreen_toggle(self, window: tk.Tk|ctk.CTk):
        window.attributes("-fullscreen", not window.attributes("-fullscreen"))
        print(f"Fullscreen: {True if window.attributes('-fullscreen') else False}\n")

    def relative_dimensions(self, rely: float, dimensions: Tuple[int, int]) -> float:
        height, width = dimensions
        real_y = rely * height
        
        # Height available for the square board
        h = height - real_y * 2
        
        # Width the square board will occupy
        square_width = h  # Since it's a square
        
        # Calculate relx to center the square
        relx = (width - square_width) / (2 * width)
        
        return relx
    
    
    

class LegalMoves:
    def search_piece(self, piece: str, matrix: np.ndarray = database.matrix):
        pos = np.where(matrix == piece)
        if len(pos[0]) == 0:
            raise ValueError(f"{piece} not found on board")
        return int(pos[0][0]), int(pos[1][0])

    
    def is_occupied(self, val): return val != 0
    def is_enemy(self, val, color: str):
        if val == 0:
            return False
        piece_is_black = '-' in str(val)
        return piece_is_black != (color == "black")
    
    def opponent_legal_search(self, color: str, coordinates: Tuple[int, int], matrix: np.ndarray = database.matrix):
        moves = database.white_legal_moves.values() if color == "black" else database.black_legal_moves.values()

        for move in moves:
            if any(move == coordinates):
                return True

        return False
    
    def can_castle(self, color: str, range: range, matrix: np.ndarray = database.matrix):
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


    
    def rook_moves(self, piece: str, matrix: np.ndarray = database.matrix):
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

            if direction and (direction == (0, -1) or direction == (0, 1)):
                for idx, value in enumerate(rook_row[:rook_pos_row][::-1]):
                    if self.is_occupied(value):
                        if self.is_enemy(value, color):
                            moves.append((row, col - idx - 1))
                        break
                    else:
                        moves.append((row, col - idx - 1))

                for idx, value in enumerate(rook_row[rook_pos_row + 1:]):
                    if self.is_occupied(value):
                        if self.is_enemy(value, color):
                            moves.append((row, idx + col + 1))
                        break
                    else:
                        moves.append((row, idx + col + 1))

            if direction and (direction == (1, 0) or direction == (-1, 0)):
                for idx, value in enumerate(rook_col[:rook_pos_col][::-1]):
                    if self.is_occupied(value):
                        if self.is_enemy(value, color):
                            moves.append((row - idx - 1, col))
                        break
                    else:
                        moves.append((row - idx - 1, col))

                for idx, value in enumerate(rook_col[rook_pos_col + 1:]):
                    if self.is_occupied(value):
                        if self.is_enemy(value, color):
                            moves.append((idx + row + 1, col))
                        break
                    else:
                        moves.append((idx + row + 1, col))

            return moves
    
    def bishop_moves(self, piece: str, matrix: np.ndarray = database.matrix):
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

        diagonal = np.diagonal(matrix, offset=col - row)
        anti_diagonal = np.diagonal(anti_matrix, offset=anti_col - anti_row)

        diagonal_pos = np.where(diagonal == piece)[0][0]
        anti_diagonal_pos = np.where(anti_diagonal == piece)[0][0]

        if direction and (direction == (1, 1) or direction == (-1, -1)):
            for idx, value in enumerate(diagonal[:diagonal_pos][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row - idx - 1, col - idx - 1))
                    break
                else:
                    moves.append((row - idx - 1, col - idx - 1))

            for idx, value in enumerate(diagonal[diagonal_pos + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row + idx + 1, col + idx + 1))
                    break
                else:
                    moves.append((row + idx + 1, col + idx + 1))
        
        if direction and (direction == (1, -1) or direction == (-1, 1)):
            for idx, value in enumerate(anti_diagonal[:anti_diagonal_pos][::-1]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row - idx - 1, col + idx + 1))
                    break
                else:
                    moves.append((row - idx - 1, col + idx + 1))

            for idx, value in enumerate(anti_diagonal[anti_diagonal_pos + 1:]):
                if self.is_occupied(value):
                    if self.is_enemy(value, color):
                        moves.append((row + idx + 1, col - idx - 1))
                    break
                else:
                    moves.append((row + idx + 1, col - idx - 1))

        return moves

    def queen_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        return self.bishop_moves(piece, matrix) + self.rook_moves(piece, matrix)
    
    def knight_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        if piece in database.pins.keys():
            return []
        
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        moves = []

        for x, y in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
            expected_row, expected_col = row + x, col + y
            if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                if not self.is_occupied(matrix[expected_row, expected_col]):
                    moves.append((expected_row, expected_col))
                elif self.is_enemy(matrix[expected_row, expected_col], color):
                    moves.append((expected_row, expected_col))

        return moves
    
    def king_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        color = "black" if '-' in piece else "white"
        row, col = self.search_piece(piece)

        moves = []

        for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            expected_row, expected_col = row + x, col + y
            if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                if not self.is_occupied(matrix[expected_row, expected_col]):
                    moves.append((expected_row, expected_col))
                elif self.is_enemy(matrix[expected_row, expected_col], color):
                    moves.append((expected_row, expected_col))
        
        # Castling
        range_1, range_2 = range(1, 4), range(5, 7)

        row_ = 0 if color == "black" else 7
        checks = [(database.r1_moved, database.k1_moved), (database.r2_moved, database.k1_moved)] if color == "white" else [(database.r1_black_moved, database.k1_black_moved), (database.r2_black_moved, database.k1_black_moved)]
        for idx, range_ in enumerate([range_1, range_2]):
            if self.can_castle(color, range_, matrix):
                col_ = 1 if range_[0] == 1 else 6
                if not any(checks[idx]):
                    moves.append((row_, col_))

        return moves
    
    def pawn_moves(self, piece: str, matrix: np.ndarray = database.matrix):

        is_pinned = piece in database.pins
        if is_pinned:
            direction_ = database.pins[piece]
            if direction_[0] >= 0:
                return []

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

        if direction_ and any(direction_) == 0:
            if row == start_row:
                if not self.is_occupied(matrix[row + direction*2, col]) and not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction*2, col))
            if (color == "white" and row > 0) or (color == "black" and row < 7):
                if not self.is_occupied(matrix[row + direction, col]):
                    moves.append((row + direction, col))

        
        if direction and all(direction_) != 0:
            for x, y in [(direction, -1), (direction, 1)]:
                expected_row, expected_col = row + x, col + y
                if 0 <= expected_row < 8 and 0 <= expected_col < 8:
                    if self.is_enemy(matrix[expected_row, expected_col], color):
                        moves.append((expected_row, expected_col))
            
            # En Passant
            en_passant_row = 3 if color == "white" else 4
            if opponent_last_pawn:
                row_l, col_l = opponent_last_pawn
                if row == en_passant_row and row_l == row and abs(col_l - col) == 1:
                    moves.append((row + direction, col_l))

        return moves
    def calculate_legal_moves(self, piece: str, matrix: np.ndarray = database.matrix):
        pass

if __name__ == "__main__":
    lg = LegalMoves()
    print(lg.find_pins("white", database.matrix))