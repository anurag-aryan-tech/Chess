# chess.py - Complete refactored version

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from utils import Utilities
from database.database import database

utils = Utilities()

@dataclass
class STYLE_CONFIG:
    """Configuration for chess game styling"""
    white_box_color: str = "white"
    black_box_color: str = "brown"
    chessboard_rely: float = 0.1
    close_button_fg: str = "transparent"
    close_button_hvr: str = "red"


class ChessGame:
    """Main chess game class handling UI and game logic"""
    
    def __init__(self) -> None:
        # Style configuration
        self.style = STYLE_CONFIG()
        
        # Window resize tracking
        self.prev_width = 0
        self.prev_height = 0
        self.size_threshold = 10
        self._resize_scheduled = False
        
        # Game state
        self.matrix = database.matrix
        self.piece_selected: Optional[str] = None
        self._legal_moves_update_scheduled = False
        
        # UI elements
        self.piece_labels: Dict[str, ctk.CTkLabel] = {}
        self.legal_move_indicators: Dict[str, ctk.CTkLabel] = {}
        self.chessboard_squares: Dict[Tuple[int, int], ctk.CTkFrame] = {}
        
        # Initialize window and UI
        self.root = self._initialize_window()
        self._setup_window(self.root)
        self._create_close_button()
        self._add_close_button()
        
        # Create chessboard
        self.chessboard_frame = self._create_chessboard_frame()
        self.root.update_idletasks()
        self._update_chessboard_position()
        self._configure_grid()
        self._create_chessboard_squares()
        self._render_all_pieces()
        
        self.root.mainloop()

    # ==================== WINDOW MANAGEMENT ====================
    
    def _initialize_window(self, title: str = "Chess Game") -> ctk.CTk:
        """Initialize the main window with bindings"""
        root = ctk.CTk()
        root.title(title)
        root.bind("<Escape>", lambda _: self._handle_escape())
        root.bind("<Configure>", lambda event: self._handle_window_resize(event))
        root.protocol("WM_DELETE_WINDOW", self._handle_close)
        return root
    
    def _setup_window(self, window: ctk.CTk) -> None:
        """Configure window properties"""
        utils.fullscreen_window(window)
        window.minsize(600, 440)
    
    def _handle_escape(self) -> None:
        """Handle escape key press"""
        self._toggle_close_button()
        utils.fullscreen_toggle(self.root)
    
    def _handle_window_resize(self, event) -> None:
        """Handle window resize events with debouncing"""
        if event.widget != self.root or self._resize_scheduled:
            return
        
        self._resize_scheduled = True
        self.root.after_idle(self._delayed_resize, event.width, event.height)
    
    def _delayed_resize(self, width: int, height: int) -> None:
        """Execute resize operations after debouncing"""
        self._resize_scheduled = False
        
        # Check if resize is significant enough
        if (abs(width - self.prev_width) < self.size_threshold and 
            abs(height - self.prev_height) < self.size_threshold):
            return
        
        self.prev_width = width
        self.prev_height = height
        
        self._update_chessboard_position()
        self.root.after(10, self._refresh_all_images)
    
    def _handle_close(self) -> None:
        """Handle window close event"""
        answer = messagebox.askyesno("Warning", "Are you sure you want to close the game?")
        if answer:
            self.root.destroy()
            print("Game closed!")

    # ==================== CLOSE BUTTON MANAGEMENT ====================
    
    def _create_close_button(self) -> None:
        """Create the close button"""
        self.close_button = ctk.CTkButton(
            self.root, 
            text="x", 
            command=self._handle_close,
            fg_color=self.style.close_button_fg,
            hover_color=self.style.close_button_hvr,
            font=("Arial", 22)
        )
    
    def _add_close_button(self) -> None:
        """Display the close button"""
        self.close_button.place(relx=0.95, rely=0, relwidth=0.05, relheight=0.05)
    
    def _remove_close_button(self) -> None:
        """Hide the close button"""
        self.close_button.place_forget()
    
    def _toggle_close_button(self) -> None:
        """Show/hide close button based on fullscreen state"""
        if self.root.attributes("-fullscreen"):
            self._add_close_button()
        else:
            self._remove_close_button()

    # ==================== CHESSBOARD SETUP ====================
    
    def _create_chessboard_frame(self) -> ctk.CTkFrame:
        """Create the main chessboard frame"""
        frame = ctk.CTkFrame(self.root, fg_color="white")
        print("Chessboard frame created!")
        return frame
    
    def _update_chessboard_position(self) -> None:
        """Update chessboard frame position based on window size"""
        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()
        
        rely = self.style.chessboard_rely
        relx = utils.relative_dimensions(rely, (screen_height, screen_width))
        
        self.chessboard_frame.place(
            relx=relx,
            rely=rely,
            relwidth=1 - relx * 2,
            relheight=1 - rely * 2
        )
    
    def _configure_grid(self) -> None:
        """Configure the 8x8 grid for the chessboard"""
        for i in range(8):
            self.chessboard_frame.rowconfigure(i, weight=1)
            self.chessboard_frame.columnconfigure(i, weight=1)
    
    def _create_chessboard_squares(self) -> None:
        """Create all 64 chessboard squares with alternating colors"""
        color = self.style.white_box_color
        color2 = self.style.black_box_color
        
        for row in range(8):
            for col in range(8):
                square = ctk.CTkFrame(
                    self.chessboard_frame,
                    fg_color=color,
                    bg_color=color
                )
                square.grid(row=row, column=col, sticky="nsew")
                square.bind("<Button-1>", lambda _, pos=(row, col): self._handle_square_click(pos))
                
                self.chessboard_squares[(row, col)] = square
                
                # Alternate colors
                color, color2 = color2, color
            
            # Alternate at end of row
            color, color2 = color2, color
        
        print("Chessboard squares created!\n")

    # ==================== PIECE RENDERING ====================
    
    def _render_all_pieces(self) -> None:
        """Render all pieces on the board"""
        for row in range(8):
            for col in range(8):
                piece = self.matrix[row, col]
                if piece != 0:
                    self._add_piece_to_square((row, col), piece)
    
    def _add_piece_to_square(self, square: Tuple[int, int], piece: str) -> None:
        """Add a piece image to a specific square"""
        color = "black" if '-' in piece else "white"
        path = f"images/{color}/{piece[:-1]}.png"
        
        image_size = self._calculate_image_size(square)
        image = utils.ctkimage_generator(path, size=(image_size, image_size))
        
        label = ctk.CTkLabel(
            self.chessboard_squares[square],
            image=image,
            text="",
            fg_color="transparent",
            bg_color="transparent"
        )
        label.place(relx=0.5, rely=0.5, anchor="center")
        label.bind("<Button-1>", lambda _, pos=square: self._handle_square_click(pos))
        
        self.piece_labels[str(square)] = label
    
    def _add_legal_move_indicator(self, square: Tuple[int, int]) -> None:
        """Add a dot indicator for a legal move"""
        path = "images/dot.png"
        
        image_size = self._calculate_image_size(square)
        image = utils.ctkimage_generator(path, size=(image_size, image_size))
        
        label = ctk.CTkLabel(
            self.chessboard_squares[square],
            image=image,
            text="",
            fg_color="transparent",
            bg_color="transparent"
        )
        label.place(relx=0.5, rely=0.5, anchor="center")
        label.bind("<Button-1>", lambda _, pos=square: self._handle_square_click(pos))
        
        self.legal_move_indicators[str(square)] = label
    
    def _calculate_image_size(self, square: Tuple[int, int]) -> int:
        """Calculate appropriate image size based on square dimensions"""
        square_frame = self.chessboard_squares[square]
        square_size = min(square_frame.winfo_width(), square_frame.winfo_height())
        
        if square_size <= 1:
            return 80  # Default size
        
        image_size = int(square_size * 0.9)
        return max(image_size, 40)  # Minimum 40px
    
    def _clear_piece_from_square(self, square: Tuple[int, int]) -> None:
        """Remove piece image from a square"""
        label_key = str(square)
        if label_key in self.piece_labels:
            self.piece_labels[label_key].destroy()
            del self.piece_labels[label_key]
    
    def _clear_all_legal_move_indicators(self) -> None:
        """Remove all legal move indicators"""
        for label in self.legal_move_indicators.values():
            label.destroy()
        self.legal_move_indicators.clear()
    
    def _clear_all_pieces(self) -> None:
        """Remove all piece images from the board"""
        for label in self.piece_labels.values():
            label.destroy()
        self.piece_labels.clear()
    
    def _refresh_all_images(self) -> None:
        """Refresh all piece and indicator images after resize"""
        self._clear_all_pieces()
        self._render_all_pieces()
        
        # Refresh legal move indicators if piece is selected
        if self.piece_selected:
            legal_moves = self._get_legal_moves_for_piece(self.piece_selected)
            self._clear_all_legal_move_indicators()
            self._show_legal_moves(legal_moves)

    # ==================== GAME LOGIC ====================
    
    def _get_legal_moves_for_piece(self, piece: str) -> np.ndarray:
        """Get legal moves for a specific piece"""
        color = "white" if '-' not in piece else "black"
        legal_moves_dict = database.get_legal_moves(color)
        return legal_moves_dict.get(piece, np.array([]))

    
    def _show_legal_moves(self, legal_moves: np.ndarray) -> None:
        """Display legal move indicators on the board"""
        for move in legal_moves:
            square = (move[0], move[1])
            self._add_legal_move_indicator(square)
    
    def _is_legal_move(self, piece: str, target_square: Tuple[int, int]) -> bool:
        """Check if a move is legal for the given piece"""
        legal_moves = self._get_legal_moves_for_piece(piece)
        return any(np.array_equal(target_square, move) for move in legal_moves)
    
    def _execute_castling(self, piece: str, from_square: Tuple[int, int], target_square: Tuple[int, int]) -> None:
        """Execute castling move"""
        self.matrix[from_square[0], from_square[1]] = 0

        # King side castling
        if target_square[1] == 6:
            rook = piece.replace("k1", "r2")
            rook_square = (from_square[0], 7)
            self.matrix[rook_square[0], rook_square[1]] = 0
            self.matrix[target_square[0], target_square[1]] = piece
            self.matrix[target_square[0], target_square[1]-1] = rook
        # Queen side castling
        else:
            rook = piece.replace("k1", "r1")
            rook_square = (from_square[0], 0)
            self.matrix[rook_square[0], rook_square[1]] = 0
            self.matrix[target_square[0], target_square[1]] = piece
            self.matrix[target_square[0], target_square[1]+1] = rook


    def _execute_move(self, from_square: Tuple[int, int], to_square: Tuple[int, int]) -> None:
        """Execute a piece move on the board"""
        piece = self.matrix[from_square[0], from_square[1]]
        
        # Validate move
        if not self._is_legal_move(piece, to_square):
            return
        
        # Castling Logic
        if "k" in piece:
            distance = abs(from_square[1] - to_square[1])
            if distance == 2:
                self._execute_castling(piece, from_square, to_square)
            else:
                self.matrix[from_square[0], from_square[1]] = 0
                self.matrix[to_square[0], to_square[1]] = piece
            if "-" in piece:
                database.k1_black_moved = True
            else:
                database.k1_moved = True
        else:
            if "p" in piece:
                distance = abs(from_square[0] - to_square[0])
                diagonal: bool = abs(from_square[1] - to_square[1]) == 1
                if diagonal:
                    direction = -1 if "-" in piece else 1
                    self.matrix[to_square[0] + direction, to_square[1]] = 0
                elif distance == 2:
                    if "-" in piece:
                        database.black_last_pawn = to_square
                    else:
                        database.white_last_pawn = to_square
                else:
                    if "-" in piece:
                        database.black_last_pawn = None
                    else:
                        database.white_last_pawn = None
            else:
                if "-" in piece:
                    database.black_last_pawn = None
                else:
                    database.white_last_pawn = None

            if "r1" in piece:
                if "-" in piece:
                    database.r1_black_moved = True
                else:
                    database.r1_moved = True
            elif "r2" in piece:
                if "-" in piece:
                    database.r2_black_moved = True
                else:
                    database.r2_moved = True
            # Update matrix IMMEDIATELY
            self.matrix[from_square[0], from_square[1]] = 0
            self.matrix[to_square[0], to_square[1]] = piece
        
        # Switch turn
        database.current_turn = "black" if database.current_turn == "white" else "white"
        if database.current_turn == "white":
            database.fullmove += 1

        print(f"Fullmove: {database.fullmove}")
        
        # Clear selection and indicators
        self.piece_selected = None
        self._clear_all_legal_move_indicators()
        
        # Refresh display IMMEDIATELY
        self._clear_all_pieces()
        self._render_all_pieces()
        
        # Schedule legal moves update for later (non-blocking)
        if not self._legal_moves_update_scheduled:
            self._legal_moves_update_scheduled = True
            self.root.after(100, self._delayed_legal_moves_update)

    def _delayed_legal_moves_update(self) -> None:
        """Update legal moves after delay"""
        utils.legal_moves.update_legal_moves(self.matrix)
        self._legal_moves_update_scheduled = False
    
    def _handle_square_click(self, square: Tuple[int, int]) -> None:
        """Handle click on a chessboard square"""
        piece = self.matrix[square[0], square[1]]
        
        # Determine piece color if square is occupied
        piece_color = None
        if piece != 0:
            piece_color = "white" if '-' not in piece else "black"
        
        # Case 1: Clicking on own piece - select it and show legal moves
        if piece != 0 and piece_color == database.current_turn:
            self.piece_selected = piece
            legal_moves = self._get_legal_moves_for_piece(piece)
            
            self._clear_all_legal_move_indicators()
            self._show_legal_moves(legal_moves)
        
        # Case 2: Clicking on empty square or opponent piece - try to move
        else:
            if self.piece_selected:
                from_square = utils.legal_moves.search_piece(self.piece_selected)
                self._execute_move(from_square, square)
            
            # Deselect and clear indicators
            self.piece_selected = None
            self._clear_all_legal_move_indicators()


if __name__ == "__main__":
    ChessGame()