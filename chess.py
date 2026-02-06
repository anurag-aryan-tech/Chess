# chess.py - The Game Entrance

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
    promotion_frame_color: str = "gray"


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
        self.view_matrix = self.matrix
        self.promoting = False
        self.promoting_square: Optional[Tuple[int, int]] = None
        self.piece_selected: Optional[str] = None
        self._legal_moves_update_scheduled = False
        self.flipped = False
        
        # UI elements
        self.piece_labels: Dict[str, ctk.CTkLabel] = {}
        self.legal_move_indicators: Dict[str, ctk.CTkLabel] = {}
        self.promotion_labels: Dict[str, ctk.CTkLabel] = {}
        self.chessboard_squares: Dict[Tuple[int, int], ctk.CTkFrame] = {}
        
        # Initialize window and UI
        self.root = self._initialize_window()
        self._setup_window(self.root)
        self._create_close_button()
        self._add_close_button()
        
        # Create chessboard
        self.chessboard_frame = self._create_chessboard_frame()
        self.promotion_frame = self._create_promotion_frame()
        self.root.update_idletasks()
        self._update_chessboard_position()
        self._configure_grid()
        self._create_chessboard_squares()
        self._render_all_pieces()
        
        self.root.mainloop()

    @property
    def matrix(self) -> np.ndarray:
        return database.matrix


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
            try:
                self.root.quit()  # Stop mainloop first
                self.root.destroy()  # Then destroy window
            except Exception as e:
                # Window already destroyed or other error
                print(f"Close error (safe to ignore): {e}")
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
    def _create_promotion_frame(self) -> ctk.CTkFrame:
        """Create the promotion frame"""
        frame = ctk.CTkFrame(self.root, fg_color=self.style.promotion_frame_color)
        
        rows, columns = 4, 1
        for row in range(rows):
            frame.rowconfigure(row, weight=1)
        
        for column in range(columns):
            frame.columnconfigure(column, weight=1)

        self._setup_promotion_labels(frame)

        return frame
    
    def _setup_promotion_labels(self, frame: ctk.CTkFrame) -> None:
        """Create promotion labels"""
        color1, color2 = self.style.white_box_color, self.style.black_box_color
        label_names = ["q", "r", "b", "n"]
        for label_name in label_names:
            label = ctk.CTkLabel(
                frame,
                fg_color=color1,
                bg_color=color1,
                text="")
            
            label.grid(row=label_names.index(label_name), column=0, sticky="nsew")
            label.bind("<Button-1>", lambda _, label_name=label_name: self._end_promotion(label_name))

            self.promotion_labels[label_name] = label
            color1, color2 = color2, color1

    def _setup_promotion_images(self, color: str) -> None:
        """Create promotion images"""
        prefix = "-" if color == "black" else ""
        for label_name in self.promotion_labels:
            self.promotion_labels[label_name].configure(
                image=utils.ctkimage_generator(f"images/{color}/{prefix}{label_name}.png", (50, 50))
            )
        self.promoting = True
        self._update_chessboard_position()
    
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

        if self.promoting:
            self._place_promotion_frame(relx, rely)
    
    def _place_promotion_frame(self, relx: float, rely: float) -> None:
        self.promotion_frame.place(
            relx= relx + (1 - relx * 2) + 0.05,
            rely = rely + (1 - rely * 2)/4,
            relheight = (1 - rely * 2)/2,
            relwidth = (1 - relx * 2)/8
        )

    def _hide_promotion_frame(self) -> None:
        self.promotion_frame.place_forget()
    
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
                # Read from view_matrix (already flipped)
                piece = self.view_matrix[row, col]
                if piece != 0:
                    # Render at visual position (row, col)
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
        square_frame = self.chessboard_squares[tuple(square)] #type: ignore
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
        # Safety check
        if not piece or piece == 0 or not isinstance(piece, str):
            return np.array([])
        
        color = "white" if '-' not in piece else "black"
        legal_moves_dict = database.get_legal_moves(color)
        return legal_moves_dict.get(piece, np.array([]))

    
    def _show_legal_moves(self, legal_moves: np.ndarray) -> None:
        """Display legal move indicators on the board"""
        for move in legal_moves:
            logical_square = (move[0], move[1])
            # Convert logical to visual before showing
            visual_square = self._logical_to_visual(logical_square)
            self._add_legal_move_indicator(visual_square)
    
    def _is_legal_move(self, piece: str, target_square: Tuple[int, int]) -> bool:
        """Check if a move is legal for the given piece"""
        # Safety check
        if not piece or piece == 0 or not isinstance(piece, str):
            return False
        
        legal_moves = self._get_legal_moves_for_piece(piece)
        return any(np.array_equal(target_square, move) for move in legal_moves)
    
    def _visual_to_logical(self, visual_square: Tuple[int, int]) -> Tuple[int, int]:
        """Convert visual grid position to logical matrix position"""
        if self.flipped:
            return (7 - visual_square[0], 7 - visual_square[1])
        return visual_square

    def _logical_to_visual(self, logical_square: Tuple[int, int]) -> Tuple[int, int]:
        """Convert logical matrix position to visual grid position"""
        if self.flipped:
            return (7 - logical_square[0], 7 - logical_square[1])
        return logical_square
    
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

    def _start_promotion(self, color: str, target_square: Tuple[int, int]) -> None:
        """Promote a pawn"""
        self.promoting = True
        self.promoting_square = target_square

        self._setup_promotion_images(color)

    def _end_promotion(self, base: str) -> None:
        """End the promotion process"""
        if self.promoting_square:
            color = "black" if self.promoting_square[0] == 7 else "white" #type: ignore

            piece = utils.next_piece(base, color)
            self.matrix[self.promoting_square[0], self.promoting_square[1]] = piece
            
            if color == "white":
                database.white_pieces = np.append(database.white_pieces, piece)
            else:
                database.black_pieces = np.append(database.black_pieces, piece)

            self.promoting = False
            self.promoting_square = None

            self._hide_promotion_frame()
            
            # FIX: Update legal moves immediately after promotion
            utils.legal_moves.update_legal_moves(self.matrix)
            
            self._refresh_all_images()

    def _show_game_over_dialog(self, result: str, winner: Optional[str] = None):
        """Show game over dialog"""
        # Create semi-transparent overlay
        overlay = ctk.CTkFrame(
            self.root, 
            fg_color=("gray50", "gray20"),
            bg_color="transparent"
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Calculate responsive sizing
        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()
        
        # Use percentage but with min/max constraints
        dialog_width = max(300, min(int(screen_width * 0.5), 600))  # 50% of screen, min 300px, max 600px
        dialog_height = max(250, min(int(screen_height * 0.4), 400))  # 40% of screen, min 250px, max 400px
        
        # Create dialog box
        dialog = ctk.CTkFrame(
            overlay,
            fg_color=("white", "gray10"),
            corner_radius=20,
            border_width=2,
            border_color=("gold" if result == "checkmate" else "gray"),
            width=dialog_width,
            height=dialog_height
        )
        dialog.place(relx=0.5, rely=0.5, anchor="center")
        
        # Make dialog resize-aware
        dialog.pack_propagate(False)  # Don't let content resize the dialog
        
        # Title - use responsive font
        title_font_size = max(20, min(int(screen_height * 0.04), 32))
        subtitle_font_size = max(16, min(int(screen_height * 0.03), 24))
        
        if result == "checkmate":
            title = "👑 CHECKMATE! 👑"
            subtitle = f"{winner.capitalize()} Wins!" if winner else ""
        else:
            title = "⚔️ STALEMATE ⚔️"
            subtitle = "Game Drawn"
        
        title_label = ctk.CTkLabel(
            dialog, 
            text=title, 
            font=("Arial Bold", title_font_size),
            wraplength=dialog_width - 40  # Wrap text if needed
        )
        title_label.pack(pady=(20, 10))
        
        subtitle_label = ctk.CTkLabel(
            dialog, 
            text=subtitle, 
            font=("Arial", subtitle_font_size),
            wraplength=dialog_width - 40
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Button frame
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=10, expand=True)
        
        # Responsive button sizing
        button_width = max(100, min(int(dialog_width * 0.3), 140))
        button_height = max(30, min(int(dialog_height * 0.12), 40))
        button_font_size = max(12, min(int(screen_height * 0.02), 16))
        
        # New Game button
        new_game_btn = ctk.CTkButton(
            button_frame,
            text="New Game",
            font=("Arial", button_font_size),
            width=button_width,
            height=button_height,
            command=lambda: self._start_new_game(overlay)
        )
        new_game_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # View Board button
        view_board_btn = ctk.CTkButton(
            button_frame,
            text="View Board",
            font=("Arial", button_font_size),
            width=button_width,
            height=button_height,
            fg_color="transparent",
            border_width=2,
            command=lambda: self._view_board_with_menu(overlay, result, winner)
        )
        view_board_btn.grid(row=0, column=1, padx=10, pady=5)

    def _view_board_with_menu(self, overlay: ctk.CTkFrame, result: str, winner: Optional[str] = None):
        """Hide dialog but keep overlay with floating menu button"""
        overlay.destroy()
        
        # Create a small floating button in corner
        menu_button = ctk.CTkButton(
            self.root,
            text="⚙ Menu",
            font=("Arial", 14),
            width=100,
            height=40,
            command=lambda: self._show_game_over_dialog(result, winner)
        )
        menu_button.place(relx=0.88, rely=0.92, anchor="center")
        
        # Store reference so we can clean it up
        self._game_over_menu_btn = menu_button


    def _start_new_game(self, overlay: ctk.CTkFrame) -> None:
        """Start a new game by resetting state"""
        global utils
        
        overlay.destroy()
        
        # Clean up floating menu button if it exists
        if hasattr(self, '_game_over_menu_btn'):
            self._game_over_menu_btn.destroy()
            del self._game_over_menu_btn
        
        # Reset database first
        database.reset()
        utils = Utilities()
        
        # Reset game state (including flip state)
        self.promoting = False
        self.promoting_square = None
        self.piece_selected = None
        self._legal_moves_update_scheduled = False
        self.flipped = False  # ADD THIS - Reset flip state
        
        # Reset view matrix to standard orientation
        self.view_matrix = database.matrix  # ADD THIS
        
        # Clear ALL UI elements (pieces AND indicators)
        self._clear_all_pieces()
        self._clear_all_legal_move_indicators()
        
        # Clear promotion labels
        for label in self.promotion_labels.values():
            label.configure(image=None)
        
        # Re-render board with starting position
        self._render_all_pieces()
        
        # Update legal moves for new position
        utils.legal_moves.update_legal_moves(database.matrix)

        print("New game started!\n")


    def _execute_move(self, from_square: Tuple[int, int], to_square: Tuple[int, int]) -> None:
        """Execute a piece move on the board"""
        # Both from_square and to_square are now LOGICAL coordinates
        piece = database.matrix[from_square[0], from_square[1]]
        
        # Safety check
        if piece == 0 or not isinstance(piece, str):
            print("Error: No piece at source square")
            self.piece_selected = None
            self._clear_all_legal_move_indicators()
            return
        
        # Validate move (legal_moves are in logical coords, to_square is logical - matches!)
        if not self._is_legal_move(piece, to_square):
            return
        
        if "p" in piece and to_square[0] in [0, 7]:
            self.matrix[from_square[0], from_square[1]] = 0

            # Switch turn
            database.current_turn = "black" if database.current_turn == "white" else "white"
            if database.current_turn == "white":
                database.fullmove += 1

            if "-" in piece:
                database.black_last_pawn = None
                self._start_promotion("black", to_square)
                database.black_pieces = database.black_pieces[database.black_pieces != piece]
            else:
                database.white_last_pawn = None
                self._start_promotion("white", to_square)
                database.white_pieces = database.white_pieces[database.white_pieces != piece]

            return  # Don't continue - promotion will handle the rest

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
            if "p" in piece and to_square[0] not in [0, 7]:
                distance = abs(from_square[0] - to_square[0])
                diagonal: bool = abs(from_square[1] - to_square[1]) == 1
                
                # En passant capture
                if diagonal and database.en_passant and utils.matrix_to_chess(to_square) == database.en_passant:
                    # Remove the captured pawn (it's on the same row as from_square)
                    self.matrix[from_square[0], to_square[1]] = 0
                    # Clear en passant
                    # database.en_passant = ""
                    if "-" in piece:
                        database.white_last_pawn = None
                    else:
                        database.black_last_pawn = None
                
                # Double pawn push
                elif distance == 2:
                    if "-" in piece:
                        database.black_last_pawn = to_square
                        en_passant_square = (to_square[0] - 1, to_square[1])
                        database.en_passant = utils.matrix_to_chess(en_passant_square)
                    else:
                        database.white_last_pawn = to_square
                        en_passant_square = (to_square[0] + 1, to_square[1])
                        database.en_passant = utils.matrix_to_chess(en_passant_square)

                # Regular pawn move
                else:
                    database.en_passant = ""
                    if "-" in piece:
                        database.black_last_pawn = None
                    else:
                        database.white_last_pawn = None
            else:
                # Non-pawn moves clear en passant
                database.en_passant = ""
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
        
        # Clear selection and indicators
        self.piece_selected = None
        self._clear_all_legal_move_indicators()

        # Flip board if needed
        self._flip_board(database.current_turn)
        
        # Refresh display IMMEDIATELY
        self._clear_all_pieces()
        self._render_all_pieces()
        
        # Schedule legal moves update for later (non-blocking)
        if not self._legal_moves_update_scheduled:
            self._legal_moves_update_scheduled = True
            self.root.after(100, self._delayed_legal_moves_update)

    def _delayed_legal_moves_update(self) -> None:
        """Update legal moves after delay"""
        utils.legal_moves.update_legal_moves(database.matrix)  # Use database.matrix
        self._legal_moves_update_scheduled = False
        
        self._refresh_all_images()
        self._check_game_over()

    def _flip_board(self, color: str) -> None:
        """Flip the board view"""
        if color == "black":
            # 180° rotation (flip both axes)
            self.view_matrix = np.flip(np.flip(database.matrix, 0), 1)
            self.flipped = True
        else:
            # Just copy (or reference is fine)
            self.view_matrix = database.matrix
            self.flipped = False

    def _flip_legal_moves(self, color) -> None:
        """Flip legal moves"""
        if color == "black":
            database.black_legal_moves = utils.flip_legal(database.black_legal_moves) 
            database.white_legal_moves = utils.flip_legal(database.white_legal_moves)

    def _check_game_over(self) -> None:
        """Check if game is over (checkmate or stalemate)"""
        if database.current_turn == "white":
            # Count total legal moves for white
            total_moves = sum(len(moves) for moves in database.white_legal_moves.values())
            
            if total_moves == 0:
                if utils.legal_moves.check_checker("white"):
                    self._show_game_over_dialog("checkmate", "black")
                else:
                    self._show_game_over_dialog("stalemate")
        else:
            # Count total legal moves for black
            total_moves = sum(len(moves) for moves in database.black_legal_moves.values())
            
            if total_moves == 0:
                if utils.legal_moves.check_checker("black"):
                    self._show_game_over_dialog("checkmate", "white")
                else:
                    self._show_game_over_dialog("stalemate")

    
    def _handle_square_click(self, visual_square: Tuple[int, int]) -> None:
        """Handle click on a chessboard square"""
        if self.promoting:
            return
        
        # Convert visual to logical
        logical_square = self._visual_to_logical(visual_square)
        
        # Read from database.matrix (logical truth)
        piece = database.matrix[logical_square[0], logical_square[1]]
        
        # Determine piece color if square is occupied
        piece_color = None
        if piece != 0 and isinstance(piece, str):
            piece_color = "white" if '-' not in piece else "black"
        
        # Case 1: Clicking on own piece - select it and show legal moves
        if piece != 0 and isinstance(piece, str) and piece_color == database.current_turn:
            self.piece_selected = piece
            legal_moves = self._get_legal_moves_for_piece(piece)
            
            self._clear_all_legal_move_indicators()
            self._show_legal_moves(legal_moves)  # This will convert internally
        
        # Case 2: Clicking on empty square or opponent piece - try to move
        else:
            if self.piece_selected and isinstance(self.piece_selected, str):
                try:
                    # Search in database.matrix (logical)
                    from_square = utils.legal_moves.search_piece(self.piece_selected, database.matrix)
                    # Both coords are now logical
                    self._execute_move(from_square, logical_square)
                except ValueError as e:
                    print(f"Piece not found: {e}")
                    self.piece_selected = None
                    self._clear_all_legal_move_indicators()
                    return
            
            # Deselect and clear indicators
            self.piece_selected = None
            self._clear_all_legal_move_indicators()


if __name__ == "__main__":
    ChessGame()