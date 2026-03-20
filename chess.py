# chess.py - The Game Entrance
import math
import platform
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from tkinter import messagebox
from typing import Any, Literal, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image

from database.database import MoveResult, database
from review_system import ReviewSystem
from utils import Utilities


@dataclass
class STYLE_CONFIG:
    """Configuration for chess game styling"""

    white_box_color: str = "white"
    black_box_color: str = "brown"
    white_highlight: str = "#F7EB72"
    black_highlight: str = "#DCC44B"
    chessboard_rely: float = 0.1
    close_button_fg: str = "transparent"
    close_button_hvr: str = "red"
    promotion_frame_color: str = "gray"
    resign_width: int = 1
    resign_hover_width: int = 2
    resign_fg_color: str = "transparent"
    resign_hover_color: str = "green"
    settings_width: int = 1
    settings_hover_width: int = 2
    settings_fg_color: str = "transparent"
    settings_hover_color: str = "blue"
    flip_width: int = 1
    flip_fg_color: str = "transparent"
    flip_hover_color: str = "lightyellow"

    # Start menu button styling
    start_menu_bg: str = "#1A1410"
    start_menu_fg: tuple = ("#20160F", "#1A1410")
    start_menu_border: str = "#D4AF37"
    start_menu_border_width: int = 2
    start_menu_hover: str = "#D4AF37"

    _DLG_SHADOW = 5
    _DLG_BG_OUTER = "#2b2b2b"
    _DLG_CARD_BG = "#3d3d3d"
    _DLG_HEADER_BG = "#2e2e2e"
    _DLG_STAT_BG = "#4a4a4a"
    _DLG_SEP = "#555555"
    _DLG_BTN_GREEN = "#5faa3a"
    _DLG_GREEN_HOV = "#4d8e2f"
    _DLG_BTN_DARK = "#4a4a4a"
    _DLG_DARK_HOV = "#606060"
    _DLG_SHADOW_CLR = "#1a1a1a"
    _DLG_WHITE = "#ffffff"
    _DLG_GRAY = "#888888"
    _DLG_BLUE = "#5b9bd5"
    _DLG_RED = "#db0b0b"
    _DLG_YELLOW = "#d4a017"
    _DLG_GREEN_ACC = "#5faa3a"
    _DLG_ICON_BLUE = "#3a6da8"
    _DLG_ICON_YELLOW = "#c89a10"
    _DLG_ICON_GREEN = "#4a8a2a"

    _IS_WIN = platform.system() == "Windows"
    _EMOJI_FONT_FAMILY = "Segoe UI Emoji" if _IS_WIN else "TkDefaultFont"

    # Start menu button positioning
    button_relx: float = 0.3
    button_relwidth: float = 0.4
    button_relheight: float = 2 / 19
    pass_play_rely: float = 5 / 19
    vs_ai_rely: float = 9 / 19
    coming_soon_rely: float = 13 / 19

    # Vs AI Selector position
    selector_relx: float = 0.5
    selector_relwidth: float = 0.5
    selector_relheight: float = 0.6
    footer_text_color: str = "#A9A9A9"


class ChessGame:
    """Main chess game class handling UI and game logic"""

    def __init__(self) -> None:
        # Style configuration
        self.style = STYLE_CONFIG()

        # Services
        self.utils = Utilities()
        self.review = ReviewSystem()

        # Window resize tracking
        self.prev_width = 0
        self.prev_height = 0
        self.size_threshold = 10
        self._resize_scheduled = False

        # Game state
        self.view_matrix = self.matrix
        self.promoting = False
        self.promoting_square: Optional[tuple[int, int]] = None
        self.promotion_from_square: Optional[tuple[int, int]] = None
        self.piece_selected: Optional[str] = None
        self._pending_move: Optional[
            tuple[tuple[int, int], tuple[int, int], str, bool, str, str]
        ] = None
        self._legal_moves_update_scheduled = False
        self.flip_allowed = True
        self.flipped = False
        self.settings_open = False
        self.game_mode: Optional[str] = None
        self.disabled_color: list = [0, 0]
        self.last_from_square = None
        self.last_to_square = None
        self.selected_square: Optional[tuple[int, int]] = None
        self.empty_image = ctk.CTkImage(
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1)
        )
        self._dot_pil_base = Image.open("images/dot.png").convert("RGBA")
        self.current_fen = 0

        # UI elements
        self.piece_labels: dict[tuple[int, int], ctk.CTkLabel] = {}
        self._ctk_image_cache: dict[tuple[str, bool, int], ctk.CTkImage] = {}
        self._render_state: dict[tuple[int, int], tuple[str, bool, int]] = {}
        self._current_legal_visual_squares: set[tuple[int, int]] = set()
        self._highlighted_visual_squares: set[tuple[int, int]] = set()
        self._square_color_state: dict[tuple[int, int], str] = {}
        self._all_visual_squares: set[tuple[int, int]] = {
            (row, col) for row in range(8) for col in range(8)
        }
        self._last_image_size = 0
        self._warmed_image_sizes: set[int] = set()
        self.promotion_labels: dict[str, ctk.CTkLabel] = {}
        self.chessboard_squares: dict[tuple[int, int], ctk.CTkFrame] = {}

        self.classifier_images: dict[str, ctk.CTkImage] = {}

        # AI scheduling (non-blocking fetch + stale request guard)
        self._ai_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chess-ai"
        )
        self._ai_future: Optional[Future[Optional[str]]] = None
        self._ai_request_token: int = 0

        # Start menu tracking
        self.start_menu: Optional[ctk.CTkFrame] = None
        self.start_menu_background: Optional[ctk.CTkLabel] = None

        # VS AI menu tracking
        self.vs_ai_menu: Optional[ctk.CTkFrame] = None
        self.vs_ai_menu_background: Optional[ctk.CTkLabel] = None
        self.vs_ai_configurations: Optional[dict[str, Any]] = None
        self.footer_frame: Optional[ctk.CTkFrame] = None
        self.footer_label: Optional[ctk.CTkLabel] = None

        # Initialize window and UI
        self.root = self._initialize_window()
        self._setup_window(self.root)
        self._create_close_button()
        self._create_footer_signature()
        self._add_close_button()
        self._bind_arrow_keys()
        self._load_classifier_images()

        # Create chessboard (hidden initially)
        self._create_chessboard()

        # Show start menu
        self._show_start_menu()

        self.root.mainloop()

    @property
    def matrix(self) -> np.ndarray:
        return database.matrix

    # ==================== WINDOW MANAGEMENT ====================

    def _initialize_window(self, title: str = "Chess Game") -> ctk.CTk:
        """Initialize the main window with bindings"""
        root = ctk.CTk()
        root.title(title)

        root.bind("<s>", lambda _: self._show_settings_overlay())
        root.bind("<Escape>", lambda _: self._handle_escape())
        root.bind("<Configure>", lambda event: self._handle_window_resize(event))
        root.protocol("WM_DELETE_WINDOW", self._handle_close)
        return root

    def _bind_arrow_keys(self):
        self.root.bind("<Left>", self._on_left_arrow_click)
        self.root.bind("<Right>", self._on_right_arrow_click)
        self.root.bind("<Control-Left>", lambda _: self._on_ctrl_arrow_click("left"))
        self.root.bind("<Control-Right>", lambda _: self._on_ctrl_arrow_click("right"))

    def _setup_window(self, window: ctk.CTk) -> None:
        """Configure window properties"""
        self.utils.fullscreen_window(window)
        window.minsize(750, 550)

    def _handle_escape(self) -> None:
        """Handle escape key press"""
        self._toggle_close_button()
        self.utils.fullscreen_toggle(self.root)

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
        if (
            abs(width - self.prev_width) < self.size_threshold
            and abs(height - self.prev_height) < self.size_threshold
        ):
            return

        self.prev_width = width
        self.prev_height = height
        self._refresh_footer_signature(height)

        # Update chessboard position
        self._update_chessboard_position()

        # Recreate start menu only if game hasn't started
        if (
            self.game_mode is None
            and self.start_menu
            and self.start_menu.winfo_exists()
        ):
            self._refresh_start_menu()

        if self.vs_ai_menu and self.vs_ai_menu.winfo_exists():
            self._refresh_vs_ai_menu()

        if hasattr(self, "game_over_menu") and self.game_over_menu and self.game_over_menu.winfo_exists():
            self.game_over_menu.destroy()
            self._show_game_over_dialog(self._last_game_result, self._last_game_winner)

        self.root.after(10, self._refresh_all_images)

    def _handle_close(self) -> None:
        """Handle window close event"""
        answer = messagebox.askyesno(
            "Warning", "Are you sure you want to close the game?"
        )
        if answer:
            try:
                self._cancel_pending_ai_work(shutdown_executor=True)
                self.review.shutdown()
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                database.gamelogger.error(f"Close error (safe to ignore): {e}")
            database.gamelogger.game("Game closed!")

    def _load_classifier_images(self):
        names = ["brilliant", "great", "best", "excellent", "good", "book", "inaccuracy", "mistake", "miss", "blunder"]

        for name in names:
            path = f"images/classifiers/{name}.png"

            # Using PIL directly, similar to your piece caching logic
            try:
                pil_image = Image.open(path).convert("RGBA")
                image = ctk.CTkImage(pil_image, size=(40, 40))
                self.classifier_images[name] = image
            except Exception as e:
                # Good practice to catch missing image files so the game doesn't crash
                database.gamelogger.error(f"Failed to load {name} icon: {e}")

    # ==================== CLOSE BUTTON MANAGEMENT ====================

    def _create_close_button(self) -> None:
        """Create the close button"""
        self.close_button = ctk.CTkButton(
            self.root,
            text="x",
            command=self._handle_close,
            fg_color=self.style.close_button_fg,
            hover_color=self.style.close_button_hvr,
            font=("Arial", 22),
        )

    def _add_close_button(self) -> None:
        """Display the close button"""
        self.close_button.place(relx=0.95, rely=0, relwidth=0.05, relheight=0.05)

    def _remove_close_button(self) -> None:
        """Hide the close button"""
        self.close_button.place_forget()

    def _toggle_close_button(self) -> None:
        """Show/hide close button based on fullscreen state"""
        self.root.update_idletasks()
        if not self.root.attributes("-fullscreen"):
            self._add_close_button()
        else:
            self._remove_close_button()

        self.root.after_idle(self._refresh_all_images)

    def _lift_persistent_widgets(self) -> None:
        """Keep always-on widgets visible above temporary overlays."""
        if self.close_button and self.close_button.winfo_exists():
            self.close_button.lift()

    def _show_footer_signature(self) -> None:
        """Show footer only when the board is in the main view."""
        if self.footer_frame and self.footer_frame.winfo_exists():
            self.footer_frame.place(relx=0.5, rely=0.995, anchor="s")

    def _hide_footer_signature(self) -> None:
        """Hide footer while any full-screen overlay/menu is open."""
        if self.footer_frame and self.footer_frame.winfo_exists():
            self.footer_frame.place_forget()

    def _create_footer_signature(self) -> None:
        """Create a subtle bottom-center signature."""
        self.footer_frame = ctk.CTkFrame(
            self.root, fg_color="transparent", bg_color="transparent"
        )
        self.footer_label = ctk.CTkLabel(
            self.footer_frame,
            text="Made with ♥ - by Anurag Aryan",
            font=("Arial", 20, "bold"),
            text_color=self.style.footer_text_color,
            fg_color="transparent",
            bg_color="transparent",
        )
        self.footer_label.pack(padx=8, pady=4)
        self.footer_frame.place(relx=0.5, rely=0.995, anchor="s")
        self._lift_persistent_widgets()

    def _refresh_footer_signature(self, window_height: int) -> None:
        """Keep footer readable but unobtrusive across window sizes."""
        if self.footer_label and self.footer_label.winfo_exists():
            footer_font_size = max(11, min(int(window_height * 0.018), 15))
            self.footer_label.configure(font=("Arial", footer_font_size, "bold"))
        self._lift_persistent_widgets()

    # ==================== CHESSBOARD setUP ====================

    def _create_chessboard(self) -> None:
        """Create all chessboard UI elements"""
        self.chessboard_frame = self._create_chessboard_frame()
        self.promotion_frame = self._create_promotion_frame()
        self.resign_button = self._create_resign_button()
        self.settings_button = self._create_settings_button()
        self.flip_button = self._create_flip_button()
        self.root.update_idletasks()
        self._update_chessboard_position()
        self._configure_grid()
        self._create_chessboard_squares()
        self._refresh_all_images()
        self._warm_image_cache_for_size(self._calculate_image_size())

        database.gamelogger.game("Chessboard Ready!")

    def _create_chessboard_frame(self) -> ctk.CTkFrame:
        """Create the main chessboard frame"""
        frame = ctk.CTkFrame(self.root, fg_color="white")
        return frame

    def _create_promotion_frame(self) -> ctk.CTkFrame:
        """Create the promotion frame"""
        frame = ctk.CTkFrame(self.root, fg_color=self.style.promotion_frame_color)

        for row in range(4):
            frame.rowconfigure(row, weight=1)
        frame.columnconfigure(0, weight=1)

        self._setup_promotion_labels(frame)
        return frame

    def _setup_promotion_labels(self, frame: ctk.CTkFrame) -> None:
        """Create promotion labels"""
        color1, color2 = self.style.white_box_color, self.style.black_box_color
        label_names = ["q", "r", "b", "n"]

        for idx, label_name in enumerate(label_names):
            label = ctk.CTkLabel(frame, fg_color=color1, bg_color=color1, text="")
            label.grid(row=idx, column=0, sticky="nsew")
            label.bind("<Button-1>", lambda _, ln=label_name: self._end_promotion(ln))
            self.promotion_labels[label_name] = label
            color1, color2 = color2, color1

    def _setup_promotion_images(self, color: str) -> None:
        """Create promotion images"""
        for label_name in self.promotion_labels:
            self.promotion_labels[label_name].configure(
                image=self.utils.ctkimage_generator(
                    f"images/{color}/{label_name}.png", (50, 50)
                )
            )
        self.promoting = True
        self._update_chessboard_position()

    def _create_resign_button(self) -> ctk.CTkButton:
        return ctk.CTkButton(
            self.root,
            fg_color=self.style.resign_fg_color,
            bg_color="transparent",
            text="🏁",
            font=ctk.CTkFont("Roboto", 22, "bold"),
            corner_radius=15,
            hover_color=self.style.resign_hover_color,
            command=self._resign_game,
        )

    def _resign_game(self) -> None:
        winner = "white" if database.current_turn == "black" else "black"
        answer = messagebox.askyesno(
            "Resign",
            f"Are you sure you want to Resign? {winner.upper()} will WIN!!",
            icon="warning",
        )
        if answer:
            # Update PGN result
            result = "1-0" if winner == "white" else "0-1"
            termination = f"{winner.capitalize()} won by resignation"
            self._update_pgn_result(result, termination)
            self.review.log_post_game_summary_if_ready()
            self._show_game_over_dialog("resign", winner)

    def _resign_hover(self, relx: float, relwidth: float):
        self.resign_button.place(relx=relx, relwidth=relwidth)
        self.resign_button.configure(
            fg_color=self.style.resign_hover_color, text="Resign 🏁"
        )

    def _resign_unhover(self, relx: float, relwidth: float):
        self.resign_button.place(relx=relx, relwidth=relwidth)
        self.resign_button.configure(fg_color=self.style.resign_fg_color, text="🏁")

    def _create_settings_button(self) -> ctk.CTkButton:
        return ctk.CTkButton(
            self.root,
            fg_color=self.style.settings_fg_color,
            bg_color="transparent",
            text="⚙",
            font=ctk.CTkFont("Roboto", 22, "bold"),
            corner_radius=15,
            hover_color=self.style.settings_hover_color,
            command=self._show_settings_overlay,
        )

    def _settings_hover(self, relx: float, relwidth: float):
        self.settings_button.place(relx=relx, relwidth=relwidth)
        self.settings_button.configure(
            fg_color=self.style.settings_hover_color, text="settings ⚙"
        )

    def _settings_unhover(self, relx: float, relwidth: float):
        self.settings_button.place(relx=relx, relwidth=relwidth)
        self.settings_button.configure(fg_color=self.style.settings_fg_color, text="⚙")

    def _place_settings_button(self, relx: float, rely: float, relwidth: float, relheight: float) -> None:
        self.settings_button.unbind("<Enter>")
        self.settings_button.unbind("<Leave>")
        self.settings_button.place(
            relx=relx, rely=rely, relheight=relheight, relwidth=relwidth
        )

        square_width = relwidth
        base_relwidth = square_width * self.style.settings_width
        hover_relwidth = square_width * self.style.settings_hover_width
        hover_offset = square_width * (
            self.style.settings_hover_width - self.style.settings_width
        )

        self.settings_button.bind(
            "<Enter>",
            lambda e=None, rx=relx - hover_offset, rw=hover_relwidth: (
                self._settings_hover(rx, rw)
            ),
        )
        self.settings_button.bind(
            "<Leave>",
            lambda e=None, rx=relx, rw=base_relwidth: self._settings_unhover(rx, rw),
        )

    def _create_flip_button(self) -> ctk.CTkButton:
        return ctk.CTkButton(
            self.root,
            fg_color=self.style.flip_fg_color,
            bg_color="transparent",
            text="🔄",
            font=ctk.CTkFont("Roboto", 22, "bold"),
            corner_radius=15,
            hover_color=self.style.flip_hover_color,
            command=self._toggle_flip_board,
        )

    def _place_flip_button(
        self, relx: float, rely: float, relwidth: float, relheight: float
    ) -> None:
        self.flip_button.place(
            relx=relx, rely=rely, relheight=relheight, relwidth=relwidth
        )

    def _place_resign_button(self, relx: float, rely: float, relwidth: float, relheight: float) -> None:
        self.resign_button.unbind("<Enter>")
        self.resign_button.unbind("<Leave>")
        self.resign_button.place(
            relx=relx, rely=rely, relheight=relheight, relwidth=relwidth
        )

        square_width = relwidth
        base_relwidth = square_width * self.style.resign_width
        hover_relwidth = square_width * self.style.resign_hover_width
        hover_offset = square_width * (
            self.style.resign_hover_width - self.style.resign_width
        )

        self.resign_button.bind(
            "<Enter>",
            lambda e=None, rx=relx - hover_offset, rw=hover_relwidth: (
                self._resign_hover(rx, rw)
            ),
        )
        self.resign_button.bind(
            "<Leave>",
            lambda e=None, rx=relx, rw=base_relwidth: self._resign_unhover(rx, rw),
        )

    def _update_chessboard_position(self) -> None:
        """Update chessboard frame position based on window size"""
        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()

        rely = self.style.chessboard_rely
        relx = self.utils.relative_dimensions(rely, (screen_height, screen_width))
        relwidth = 1 - relx * 2
        relheight = 1 - rely * 2

        self.chessboard_frame.place(
            relx=relx, rely=rely, relwidth=relwidth, relheight=relheight
        )

        # Button positions
        resign_relx = relx - relwidth / 8
        resign_rely = rely + (relheight / 8) * 3
        self._place_resign_button(resign_relx, resign_rely, relwidth / 8, relheight / 8)

        settings_relx = resign_relx
        settings_rely = resign_rely + relheight / 8
        self._place_settings_button(
            settings_relx, settings_rely, relwidth / 8, relheight / 8
        )

        flip_relx = relx + relwidth
        flip_rely = rely + (relheight / 8) * 3
        self._place_flip_button(flip_relx, flip_rely, relwidth / 8, relheight / 8)

        if self.promoting:
            self._place_promotion_frame(relx, rely)

    def _place_promotion_frame(self, relx: float, rely: float) -> None:
        self.promotion_frame.place(
            relx=relx + (1 - relx * 2) + 0.05,
            rely=rely + (1 - rely * 2) / 4,
            relheight=(1 - rely * 2) / 2,
            relwidth=(1 - relx * 2) / 8,
        )

    def _hide_promotion_frame(self) -> None:
        self.promotion_frame.place_forget()

    def _configure_grid(self) -> None:
        """Configure the 8x8 grid for the chessboard"""
        for i in range(8):
            self.chessboard_frame.rowconfigure(i, weight=1)
            self.chessboard_frame.columnconfigure(i, weight=1)

    def _create_chessboard_squares(self) -> None:
        """Create all 64 chessboard squares with alternating colors and a single label per square"""
        color = self.style.white_box_color
        color2 = self.style.black_box_color

        self.chessboard_squares.clear()
        self.piece_labels.clear()
        self._square_color_state.clear()
        self._render_state.clear()
        self._current_legal_visual_squares.clear()
        self._highlighted_visual_squares.clear()

        for row in range(8):
            for col in range(8):
                square_pos = (row, col)

                # 1. Create the background square
                square = ctk.CTkFrame(
                    self.chessboard_frame, fg_color=color, bg_color=color
                )
                square.grid(row=row, column=col, sticky="nsew")
                square.bind(
                    "<Button-1>",
                    lambda _, pos=square_pos: self._handle_square_click(pos),
                )
                self.chessboard_squares[square_pos] = square
                self._square_color_state[square_pos] = color

                # 2. Pre-allocate a single label for both piece and legal move dot
                piece_label = ctk.CTkLabel(
                    square, text="", fg_color="transparent", bg_color="transparent"
                )
                piece_label.place(relx=0.5, rely=0.5, anchor="center")
                piece_label.bind(
                    "<Button-1>",
                    lambda _, pos=square_pos: self._handle_square_click(pos),
                )
                self.piece_labels[square_pos] = piece_label

                color, color2 = color2, color
            color, color2 = color2, color

    # ==================== PIECE RENDERING ====================

    @staticmethod
    def _normalize_piece_key(piece: Any) -> str:
        """Normalize piece id (e.g. '-p1' -> '-p', 0 -> 'empty')."""
        if piece != 0 and isinstance(piece, str):
            return piece[:-1]
        return "empty"

    def _square_base_color(self, visual_square: tuple[int, int]) -> str:
        row, col = visual_square
        return (
            self.style.white_box_color
            if (row + col) % 2 == 0
            else self.style.black_box_color
        )

    def _set_square_color(self, visual_square: tuple[int, int], color: str) -> None:
        frame = self.chessboard_squares.get(visual_square)
        if not frame:
            return
        if self._square_color_state.get(visual_square) == color:
            return
        frame.configure(fg_color=color, bg_color=color)
        self._square_color_state[visual_square] = color

    def _get_highlight_targets(self) -> set[tuple[int, int]]:
        targets: set[tuple[int, int]] = set()
        if self.last_from_square and self.last_to_square:
            targets.add(self._logical_to_visual(self.last_from_square))
            targets.add(self._logical_to_visual(self.last_to_square))
        return targets

    def _update_highlights_incremental(self, force: bool = False) -> None:
        """Update highlight colors by touching only changed squares."""
        new_highlights = self._get_highlight_targets()

        if force:
            stale_highlights = set(self._all_visual_squares) - new_highlights
        else:
            stale_highlights = self._highlighted_visual_squares - new_highlights

        for visual_sq in stale_highlights:
            self._set_square_color(visual_sq, self._square_base_color(visual_sq))

        if force:
            new_or_changed = new_highlights
        else:
            new_or_changed = new_highlights - self._highlighted_visual_squares

        for visual_sq in new_or_changed:
            is_light = (visual_sq[0] + visual_sq[1]) % 2 == 0
            highlight = (
                self.style.white_highlight if is_light else self.style.black_highlight
            )
            self._set_square_color(visual_sq, highlight)

        self._highlighted_visual_squares = new_highlights

    def _compute_legal_visual_squares(
        self, piece_id: Optional[str]
    ) -> set[tuple[int, int]]:
        legal_visual_squares: set[tuple[int, int]] = set()
        if not piece_id:
            return legal_visual_squares

        legal_moves = self._get_legal_moves_for_piece(piece_id)
        for move in legal_moves:
            visual_sq = self._logical_to_visual((int(move[0]), int(move[1])))
            legal_visual_squares.add(visual_sq)

        return legal_visual_squares

    def _warm_image_cache_for_size(self, size: int) -> None:
        """Preload piece+legal image variants once per size to avoid first-click hitching."""
        if size <= 0 or size in self._warmed_image_sizes:
            return

        piece_variants = [
            "empty",
            "p",
            "r",
            "n",
            "b",
            "q",
            "k",
            "-p",
            "-r",
            "-n",
            "-b",
            "-q",
            "-k",
        ]
        for piece_variant in piece_variants:
            for is_legal in (False, True):
                piece_for_loader: Any = (
                    0 if piece_variant == "empty" else f"{piece_variant}1"
                )
                self._get_cached_ctk_image(piece_for_loader, is_legal, size)

        self._warmed_image_sizes.add(size)

    def _refresh_squares(
        self,
        visual_squares: set[tuple[int, int]],
        legal_visual_squares: set[tuple[int, int]],
        image_size: int,
        force: bool = False,
    ) -> None:
        """Refresh only specified squares using current board state."""
        if not visual_squares:
            return

        for visual_sq in visual_squares:
            row, col = visual_sq
            piece = self.view_matrix[row, col]
            is_legal = visual_sq in legal_visual_squares
            self._render_square(visual_sq, piece, is_legal, image_size, force=force)

        self._last_image_size = image_size

    def _get_cached_ctk_image(
        self, piece: Any, is_legal: bool, size: int
    ) -> Optional[ctk.CTkImage]:
        """Fetch/create and fully cache the CTkImage to prevent lag during rapid clicks"""
        base_piece = self._normalize_piece_key(piece)

        cache_key = (base_piece, is_legal, size)

        if cache_key in self._ctk_image_cache:
            return self._ctk_image_cache[cache_key]

        try:
            # 1. Generate Base PIL Image
            if base_piece == "empty":
                final_pil = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            else:
                color = "black" if "-" in base_piece else "white"
                path = f"images/{color}/{base_piece.replace('-', '')}.png"
                final_pil = (
                    Image.open(path)
                    .convert("RGBA")
                    .resize((size, size), Image.Resampling.LANCZOS)
                )

            # 2. Add Dot if legal
            if is_legal:
                dot_pil = self._dot_pil_base.resize(
                    (size, size), Image.Resampling.LANCZOS
                )
                if base_piece == "empty":
                    final_pil = dot_pil
                else:
                    final_pil = Image.alpha_composite(final_pil, dot_pil)

            # 3. Create CTkImage and Cache it permanently for this size
            ctk_img = ctk.CTkImage(final_pil, size=(size, size))
            self._ctk_image_cache[cache_key] = ctk_img
            return ctk_img

        except Exception as e:
            database.gamelogger.error(f"Image load error: {e}")
            return None

    def _render_square(
        self,
        visual_square: tuple[int, int],
        piece: Any,
        is_legal: bool,
        image_size: int,
        force: bool = False,
    ) -> None:
        """Fetch the pre-composited image and apply it only if visual state changed."""
        label = self.piece_labels.get(visual_square)
        if not label:
            return

        render_key = (self._normalize_piece_key(piece), is_legal, image_size)
        if not force and self._render_state.get(visual_square) == render_key:
            return

        ctk_img = self._get_cached_ctk_image(piece, is_legal, image_size)
        if ctk_img:
            label.configure(image=ctk_img)
            self._render_state[visual_square] = render_key

    def _calculate_image_size(self) -> int:
        """Calculate appropriate image size ONCE for the whole board"""
        square_frame = self.chessboard_squares.get((0, 0))
        if not square_frame:
            return 80

        square_size = min(square_frame.winfo_width(), square_frame.winfo_height())
        if square_size <= 1:
            return 80

        image_size = int(square_size * 0.9)
        return max(image_size, 40)

    def _refresh_all_images(self, force: bool = False) -> None:
        """Refresh board visuals, skipping unchanged square renders whenever possible."""
        legal_visual_squares = self._compute_legal_visual_squares(self.piece_selected)
        current_size = self._calculate_image_size()
        size_changed = current_size != self._last_image_size
        if current_size not in self._warmed_image_sizes:
            self._warm_image_cache_for_size(current_size)

        if force or size_changed or not self._render_state:
            squares_to_refresh = set(self._all_visual_squares)
            force_render = True
        else:
            squares_to_refresh = (
                self._current_legal_visual_squares.symmetric_difference(
                    legal_visual_squares
                )
            )
            for row in range(8):
                for col in range(8):
                    visual_sq = (row, col)
                    piece = self.view_matrix[row, col]
                    render_key = (
                        self._normalize_piece_key(piece),
                        visual_sq in legal_visual_squares,
                        current_size,
                    )
                    if self._render_state.get(visual_sq) != render_key:
                        squares_to_refresh.add(visual_sq)
            force_render = False

        self._current_legal_visual_squares = legal_visual_squares
        self._refresh_squares(
            squares_to_refresh, legal_visual_squares, current_size, force=force_render
        )
        self._update_highlights_incremental(force=force or size_changed)

    def _reapply_highlights(self) -> None:
        """Compatibility wrapper for full highlight reapplication."""
        self._update_highlights_incremental(force=True)

    def _show_fen_board(self, fen: str) -> None:
        """Display the FEN on the board"""
        board = self.utils.notations.generate_matrix(fen)
        if self.flipped:
            board = np.flip(board, axis=(0, 1))
        self.view_matrix = board

        # Clear move highlights while viewing history
        self.last_from_square = None
        self.last_to_square = None

        self._refresh_all_images(force=True)

        self.disabled_color = ["black", "white"]

    def _reset_board_to_matrix(self) -> None:
        """Display the database matrix on the board with flipped if required"""
        # Use a reference/view instead of .copy() so that subsequent moves
        # to database.matrix are reflected in self.view_matrix automatically.
        if self.flipped:
            self.view_matrix = np.flip(database.matrix, axis=(0, 1))
        else:
            self.view_matrix = database.matrix

        # Restore last-move highlights from game history
        if database.game_history:
            last_uci = database.game_history[-1]
            self.last_from_square = self.utils.coords.chess_to_matrix(last_uci[:2])
            self.last_to_square = self.utils.coords.chess_to_matrix(last_uci[2:4])

        self._refresh_all_images(force=True)

        if self.game_mode == "pass_n_play":
            self.disabled_color = [0, 0]
        else:
            self.disabled_color = [
                "white"
                if self.vs_ai_configurations
                and self.vs_ai_configurations.get("color") == "black"
                else "black",
                0,
            ]

        self.current_fen = len(database.fen_history) - 1

    def _on_left_arrow_click(self, _=None) -> None:
        """Handle left arrow click"""
        if self.game_mode is None:
            return
        if self.current_fen <= 0:
            return

        self.current_fen -= 1
        self._show_fen_board(database.fen_history[self.current_fen])

    def _on_right_arrow_click(self, _=None) -> None:
        """Handle right arrow click"""
        if self.game_mode is None:
            return
        if self.current_fen >= len(database.fen_history) - 1:
            return

        self.current_fen += 1

        if self.current_fen == len(database.fen_history) - 1:
            self._reset_board_to_matrix()
        else:
            self._show_fen_board(database.fen_history[self.current_fen])

    def _on_ctrl_arrow_click(self, arrow: Literal["left", "right"]):
        if self.game_mode is None:
            return
        if arrow == "left":
            if len(database.fen_history) <= 1:
                return
            self.current_fen = 0
            self._show_fen_board(database.fen_history[0])
        else:
            self.current_fen = len(database.fen_history) - 1
            self._reset_board_to_matrix()

    # ==================== START MENU ====================

    def _show_start_menu(self) -> None:
        """Display the start menu overlay"""
        self._hide_footer_signature()
        overlay = ctk.CTkFrame(
            self.root, fg_color=("gray50", "gray20"), bg_color="transparent"
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.start_menu = overlay

        # Create background
        self._create_start_menu_background(overlay)

        # Create buttons
        self._create_start_menu_buttons(overlay)
        self._lift_persistent_widgets()

    def _create_start_menu_background(self, parent: ctk.CTkFrame) -> None:
        """Create background image for start menu"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        self.start_menu_background = ctk.CTkLabel(
            parent,
            image=self.utils.ctkimage_generator(
                "images/start/start_background.png", size=(width, height)
            ),
            text="",
        )
        self.start_menu_background.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _create_start_menu_buttons(self, parent: ctk.CTkFrame) -> None:
        """Create all start menu buttons"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # Button click handler
        def button_click(mode: Literal["pass_n_play", "vs_ai"]):
            self.game_mode = mode
            self.flip_allowed = mode == "pass_n_play"
            parent.destroy()
            if mode == "vs_ai":
                self._show_vs_ai_menu()
                self.root.after(100, self.utils.ai.add_stockfish)
                return
            # Pass n Play game - set up PGN metadata
            self._setup_pgn_metadata("Pass and Play", "Player 1", "Player 2", "-", "-")
            self._show_footer_signature()
            self.root.after(10, self._refresh_all_images)

        # Button configurations
        buttons = [
            (
                "pass_n_play",
                "images/start/buttons/pass_n_play.png",
                self.style.pass_play_rely,
                lambda: button_click("pass_n_play"),
                "normal",
            ),
            (
                "vs_ai",
                "images/start/buttons/vs_ai.png",
                self.style.vs_ai_rely,
                lambda: button_click("vs_ai"),
                "normal",
            ),
            (
                "coming_soon",
                "images/start/buttons/coming_soon.png",
                self.style.coming_soon_rely,
                None,
                "disabled",
            ),
        ]

        for name, img_path, rely, command, state in buttons:
            btn = self._create_menu_button(
                parent,
                img_path,
                (
                    int(width * self.style.button_relwidth),
                    int(height * self.style.button_relheight),
                ),
                command if state == "normal" else None,
                state,
                name,
            )
            btn.place(
                relx=self.style.button_relx,
                rely=rely,
                relwidth=self.style.button_relwidth,
                relheight=self.style.button_relheight,
            )

    def _create_menu_button(
        self,
        parent: ctk.CTkFrame,
        image_path: str,
        image_size: tuple[int, int],
        command: Optional[Any],
        state: str,
        name: str,
    ) -> ctk.CTkButton:
        """Helper to create a styled menu button"""
        btn = ctk.CTkButton(
            parent,
            text="",
            bg_color=self.style.start_menu_bg,
            fg_color=self.style.start_menu_fg,
            border_color=self.style.start_menu_border,
            border_width=self.style.start_menu_border_width,
            hover_color=self.style.start_menu_hover,
            image=self.utils.ctkimage_generator(image_path, size=image_size),
            command=command if state == "normal" else lambda: None,
            state=state,
        )

        # Cursor bindings (DRY principle)
        cursor_style = "hand2" if state == "normal" else "no"
        self._bind_cursor(btn, cursor_style)

        return btn

    def _bind_cursor(self, widget, hover_cursor: str) -> None:
        """Helper to bind cursor changes to a widget"""
        widget.bind(
            "<Enter>", lambda e, style=hover_cursor: self._change_cursor(e, style)
        )
        widget.bind("<Leave>", lambda e, style="arrow": self._change_cursor(e, style))

    def _change_cursor(self, event, style: str) -> None:
        """Change cursor style for a widget"""
        event.widget.configure(cursor=style)

    def _refresh_start_menu(self) -> None:
        """Refresh start menu on resize"""
        if self.start_menu:
            self.start_menu.destroy()
        self._show_start_menu()

    # ==================== VS AI MENU ====================

    def _show_vs_ai_menu(self) -> None:
        """Display the vs AI menu overlay"""
        self._hide_footer_signature()
        overlay = ctk.CTkFrame(
            self.root, fg_color=("gray50", "gray20"), bg_color="transparent"
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.vs_ai_menu = overlay

        # Create background
        self._create_vs_ai_menu_background(overlay)

        # Create buttons
        self._create_vs_ai_menu_elements(overlay)
        self._lift_persistent_widgets()

    def _create_vs_ai_menu_background(self, parent: ctk.CTkFrame) -> None:
        """Create background image for vs AI menu"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        self.vs_ai_menu_background = ctk.CTkLabel(
            parent,
            image=self.utils.ctkimage_generator(
                "images/vs_ai/background.png", size=(width, height)
            ),
            text="",
        )
        self.vs_ai_menu_background.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _create_vs_ai_menu_elements(self, parent: ctk.CTkFrame) -> None:
        """Create all vs AI menu buttons"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # Calculate responsive sizes
        frame_width = int(width * self.style.selector_relwidth)
        frame_height = int(height * 0.17)

        # Combobox select command
        def on_select(event, name: str):
            if self.vs_ai_configurations is None:
                self.vs_ai_configurations = {}
            if name == "elo":
                self.vs_ai_configurations[name] = int(event[: event.find("-")])
            else:
                if event == "Random":
                    event = np.random.choice(["White", "Black"])
                self.vs_ai_configurations[name] = event.lower()

        # Combobox configurations with all ELO levels
        combos = [
            (
                "Elo:",
                "Select Elo Rating",
                [
                    "400-Absolute Beginner",
                    "600-Novice",
                    "800-Beginner",
                    "1000-Casual Player",
                    "1200-Club Player",
                    "1400-Intermediate",
                    "1600-Advanced",
                    "1800-Strong Player",
                    "2000-Expert",
                    "2200-Master",
                    "2400-International Master",
                    "2600-Grandmaster",
                    "2800-Super GM",
                    "3000-Maximum",
                ],
                lambda event: on_select(event, "elo"),
                0.35,
            ),
            (
                "Color:",
                "Select Your Color",
                ["White", "Black", "Random"],
                lambda event: on_select(event, "color"),
                0.6,
            ),
        ]

        for label_text, placeholder_text, values, command, rely in combos:
            frame, combobox = self.utils.create_selector_frame(
                parent,
                label_text,
                placeholder_text,
                values,
                command,
                width=frame_width,
                height=frame_height,
            )

            self._bind_cursor(combobox, "hand2")
            frame.place(relx=self.style.selector_relx, rely=rely, anchor="center")

        self._create_vs_ai_start_button(parent, width, height)

    def _start_ai_game(self, parent):
            # Validate selections
            if not self.vs_ai_configurations or len(self.vs_ai_configurations) < 2:
                messagebox.showwarning(
                    "Incomplete Selection", "Please select both Elo rating and color!"
                )
                return

            elo = self.vs_ai_configurations["elo"]
            color = self.vs_ai_configurations["color"]
            color2 = "white" if color == "black" else "black"

            # Destroy menu and start game
            if parent:
                parent.destroy()
            self._show_footer_signature()

            self._flip_board(color)
            self._disable_color(color2)
            self.root.after(10, self._refresh_all_images)

            database.gamelogger.game(f"VS AI | ELO: {elo} | Color: {color}")

            white_name = "Player" if color == "white" else f"Stockfish ({elo})"
            black_name = "Player" if color == "black" else f"Stockfish ({elo})"
            white_elo = "-" if color == "white" else str(elo)
            black_elo = "-" if color == "black" else str(elo)
            self._setup_pgn_metadata("VS AI", white_name, black_name, white_elo, black_elo)

            # Configure AI and RESET its memory
            if database.stockfish is None:
                self.utils.ai.add_stockfish()
            else:
                # Tell the existing Stockfish engine process to forget the old game
                # and set up a fresh board.
                database.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

            self.utils.ai.configure_strength(elo)
            self._cancel_pending_ai_work()

            if color == "black":
                self._execute_stockfish_move()

    def _create_vs_ai_start_button(
        self, parent: ctk.CTkFrame, width: int, height: int
    ) -> None:
        """Create start button for VS AI menu"""

        button_width = int(width * 0.3)
        button_height = int(height * 0.06)
        button_font_size = max(14, min(int(height * 0.02), 20))

        start_btn = ctk.CTkButton(
            parent,
            text="START GAME",
            font=("Georgia", button_font_size, "bold"),
            fg_color="#2a2420",
            hover_color="#c9a961",
            border_color="#c9a961",
            border_width=3,
            text_color="#c9a961",
            width=button_width,
            height=button_height,
            corner_radius=8,
            command=lambda: self._start_ai_game(parent),
        )

        self._bind_cursor(start_btn, "hand2")

        start_btn.place(relx=0.5, rely=0.82, anchor="center")

    def _refresh_vs_ai_menu(self) -> None:
        """Refresh vs AI menu on resize"""
        if self.vs_ai_menu:
            self.vs_ai_menu.destroy()
        self._show_vs_ai_menu()

    # ==================== GAME LOGIC ====================

    def _get_legal_moves_for_piece(self, piece: str) -> np.ndarray:
        """Get legal moves for a specific piece"""
        if not piece or piece == 0 or not isinstance(piece, str):
            return np.array([])

        color = "white" if "-" not in piece else "black"
        legal_moves_dict = database.get_legal_moves(color)
        return legal_moves_dict.get(piece, np.array([]))

    def _is_legal_move(self, piece: str, target_square: tuple[int, int]) -> bool:
        """Check if a move is legal for the given piece"""
        if not piece or piece == 0 or not isinstance(piece, str):
            return False

        legal_moves = self._get_legal_moves_for_piece(piece)
        return any(np.array_equal(target_square, move) for move in legal_moves)

    def _visual_to_logical(self, visual_square: tuple[int, int]) -> tuple[int, int]:
        """Convert visual grid position to logical matrix position"""
        if self.flipped:
            return (7 - visual_square[0], 7 - visual_square[1])
        return visual_square

    def _logical_to_visual(self, logical_square: tuple[int, int]) -> tuple[int, int]:
        """Convert logical matrix position to visual grid position"""
        if self.flipped:
            return (7 - logical_square[0], 7 - logical_square[1])
        return logical_square

    def _start_promotion(
       self,
       color: str,
       target_square: tuple[int, int],
       from_square: tuple[int, int],
       base: Optional[str] = None,
   ) -> None:
       piece = database.matrix[from_square[0], from_square[1]]

       database.matrix[from_square[0], from_square[1]] = 0
       database.current_turn = "black" if database.current_turn == "white" else "white"
       if database.current_turn == "white":
           database.fullmove += 1

       if "-" in piece:
           database.black_last_pawn = None
           database.black_pieces = database.black_pieces[database.black_pieces != piece]
       else:
           database.white_last_pawn = None
           database.white_pieces = database.white_pieces[database.white_pieces != piece]

       # Always show the UI — AI promotions are handled before this is called
       self._setup_promotion_images(color)

    def _end_promotion(self, base: str) -> None:
        """Finalize pawn promotion by placing the promoted piece"""
        if self.promoting_square is None or self.promotion_from_square is None:
            return

        to_square = self.promoting_square
        from_square = self.promotion_from_square

        # Determine the promoted piece
        is_black = to_square[0] == 7  # Black pawns promote on rank 7 (row 7 in matrix)
        promoted_piece = self._get_promoted_piece(base, is_black)

        # Capture target must be read before writing promoted piece
        captured_piece = database.matrix[to_square[0], to_square[1]]
        is_capture = captured_piece != 0

        # Place the promoted piece on the board
        self.matrix[to_square[0], to_square[1]] = promoted_piece

        # Update piece lists
        pawn_piece = "-p" if is_black else "p"
        database.black_pieces = (
            database.black_pieces[database.black_pieces != pawn_piece]
            if is_black
            else database.black_pieces
        )
        database.white_pieces = (
            database.white_pieces[database.white_pieces != pawn_piece]
            if not is_black
            else database.white_pieces
        )

        # Remove captured piece if it exists
        if is_capture:
            if is_black:
                database.white_pieces = database.white_pieces[
                    database.white_pieces != captured_piece
                ]
            else:
                database.black_pieces = database.black_pieces[
                    database.black_pieces != captured_piece
                ]

        # Add promoted piece to piece list
        if is_black:
            database.black_pieces = np.append(database.black_pieces, promoted_piece)
        else:
            database.white_pieces = np.append(database.white_pieces, promoted_piece)

        self.promoting = False
        self.promoting_square = None
        self.promotion_from_square = None
        self._hide_promotion_frame()
        self.utils.legal_moves.update_legal_moves(database.matrix)
        self._refresh_all_images()

        # Generate notation with check/checkmate detection
        if self._check_game_over() == "checkmate":
            chess_notation = self.utils.notations.chess_notation(
                "p",
                to_square,
                promotion=base,
                capture=is_capture,
                from_square=from_square,
                checkmate=True,
            )
        else:
            if self.utils.legal_moves.check_checker(database.current_turn):
                chess_notation = self.utils.notations.chess_notation(
                    "p",
                    to_square,
                    promotion=base,
                    capture=is_capture,
                    from_square=from_square,
                    check=True,
                )
            else:
                chess_notation = self.utils.notations.chess_notation(
                    "p",
                    to_square,
                    promotion=base,
                    capture=is_capture,
                    from_square=from_square,
                )

        stockfish_notation = self.utils.notations.stockfish_notation(
            from_square, to_square, promotion=base
        )

        # Store in game history
        database.game_history.append(stockfish_notation)
        database.game_pgn.append(chess_notation)
        database.fen_history.append(
            self.utils.notations.generate_fen(
                side_to_move=database.current_turn[0], fullmove_number=database.fullmove
            )
        )  # type: ignore
        self.current_fen = len(database.fen_history) - 1
        self.review.evaluate_last_move_async()

        database.gamelogger.move(f"Move: {chess_notation}")

        # Update Stockfish
        self.utils.ai.update_last_move()

        # Trigger AI move if needed
        if self.game_mode == "vs_ai" and self.vs_ai_configurations:
            ai_color = (
                "black" if self.vs_ai_configurations["color"] == "white" else "white"
            )
            if database.current_turn == ai_color:
                self._execute_stockfish_move()

    def _get_promoted_piece(self, promotion_choice: str, is_black: bool) -> str:
        """Convert promotion choice (q, r, b, n) to piece identifier"""
        # Use the utils method if available
        return self.utils.next_piece(promotion_choice, "black" if is_black else "white")

    def _make_result_btn(
        self,
        parent: "ctk.CTkFrame",
        text: str,
        fg: str,
        hover: str,
        width: int,
        height: int,
        font_size: int = 13,
        command=None,
    ) -> "ctk.CTkFrame":
        """
        Shadowed rounded button.
        Wrapper = (width + _DLG_SHADOW) × (height + _DLG_SHADOW).
        Dark shadow frame placed at (+shadow, +shadow), button face at (0, 0).
        """
        s = self.style._DLG_SHADOW
        wrapper = ctk.CTkFrame(
            parent, fg_color="transparent", width=width + s, height=height + s
        )
        wrapper.pack_propagate(False)
        wrapper.grid_propagate(False)

        # Shadow
        ctk.CTkFrame(
            wrapper,
            width=width,
            height=height,
            corner_radius=10,
            fg_color=self.style._DLG_SHADOW_CLR,
        ).place(x=s, y=s)

        # Button face
        ctk.CTkButton(
            wrapper,
            text=text,
            width=width,
            height=height,
            corner_radius=10,
            fg_color=fg,
            hover_color=hover,
            text_color=self.style._DLG_WHITE,
            font=ctk.CTkFont("Helvetica", font_size, "bold"),
            command=command,
            cursor="hand2",
        ).place(x=0, y=0)

        return wrapper

    def _make_stat_box(
        self,
        parent: "ctk.CTkFrame",
        image: "ctk.CTkImage",
        count: str,
        label: str,
        label_color: str,
        box_w: int,
        box_h: int,
    ) -> "ctk.CTkFrame":
        """
        Single stat box:  circular icon  +  count  +  label text.
        circle = CTkFrame with corner_radius = half its size.
        """
        box = ctk.CTkFrame(
            parent,
            fg_color=self.style._DLG_STAT_BG,
            corner_radius=10,
            width=box_w,
            height=box_h,
        )
        box.pack_propagate(False)

        # Circular icon
        classifier_frame = ctk.CTkFrame(
            box, width=42, height=42, fg_color = "transparent", bg_color = "transparent"
        )
        classifier_frame.pack(pady=(12, 2))
        classifier_frame.pack_propagate(False)

        ctk.CTkLabel(
            classifier_frame,
            text="",
            image=image,
            font=ctk.CTkFont(self.style._EMOJI_FONT_FAMILY, 15, "bold"),
            text_color=self.style._DLG_WHITE,
            fg_color="transparent",
            bg_color="transparent"
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Count
        ctk.CTkLabel(
            box,
            text=count,
            font=ctk.CTkFont("Helvetica", 19, "bold"),
            text_color=self.style._DLG_WHITE,
            fg_color="transparent",
        ).pack()

        # Label
        ctk.CTkLabel(
            box,
            text=label,
            font=ctk.CTkFont("Helvetica", 11, "bold"),
            text_color=label_color,
            fg_color="transparent",
        ).pack(pady=(0, 8))

        return box

    def _show_game_over_dialog(
        self, result: str, winner: "Optional[str]" = None
    ) -> None:
        """
        Chess.com-style game-over card.
        """
        # Save state for window resizing
        self._last_game_result = result
        self._last_game_winner = winner

        self._hide_footer_signature()

        # ── Full-window dark overlay ──────────────────────────────────────────
        overlay = ctk.CTkFrame(
            self.root, fg_color=("gray50", "gray20"), bg_color="transparent"
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Track the overlay frame for the resize event
        self.game_over_menu = overlay

        self._lift_persistent_widgets()

        sw = self.root.winfo_width()
        sh = self.root.winfo_height()

        # ── Card sizing (Adjusted for better small-screen scaling) ────────────
        card_w = max(360, min(int(sw * 0.52), 620))
        card_h = max(380, min(int(sh * 0.85), 560)) # Lowered floor to 380
        scale = card_h / 520  # scale fonts/buttons proportionally

        # ── Card shadow ───────────────────────────────────────────────────────
        ctk.CTkFrame(
            overlay,
            width=card_w,
            height=card_h,
            corner_radius=14,
            fg_color=self.style._DLG_SHADOW_CLR,
        ).place(
            relx=0.5,
            rely=0.5,
            anchor="center",
            x=self.style._DLG_SHADOW,
            y=self.style._DLG_SHADOW,
        )

        # ── Card body ─────────────────────────────────────────────────────────
        card = ctk.CTkFrame(
            overlay,
            width=card_w,
            height=card_h,
            corner_radius=14,
            fg_color=self.style._DLG_CARD_BG,
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)

        # ── HEADER ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(
            card, fg_color=self.style._DLG_HEADER_BG, corner_radius=14
        )
        header.pack(fill="x")

        # Build title / subtitle strings
        if result == "checkmate":
            title_text = "Checkmate!"
            subtitle = f"{winner.capitalize()} Wins!" if winner else ""
        elif result == "resign":
            title_text = "Resignation!"
            subtitle = f"{winner.capitalize()} Wins!" if winner else ""
        else:
            title_text = "Stalemate!"
            subtitle = "The game is a draw"

        title_fs = max(20, int(28 * scale))
        subtitle_fs = max(12, int(14 * scale))

        # Crown + title row
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(pady=(int(18 * scale), int(4 * scale)))

        ctk.CTkLabel(
            title_row,
            text="👑",
            font=ctk.CTkFont(self.style._EMOJI_FONT_FAMILY, max(16, int(22 * scale))),
            text_color=self.style._DLG_WHITE,
            fg_color="transparent",
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            title_row,
            text=title_text,
            font=ctk.CTkFont("Helvetica", title_fs, "bold"),
            text_color=self.style._DLG_WHITE,
            fg_color="transparent",
        ).pack(side="left")

        ctk.CTkLabel(
            title_row,
            text="👑",
            font=ctk.CTkFont(self.style._EMOJI_FONT_FAMILY, max(16, int(22 * scale))),
            text_color=self.style._DLG_WHITE,
            fg_color="transparent",
        ).pack(side="left", padx=(10, 0))

        # Subtitle
        ctk.CTkLabel(
            header,
            text=subtitle,
            font=ctk.CTkFont("Helvetica", subtitle_fs),
            text_color=self.style._DLG_GRAY,
            fg_color="transparent",
        ).pack(pady=(0, int(16 * scale)))

        # Separator
        ctk.CTkFrame(
            header, height=2, fg_color=self.style._DLG_SEP, corner_radius=0
        ).pack(fill="x")

        # ── STAT BOXES ────────────────────────────────────────────────────────

        stats_frame = ctk.CTkFrame(card, fg_color="transparent")
        stats_frame.pack(padx=20, pady=int(18 * scale), fill="x")
        stats_frame.columnconfigure((0, 1, 2), weight=1)

        box_w = max(100, (card_w - 40 - 32) // 3)
        box_h = max(80, int(108 * scale))

        images = self.classifier_images

        # ── NEW ANIMATION CODE GOES HERE ──────────────────────────────────────

        icon_labels = []
        count_labels = []
        type_labels = []

        for col in range(3):
            box = ctk.CTkFrame(
                stats_frame,
                fg_color=self.style._DLG_STAT_BG,
                corner_radius=10,
                width=box_w,
                height=box_h,
            )
            box.grid(row=0, column=col, padx=4, sticky="nsew")
            box.pack_propagate(False)
            box.grid_propagate(False)

            icon_lbl = ctk.CTkLabel(
                box, text="", image=self.empty_image,
                fg_color="transparent", bg_color="transparent"
            )
            icon_lbl.pack(pady=(12, 2))

            count_lbl = ctk.CTkLabel(
                box, text="-",
                font=ctk.CTkFont("Helvetica", 19, "bold"),
                text_color=self.style._DLG_WHITE,
                fg_color="transparent",
            )
            count_lbl.pack()

            type_lbl = ctk.CTkLabel(
                box, text="Analyzing...",
                font=ctk.CTkFont("Helvetica", 11, "bold"),
                text_color=self.style._DLG_GRAY,
                fg_color="transparent",
            )
            type_lbl.pack(pady=(0, 8))

            icon_labels.append(icon_lbl)
            count_labels.append(count_lbl)
            type_labels.append(type_lbl)

        animation_ticks = 0
        all_icon_keys = list(images.keys())

        def show_stats():
            nonlocal animation_ticks

            if not icon_labels[0].winfo_exists():
                return

            if animation_ticks < 4 or not self.review.is_summary_ready():
                rng = np.random.default_rng()
                random_choices = rng.choice(all_icon_keys, size=3, replace=False)

                for i, choice in enumerate(random_choices):
                    icon_labels[i].configure(image=images.get(choice, self.empty_image))
                    count_labels[i].configure(text="-")
                    type_labels[i].configure(
                        text="Analyzing...", text_color=self.style._DLG_GRAY
                    )

                animation_ticks += 1
                self.root.after(120, show_stats)

            else:
                if self.game_mode == "pass_n_play":
                    color = winner if winner else "white"
                else:
                    color = self.vs_ai_configurations["color"] #type: ignore

                summary_data = self.review.get_summary().get(color, {}).get("move_types", {})

                data = []
                for key, value in summary_data.items():
                    if len(data) >= 3:
                        break
                    if int(value) > 0:
                        if key.lower() in ["brilliant", "great"]:
                            label_clr = self.style._DLG_BLUE
                        elif key.lower() in ["best", "excellent", "good"]:
                            label_clr = self.style._DLG_GREEN_ACC
                        elif key.lower() in ["mistake", "inaccuracy"]:
                            label_clr = self.style._DLG_YELLOW
                        else:
                            label_clr = self.style._DLG_RED
                        data.append((images.get(key, self.empty_image), str(value), key.title(), label_clr))

                while len(data) < 3:
                    data.append((self.empty_image, "0", "-", self.style._DLG_GRAY))

                for i, (img, cnt, lbl, lbl_clr) in enumerate(data):
                    icon_labels[i].configure(image=img)
                    count_labels[i].configure(text=cnt)
                    type_labels[i].configure(text=lbl, text_color=lbl_clr)

        show_stats()

        # ── BUTTONS ───────────────────────────────────────────────────────────
        btn_area = ctk.CTkFrame(card, fg_color="transparent")
        btn_area.pack(fill="x", padx=22, pady=(0, int(16 * scale)))

        BW = card_w - 44  # full usable width
        s = self.style._DLG_SHADOW
        bfs = max(11, int(13 * scale))  # button font size

        # ── Game Review  (full-width green) ──────────────────────────────────
        self._make_result_btn(
            btn_area,
            "Game Review",
            self.style._DLG_BTN_GREEN,
            self.style._DLG_GREEN_HOV,
            width=BW - s,
            height=max(36, int(52 * scale)), # Lowered minimum height
            font_size=max(13, int(15 * scale)),
            command=lambda: print("Game Review clicked"),
        ).pack(pady=(0, int(12 * scale)))

        # ── Rematch | New Game ────────────────────────────────────────────────
        row2 = ctk.CTkFrame(btn_area, fg_color="transparent")
        row2.pack(fill="x", pady=(0, int(10 * scale)))

        hw = (BW - 10 - 2 * s) // 2
        bh = max(32, int(46 * scale)) # Lowered minimum height

        self._make_result_btn(
            row2,
            "Rematch",
            self.style._DLG_BTN_DARK,
            self.style._DLG_DARK_HOV,
            hw,
            bh,
            bfs,
            command=lambda: self._rematch_game(overlay),
        ).pack(side="left")

        ctk.CTkFrame(row2, fg_color="transparent", width=10, height=1).pack(side="left")

        self._make_result_btn(
            row2,
            "New Game",
            self.style._DLG_BTN_DARK,
            self.style._DLG_DARK_HOV,
            hw,
            bh,
            bfs,
            command=lambda: self._start_new_game(overlay),
        ).pack(side="left")

        # ── View Board | Copy PGN | Save PGN ─────────────────────────────────
        row3 = ctk.CTkFrame(btn_area, fg_color="transparent")
        row3.pack(fill="x")

        tw = (BW - 20 - 3 * s) // 3
        tbh = max(30, int(44 * scale)) # Lowered minimum height

        btn3 = [
            ("View Board", lambda: self._view_board_with_menu(overlay, result, winner)),
            ("Copy PGN", self._copy_pgn_to_clipboard),
            ("Save PGN", self._save_pgn_to_file),
        ]

        for i, (lbl, cmd) in enumerate(btn3):
            self._make_result_btn(
                row3,
                lbl,
                self.style._DLG_BTN_DARK,
                self.style._DLG_DARK_HOV,
                tw,
                tbh,
                max(10, int(12 * scale)),
                command=cmd,
            ).pack(side="left")
            if i < 2:
                ctk.CTkFrame(row3, fg_color="transparent", width=10, height=1).pack(
                    side="left"
                )

    def _view_board_with_menu(
        self, overlay: ctk.CTkFrame, result: str, winner: Optional[str] = None
    ):
        """Hide dialog but keep overlay with floating menu button"""
        self.game_over_menu = None
        overlay.destroy()
        self._show_footer_signature()

        menu_button = ctk.CTkButton(
            self.root,
            text="⚙ Menu",
            font=("Arial", 14),
            width=100,
            height=40,
            command=lambda: self._show_game_over_dialog(result, winner),
        )
        menu_button.place(relx=0.88, rely=0.92, anchor="center")
        self._game_over_menu_btn = menu_button

    def _setup_pgn_metadata(
        self, event: str, white: str, black: str, white_elo: str, black_elo: str
    ) -> None:
        """set up PGN metadata for the game"""
        from datetime import datetime

        database.pgn_event = event
        database.pgn_date = datetime.now().strftime("%Y.%m.%d")
        database.pgn_round = "-"
        database.pgn_white = white
        database.pgn_black = black
        database.pgn_result = "*"  # Will be updated when game ends
        database.pgn_white_elo = white_elo
        database.pgn_black_elo = black_elo
        database.pgn_time_control = "-"
        database.pgn_termination = "-"  # Will be updated when game ends

    def _update_pgn_result(self, result: str, termination: str) -> None:
        """Update PGN result and termination when game ends"""
        database.pgn_result = result
        database.pgn_termination = termination

    def _copy_pgn_to_clipboard(self) -> None:
        """Copy the current game's PGN to clipboard"""
        try:
            pgn = self.utils.get_full_pgn()
            self.root.clipboard_clear()
            self.root.clipboard_append(pgn)
            self.root.update()  # Required for clipboard to work
            messagebox.showinfo("PGN Copied", "Game PGN has been copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy PGN: {e}")

    def _save_pgn_to_file(self) -> None:
        """Save the current game's PGN to a file with safe filename handling"""
        import os
        from pathlib import Path

        # Ask user for filename using simpledialog
        from tkinter import simpledialog

        filename = simpledialog.askstring(
            "Save PGN", "Enter filename for PGN (default: PGN):", parent=self.root
        )

        # Use default if user cancels or provides empty input
        if filename is None:  # User clicked Cancel
            return
        if filename == "":  # User clicked OK without entering anything
            filename = "PGN"

        # Remove any .pgn extension if user included it
        if filename.lower().endswith(".pgn"):
            filename = filename[:-4]

        # Create PGNs directory if it doesn't exist
        pgn_dir = Path("PGNs")
        pgn_dir.mkdir(exist_ok=True)

        # Find a unique filename by checking for existing files
        base_filename = filename
        final_filename = filename
        counter = 1
        pgn_path = pgn_dir / f"{final_filename}.pgn"

        while pgn_path.exists():
            final_filename = f"{base_filename}({counter})"
            pgn_path = pgn_dir / f"{final_filename}.pgn"
            counter += 1

        # Save the PGN
        try:
            pgn_content = self.utils.get_full_pgn()
            with open(pgn_path, "w", encoding="utf-8") as f:
                f.write(pgn_content)
            messagebox.showinfo(
                "PGN Saved", f"Game PGN saved as 'PGNs/{final_filename}.pgn'!"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PGN: {e}")

    def _start_new_game(self, overlay: ctk.CTkFrame) -> None:
            """Start a new game by resetting state"""
            self.game_over_menu = None
            overlay.destroy()
            self._cancel_pending_ai_work()

            if hasattr(self, "_game_over_menu_btn"):
                self._game_over_menu_btn.destroy()
                del self._game_over_menu_btn

            database.reset()
            self.review.reset_async_state()
            self.utils = Utilities()

            self.promoting = False
            self.promoting_square = None
            self.piece_selected = None
            self.selected_square = None
            self._legal_moves_update_scheduled = False
            self.flipped = False
            self.settings_open = False
            self.game_mode = None
            self.last_from_square = None
            self.last_to_square = None

            # --- THE FIX ---
            # Explicitly unfreeze the board colors before returning to the start menu
            self.disabled_color = [0, 0]

            self.view_matrix = database.matrix
            self._current_legal_visual_squares.clear()
            self._highlighted_visual_squares.clear()
            self._render_state.clear()
            self._last_image_size = 0

            # It's safer to explicitly set this to 0 for a brand new board
            self.current_fen = 0

            self._refresh_board_colors()

            for label in self.promotion_labels.values():
                label.configure(image=None)

            self._refresh_all_images()
            self.utils.legal_moves.update_legal_moves(database.matrix)

            self._show_start_menu()
            database.gamelogger.game("New game started!")

    def _rematch_game(self, overlay: ctk.CTkFrame) -> None:
            """Rematch the last game by resetting state"""
            self.game_over_menu = None
            overlay.destroy()
            self._cancel_pending_ai_work()

            if hasattr(self, "_game_over_menu_btn"):
                self._game_over_menu_btn.destroy()
                del self._game_over_menu_btn

            database.reset()
            self.review.reset_async_state()
            self.utils = Utilities()

            self.promoting = False
            self.promoting_square = None
            self.piece_selected = None
            self.selected_square = None
            self._legal_moves_update_scheduled = False
            self.flipped = False if self.game_mode != "vs_ai" else self.flipped
            self.settings_open = False
            self.last_from_square = None
            self.last_to_square = None
            self.view_matrix = database.matrix
            self._current_legal_visual_squares.clear()
            self._highlighted_visual_squares.clear()
            self._render_state.clear()
            self._last_image_size = 0
            self.current_fen = 0

            self._refresh_board_colors()

            for label in self.promotion_labels.values():
                label.configure(image=None)

            self._refresh_all_images()
            self.utils.legal_moves.update_legal_moves(database.matrix)

            # Route the specific game mode logic properly
            if self.game_mode == "vs_ai":
                # Standard chess rules: Swap colors on rematch
                if self.vs_ai_configurations and "color" in self.vs_ai_configurations:
                    current_color = self.vs_ai_configurations["color"]
                    if current_color in ["white", "black"]: # Ignore if they chose 'random'
                        self.vs_ai_configurations["color"] = "black" if current_color == "white" else "white"

                self._start_ai_game(None)

            elif self.game_mode == "pass_n_play":
                self.disabled_color = [0, 0]
                self._setup_pgn_metadata("Pass and Play", "Player 1", "Player 2", "-", "-")
                self._show_footer_signature()

    def _execute_move(
        self,
        from_square: tuple[int, int],
        to_square: tuple[int, int],
        base_piece: Optional[str] = None,
    ) -> bool:
        """Execute a piece move on the board - delegates to centralized state manager"""
        # Snap view_matrix back to live board if viewing history
        if self.flipped:
            self.view_matrix = np.flip(database.matrix, axis=(0, 1))
        else:
            self.view_matrix = database.matrix
        self.current_fen = len(database.fen_history) - 1

        piece = database.matrix[from_square[0], from_square[1]]

        if piece == 0 or not isinstance(piece, str):
            database.gamelogger.error("Error: No piece at source square")
            self.piece_selected = None
            self.selected_square = None
            self._refresh_all_images()
            return False

        if not self._is_legal_move(piece, to_square):
            return False

        legal_mvs = database.get_legal_moves(database.current_turn)
        if len(legal_mvs) == 1 and len(legal_mvs.get(piece, [])) == 1:
            database.last_forced = True
        else:
            database.last_forced = False

        # Check if this is a pawn promotion that requires user input
        if "p" in piece and to_square[0] in [0, 7]:
            promotion_color = "black" if "-" in piece else "white"

            self.promoting = True
            self.promoting_square = to_square
            self.promotion_from_square = from_square

            if self.game_mode == "vs_ai" and self.vs_ai_configurations and database.current_turn != self.vs_ai_configurations["color"]:
                if base_piece is not None:
                    database.matrix[from_square[0], from_square[1]] = 0
                    database.current_turn = "black" if database.current_turn == "white" else "white"
                    if database.current_turn == "white":
                        database.fullmove += 1
                    if "-" in piece:
                        database.black_last_pawn = None
                        database.black_pieces = database.black_pieces[database.black_pieces != piece]
                    else:
                        database.white_last_pawn = None
                        database.white_pieces = database.white_pieces[database.white_pieces != piece]

                    self._end_promotion(base_piece)
                    return True

            self._start_promotion(promotion_color, to_square, from_square, base_piece)
            return True

        # For all other moves: use the centralized state manager
        result = database.state_manager.execute_move(from_square, to_square)

        if result is None:
            database.gamelogger.error("Move execution failed")
            self.piece_selected = None
            self.selected_square = None
            self._refresh_all_images()
            return False

        # Generate notation based on move type
        chess_notation = self._generate_chess_notation(result)
        stockfish_notation = self.utils.notations.stockfish_notation(
            from_square, to_square
        )

        # UI updates
        self.piece_selected = None
        self.selected_square = None
        previous_legal_visuals = set(self._current_legal_visual_squares)
        previous_flipped = self.flipped

        if self.flip_allowed:
            self._flip_board(database.current_turn)
        orientation_changed = self.flipped != previous_flipped

        self._apply_move_highlights(from_square, to_square)

        if orientation_changed:
            self._current_legal_visual_squares = set()
            self._refresh_all_images(force=True)
        else:
            current_size = self._calculate_image_size()
            size_changed = current_size != self._last_image_size
            if current_size not in self._warmed_image_sizes:
                self._warm_image_cache_for_size(current_size)

            changed_logical_squares: set[tuple[int, int]] = {from_square, to_square}

            if result.is_en_passant:
                changed_logical_squares.add((from_square[0], to_square[1]))

            if result.is_castling:
                row = to_square[0]
                if to_square[1] == 6:
                    changed_logical_squares.update({(row, 7), (row, 5)})
                else:
                    changed_logical_squares.update({(row, 0), (row, 3)})

            changed_visual_squares = {
                self._logical_to_visual(square) for square in changed_logical_squares
            }
            changed_visual_squares.update(previous_legal_visuals)

            self._current_legal_visual_squares = set()
            if size_changed:
                self._refresh_squares(
                    set(self._all_visual_squares), set(), current_size, force=True
                )
            else:
                self._refresh_squares(changed_visual_squares, set(), current_size)

            self._update_highlights_incremental()

        # Store move info for notation generation after legal moves update
        self._pending_move = (
            from_square,
            to_square,
            piece,
            result.is_capture,
            chess_notation,
            stockfish_notation,
        )

        if not self._legal_moves_update_scheduled:
            self._legal_moves_update_scheduled = True
            self.root.after_idle(self._delayed_legal_moves_update)

        return True

    def _generate_chess_notation(self, result: "MoveResult") -> str:
        """Generate chess notation for a move"""
        piece = result.piece
        to_square = result.to_square
        is_capture = result.is_capture
        from_square = result.from_square
        disambiguation = result.disambiguation if result.disambiguation else ""

        if result.is_castling:
            return self.utils.notations.chess_notation(
                piece, to_square, castle=True, disambiguation=disambiguation
            )
        elif result.is_en_passant:
            return self.utils.notations.chess_notation(
                piece,
                to_square,
                capture=True,
                from_square=from_square,
                disambiguation=disambiguation,
            )
        else:
            return self.utils.notations.chess_notation(
                piece,
                to_square,
                capture=is_capture,
                from_square=from_square,
                disambiguation=disambiguation,
            )

    def _apply_move_highlights(
        self, from_square: tuple[int, int], to_square: tuple[int, int]
    ) -> None:
        """Store logical coordinates of the last move. Visuals are handled by _reapply_highlights."""
        self.last_from_square = from_square
        self.last_to_square = to_square

    def _delayed_legal_moves_update(self) -> None:
        self.utils.legal_moves.update_legal_moves(database.matrix)
        self._legal_moves_update_scheduled = False

        if hasattr(self, "_pending_move") and self._pending_move:
            (from_square, to_square, piece, is_capture,
             chess_notation, stockfish_notation) = self._pending_move
            self._pending_move = None

            game_over = self._check_game_over()  # opens dialog internally — fix below

            if game_over == "checkmate":
                chess_notation = self.utils.notations.chess_notation(
                    piece, to_square, capture=is_capture,
                    from_square=from_square, checkmate=True,
                )
            elif self.utils.legal_moves.check_checker(database.current_turn):
                chess_notation = self.utils.notations.chess_notation(
                    piece, to_square, capture=is_capture,
                    from_square=from_square, check=True,
                )

            database.game_history.append(stockfish_notation)
            database.game_pgn.append(chess_notation)
            database.fen_history.append(
                self.utils.notations.generate_fen(
                    side_to_move=database.current_turn[0],
                    fullmove_number=database.fullmove,
                )
            )
            self.current_fen += 1
            self.review.evaluate_last_move_async()
            database.gamelogger.move(f"Move: {chess_notation}")
            self.utils.ai.update_last_move()

            if game_over in ("checkmate", "stalemate"):
                winner = None
                if game_over == "checkmate":
                    winner = "black" if database.current_turn == "white" else "white"
                self._show_game_over_dialog(game_over, winner)
                return

            if self.game_mode == "vs_ai" and self.vs_ai_configurations:
                ai_color = (
                    "black" if self.vs_ai_configurations["color"] == "white" else "white"
                )
                if database.current_turn == ai_color:
                    self._execute_stockfish_move()

    def _flip_board(self, color: str) -> None:
        """Flip the board view"""
        if color == "black":
            self.view_matrix = np.flip(np.flip(database.matrix, 0), 1)
            self.flipped = True
        else:
            self.view_matrix = database.matrix
            self.flipped = False

    def _check_game_over(self) -> Optional[str]:
        legal_moves_dict = (
            database.white_legal_moves
            if database.current_turn == "white"
            else database.black_legal_moves
        )
        total_moves = sum(len(moves) for moves in legal_moves_dict.values())

        if total_moves == 0:
            if self.utils.legal_moves.check_checker(database.current_turn):
                winner = "black" if database.current_turn == "white" else "white"
                result = "1-0" if winner == "white" else "0-1"
                termination = f"{winner.capitalize()} checkmated {('black' if winner == 'white' else 'white').capitalize()}"
                self._update_pgn_result(result, termination)
                return "checkmate"
            else:
                self._update_pgn_result("1/2-1/2", "Stalemate")
                return "stalemate"

    def _disable_color(self, color: Literal["white", "black"], pos: int = 0):
        if not self.game_mode == "pass_n_play":
            self.disabled_color[pos] = color

    # ==================== setTINGS OVERLAY ====================

    def _show_settings_overlay(self) -> None:
        """Show settings overlay with configuration options"""
        if self.settings_open:
            return

        self.settings_open = True
        self._hide_footer_signature()

        overlay = ctk.CTkFrame(
            self.root, fg_color=("gray50", "gray20"), bg_color="transparent"
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._lift_persistent_widgets()
        self.root.bind("<Escape>", lambda _: self._close_settings_overlay(overlay))

        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()

        dialog_width = max(350, min(int(screen_width * 0.45), 550))
        dialog_height = max(300, min(int(screen_height * 0.5), 500))

        title_font_size = max(20, min(int(screen_height * 0.04), 28))
        button_font_size = max(12, min(int(screen_height * 0.02), 16))

        dialog = ctk.CTkFrame(
            overlay,
            fg_color=("white", "gray10"),
            corner_radius=20,
            border_width=2,
            border_color=("blue", "blue"),
            width=dialog_width,
            height=dialog_height,
        )
        dialog.place(relx=0.5, rely=0.5, anchor="center")

        title_label = ctk.CTkLabel(
            dialog, text="⚙ settings", font=("Arial Bold", title_font_size)
        )
        title_label.place(relx=0.5, rely=0.05, anchor="n", relwidth=0.9)

        content_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        content_frame.place(relx=0.05, rely=0.15, relwidth=0.9, relheight=0.65)

        def flip_allowed_controller() -> None:
            self._toggle_flip_allowed()
            flip_toggle.select() if self.flip_allowed else flip_toggle.deselect()

        flip_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        flip_frame.place(relx=0.05, rely=0.15, relwidth=0.9, relheight=0.12)

        flip_label = ctk.CTkLabel(
            flip_frame,
            text="Flip Board",
            font=("Arial", 14),
            wraplength=dialog_width - 100,
        )
        flip_label.place(relx=0, rely=0.5, anchor="w", relwidth=0.6)

        flip_toggle = ctk.CTkSwitch(
            flip_frame,
            text="",
            command=flip_allowed_controller,
            state="disabled" if self.game_mode != "pass_n_play" else "normal",
        )
        flip_toggle.place(relx=0.95, rely=0.5, anchor="e")
        flip_toggle.select() if self.flip_allowed else flip_toggle.deselect()

        cursor_style = "hand2" if self.game_mode == "pass_n_play" else "no"
        self._bind_cursor(flip_toggle, cursor_style)

        colors_label = ctk.CTkLabel(
            dialog, text="Board Colors", font=("Arial", 14, "bold")
        )
        colors_label.place(relx=0.05, rely=0.3, relwidth=0.9, relheight=0.08)

        white_label = ctk.CTkLabel(dialog, text="Light Squares:", font=("Arial", 12))
        white_label.place(relx=0.05, rely=0.4, relwidth=0.5, relheight=0.08)

        white_color_combo = ctk.CTkComboBox(
            dialog,
            values=["white", "beige", "ivory", "lightgray", "snow", "wheat"],
            state="readonly",
            width=120,
        )
        white_color_combo.set(self.style.white_box_color)
        white_color_combo.place(relx=0.55, rely=0.4, relwidth=0.4, relheight=0.08)
        white_color_combo.configure(
            command=lambda choice: self._change_white_color(choice)
        )

        black_label = ctk.CTkLabel(dialog, text="Dark Squares:", font=("Arial", 12))
        black_label.place(relx=0.05, rely=0.52, relwidth=0.5, relheight=0.08)

        black_color_combo = ctk.CTkComboBox(
            dialog,
            values=["brown", "darkbrown", "darkgray", "saddlebrown", "black", "sienna"],
            state="readonly",
            width=120,
        )
        black_color_combo.set(self.style.black_box_color)
        black_color_combo.place(relx=0.55, rely=0.52, relwidth=0.4, relheight=0.08)
        black_color_combo.configure(
            command=lambda choice: self._change_black_color(choice)
        )

        back_btn = ctk.CTkButton(
            dialog,
            text="Back",
            font=("Arial", button_font_size),
            width=120,
            height=40,
            fg_color="transparent",
            border_width=2,
            command=lambda: self._close_settings_overlay(overlay),
        )
        back_btn.place(
            relx=0.5, rely=0.8, anchor="center", relwidth=0.4, relheight=0.12
        )

    def _toggle_flip_board(self) -> None:
        self._flip_board("black" if not self.flipped else "white")
        self._refresh_all_images()

    def _toggle_flip_allowed(self) -> None:
        self.flip_allowed = not self.flip_allowed
        if not self.flip_allowed:
            self._flip_board("white")
        else:
            self._flip_board(database.current_turn)
        self._refresh_all_images()

    def _change_white_color(self, color: str) -> None:
        """Change white square color"""
        self.style.white_box_color = color
        self._refresh_board_colors()

    def _change_black_color(self, color: str) -> None:
        """Change black square color"""
        self.style.black_box_color = color
        self._refresh_board_colors()

    def _refresh_board_colors(self) -> None:
        """Refresh board colors after change"""
        for square in self._all_visual_squares:
            self._set_square_color(square, self._square_base_color(square))
        self._update_highlights_incremental(force=True)

    def _close_settings_overlay(self, overlay: ctk.CTkFrame) -> None:
        """Close settings overlay"""
        overlay.destroy()
        self.settings_open = False
        if self.game_mode is not None:
            self._show_footer_signature()
        self.root.bind("<Escape>", lambda _: self._handle_escape())

    def _handle_square_click(self, visual_square: tuple[int, int]) -> None:
        if (
            self.promoting
            or self.disabled_color is None
            or database.current_turn in self.disabled_color
        ):
            return
        # Block clicks while viewing history
        if self.current_fen != len(database.fen_history) - 1:
            return

        logical_square = self._visual_to_logical(visual_square)
        piece = database.matrix[logical_square[0], logical_square[1]]
        piece_color = (
            "white"
            if isinstance(piece, str) and "-" not in piece
            else "black"
            if piece != 0
            else None
        )
        previous_legal_visuals = set(self._current_legal_visual_squares)

        if (
            piece != 0
            and isinstance(piece, str)
            and piece_color == database.current_turn
        ):
            self.piece_selected = piece
            self.selected_square = logical_square

            new_legal_visuals = self._compute_legal_visual_squares(self.piece_selected)
            current_size = self._calculate_image_size()
            size_changed = current_size != self._last_image_size
            if current_size not in self._warmed_image_sizes:
                self._warm_image_cache_for_size(current_size)

            self._current_legal_visual_squares = new_legal_visuals
            if size_changed or not self._render_state:
                self._refresh_squares(
                    set(self._all_visual_squares),
                    new_legal_visuals,
                    current_size,
                    force=True,
                )
            else:
                changed_visuals = previous_legal_visuals.symmetric_difference(
                    new_legal_visuals
                )
                self._refresh_squares(changed_visuals, new_legal_visuals, current_size)

            self._update_highlights_incremental()
            return
        else:
            if self.piece_selected and isinstance(self.piece_selected, str):
                try:
                    from_square = (
                        self.selected_square
                        if self.selected_square is not None
                        else self.utils.legal_moves.search_piece(
                            self.piece_selected, database.matrix
                        )
                    )
                    moved = self._execute_move(from_square, logical_square)
                    if moved:
                        return
                except ValueError as e:
                    database.gamelogger.error(f"Piece not found: {e}")
                    self.piece_selected = None
                    self.selected_square = None

            self.piece_selected = None
            self.selected_square = None
            self._current_legal_visual_squares = set()

            current_size = self._calculate_image_size()
            size_changed = current_size != self._last_image_size
            if current_size not in self._warmed_image_sizes:
                self._warm_image_cache_for_size(current_size)

            if size_changed or not self._render_state:
                self._refresh_squares(
                    set(self._all_visual_squares), set(), current_size, force=True
                )
            else:
                self._refresh_squares(previous_legal_visuals, set(), current_size)

            self._update_highlights_incremental()

    def _cancel_pending_ai_work(self, shutdown_executor: bool = False) -> None:
        """Invalidate pending AI tasks and optionally shutdown executor."""
        self._ai_request_token += 1
        if self._ai_future and not self._ai_future.done():
            self._ai_future.cancel()
        self._ai_future = None

        if shutdown_executor:
            try:
                self._ai_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # Recreate executor so the next game can still submit AI tasks
            self._ai_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chess-ai"
            )

    def _get_ai_color(self) -> Optional[str]:
        if not self.vs_ai_configurations:
            return None
        return "black" if self.vs_ai_configurations["color"] == "white" else "white"

    def _poll_ai_move_result(self, request_token: int) -> None:
        if request_token != self._ai_request_token or self._ai_future is None:
            return

        if not self._ai_future.done():
            self.root.after(
                10, lambda token=request_token: self._poll_ai_move_result(token)
            )
            return

        future = self._ai_future
        self._ai_future = None

        try:
            move = future.result()
        except Exception as e:
            database.gamelogger.error(f"AI move fetch failed: {e}")
            move = None

        if request_token != self._ai_request_token:
            return

        if not move:
            database.gamelogger.ai("Stockfish returned no move!")
            return

        database.gamelogger.move(f"Stockfish plays: {move}")
        elo = self.utils.ai.current_config.elo if self.utils.ai.current_config else 0
        delay = 50 if elo > 2000 else random.randint(250, 500)
        self.root.after(
            delay, lambda mv=move, token=request_token: self._apply_ai_move(mv, token)
        )

    def _apply_ai_move(self, move: str, request_token: int) -> None:
        if request_token != self._ai_request_token:
            return

        ai_color = self._get_ai_color()
        if ai_color is None or database.current_turn != ai_color:
            return

        from_sq = move[:2]
        to_sq = move[2:4]
        base = move[-1] if len(move) == 5 else None

        from_square = self.utils.coords.chess_to_matrix(from_sq)
        to_square = self.utils.coords.chess_to_matrix(to_sq)
        promotion = base.lower() if base else None

        # Delay already applied in _poll_ai_move_result; execute immediately here
        self._execute_move(from_square, to_square, promotion)

    def _execute_stockfish_move(self) -> None:
        """Queue asynchronous Stockfish move fetch without blocking UI thread."""
        ai_color = self._get_ai_color()
        if ai_color is None or database.current_turn != ai_color:
            return

        if self._ai_future and not self._ai_future.done():
            return

        self._ai_request_token += 1
        request_token = self._ai_request_token
        self._ai_future = self._ai_executor.submit(self.utils.ai.get_ai_move)
        self._poll_ai_move_result(request_token)


if __name__ == "__main__":
    ChessGame()
