import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from utils import Utilities
from database.database import database

utils = Utilities()
@dataclass
class STYLE_CONFIG:
    white_box_color: str = "white"
    black_box_color: str = "brown"

    chessboard_rely: float = 0.1

    close_button_fg = "transparent"
    close_button_hvr = "red"

class ChessGame:
    def __init__(self) -> None:
        self.style = STYLE_CONFIG()
        self.prev_width = 0
        self.prev_height = 0
        self.size_threshold = 10
        self._resize_scheduled = False  # NEW: Prevent recursive triggers

        self.matrix = database.matrix

        self.root = self._initialize_window()
        self._setup_window(self.root)

        self._create_close_button()
        self._add_close_button()

        self.chessboard_frame = self._create_chessboard_frame()
        
        # Initialize chessboard position ONCE after window is ready
        self.root.update_idletasks()
        self._update_chessboard_position()

        self._make_matrix()
        self._create_chessboard_squares()
        
        self.root.mainloop()

    def _initialize_window(self, title: str="Chess Game") -> tk.Tk|ctk.CTk:
        root = ctk.CTk()
        root.title(title)
        root.bind("<Escape>", lambda event=None: self._root_binding(root))
        root.bind("<Configure>", lambda event=None: self._root_configure_bindings(root, event))
        root.protocol("WM_DELETE_WINDOW", self._close_button_command)
        root.protocol("")
        return root
    
    def _setup_window(self, window: tk.Tk|ctk.CTk) -> None:
        utils.fullscreen_window(window)
        window.minsize(600, 440)

    def _root_binding(self, window: tk.Tk|ctk.CTk) -> None:
        self._close_button_control(window)
        utils.fullscreen_toggle(window)

    def _root_configure_bindings(self, window: tk.Tk|ctk.CTk, event) -> None:
        self._close_button_control(window)
        
        # Only respond to root window resize events, not child widget events
        if event.widget == window and not self._resize_scheduled:
            self._resize_scheduled = True
            # Use after_idle to batch multiple rapid resize events
            window.after_idle(self._delayed_resize, event.width, event.height)

    def _delayed_resize(self, width: int, height: int) -> None:
        """Handle resize with debouncing"""
        self._resize_scheduled = False
        self._chessboard_frame_control(width, height)

    def _close_button_control(self, window: tk.Tk|ctk.CTk) -> None:
        if window.attributes("-fullscreen"):
            self._add_close_button()

        else:
            self._remove_close_button()

    def _create_close_button(self):
        self.close_button = ctk.CTkButton(self.root, text="x", command=self._close_button_command, fg_color=self.style.close_button_fg, hover_color=self.style.close_button_hvr, font=("Arial", 22))

    def _add_close_button(self):
        self.close_button.place(relx=0.95, rely=0, relwidth=0.05, relheight=0.05)

    def _remove_close_button(self):
        self.close_button.place_forget()

    def _close_button_command(self):
        answer = messagebox.askyesno("Warning", "Are you sure you want to close the game?")
        if answer:
            self.root.destroy()
            print("Game closed!")

    def _create_chessboard_frame(self):
        chessboard_frame = ctk.CTkFrame(self.root, fg_color="white")
        print("Chessboard frame created!")
        return chessboard_frame
    def _chessboard_frame_control(self, screen_width: int, screen_height: int) -> None:
        # Check threshold BEFORE updating stored values
        if (abs(screen_width - self.prev_width) < self.size_threshold and 
            abs(screen_height - self.prev_height) < self.size_threshold):
            return
        
        # Update stored dimensions
        self.prev_width = screen_width
        self.prev_height = screen_height
        
        self._update_chessboard_position()

    def _update_chessboard_position(self) -> None:
        """Separate method to actually position the frame"""
        screen_width = self.root.winfo_width()
        screen_height = self.root.winfo_height()
        
        rely = self.style.chessboard_rely
        relx = utils.relative_dimensions(rely, (screen_height, screen_width))
        
        # Just update placement - no need to forget first
        self.chessboard_frame.place(
            relx=relx, 
            rely=rely, 
            relwidth=1-relx*2, 
            relheight=1-rely*2
        )

    def _make_matrix(self):
        rows = 8
        columns = 8

        for i in range(rows):
            self.chessboard_frame.rowconfigure(i, weight=1)

        for j in range(columns):
            self.chessboard_frame.columnconfigure(j, weight=1)
    
    def _create_chessboard_squares(self) -> None:
        color = self.style.white_box_color
        color2 = self.style.black_box_color

        self.chessboard_squares = {}

        for i in range(8):
            for j in range(8):
                square = ctk.CTkFrame(self.chessboard_frame, fg_color=color, bg_color=color)
                square.grid(row=i, column=j, sticky="nsew")

                self.chessboard_squares[(i, j)] = square

                color, color2 = color2, color
            color, color2 = color2, color
        print("Chessboard squares created!\n")

ChessGame()