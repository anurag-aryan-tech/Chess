import tkinter as tk
import customtkinter as ctk
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict

class Database:
    def __init__(self):
        self.matrix_path = Path("database/matrix.json")
        self.current_turn = "white"

        self.r1_moved = False
        self.r2_moved = False
        self.k1_moved = False

        self.r1_black_moved = False
        self.r2_black_moved = False
        self.k1_black_moved = False

        self.black_last_pawn: None|Tuple[int, int] = None
        self.white_last_pawn: None|Tuple[int, int] = None

        self.white_pieces = np.array([
            [f"p{i}" for i in range(1, 9)],
            ['r1', 'n1', 'b1', 'q1', 'k1', 'b2', 'n2', 'r2'],
        ], dtype=object)

        self.black_pieces = np.array([
            ['-r1', '-n1', '-b1', '-q1', '-k1', '-b2', '-n2', '-r2'],
            [f"-p{i}" for i in range(1, 9)]
        ], dtype=object)

        self.matrix = np.zeros((8, 8), dtype=object)
        self.white_legal_moves: Dict[str, np.ndarray] = {}
        self.black_legal_moves: Dict[str, np.ndarray] = {}

        self.pins: Dict[str, Tuple[int, int]] = {}

        self.initialize_matrix()
        self.save_matrix()

        self.initialize_legal_moves()

    def initialize_matrix(self) -> None:
        self.matrix[:2, :] = self.black_pieces
        self.matrix[6:, :] = self.white_pieces

        print("Board Matrix initialized!")

    def save_matrix(self) -> None:
        matrix = json.dumps({'matrix' : self.matrix.tolist()})
        self.matrix_path.write_text(matrix)

        print("Game saved!\n")

    def import_matrix(self):
        matrix = json.loads(self.matrix_path.read_text())
        self.matrix = np.array(matrix['matrix'])

    def initialize_legal_moves(self):
        self.white_legal_moves = {piece: np.array([]) for piece in self.white_pieces.flatten()}
        self.black_legal_moves = {piece: np.array([]) for piece in self.black_pieces.flatten()}


database = Database()
