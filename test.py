"""
Chess Game Result Screen — CustomTkinter (Pure Widgets, No Canvas)
All layout done with: CTkFrame, CTkButton, CTkLabel, CTkFont
Shadow trick: dark CTkFrame placed behind each button/card via .place(), offset by SHADOW px
Circular icons: CTkFrame with corner_radius = half its size (acts as circle)
"""

import customtkinter as ctk
import platform

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Palette ───────────────────────────────────────────────────────────────────
BG_OUTER      = "#2b2b2b"
CARD_BG       = "#3d3d3d"
HEADER_BG     = "#2e2e2e"
STAT_BOX_BG   = "#4a4a4a"
SEPARATOR_CLR = "#555555"
BTN_GREEN     = "#5faa3a"
BTN_GREEN_HOV = "#4d8e2f"
BTN_DARK      = "#4a4a4a"
BTN_DARK_HOV  = "#606060"
SHADOW_CLR    = "#1a1a1a"

WHITE         = "#ffffff"
GRAY          = "#888888"
BLUE_ACCENT   = "#5b9bd5"
YELLOW_ACCENT = "#d4a017"
GREEN_ACCENT  = "#5faa3a"
ICON_BLUE     = "#3a6da8"
ICON_YELLOW   = "#c89a10"
ICON_GREEN    = "#4a8a2a"

# ── Sizing ────────────────────────────────────────────────────────────────────
WIN_W         = 860
WIN_H         = 580
CARD_W        = 600
CARD_H        = 515
CARD_RADIUS   = 14
SHADOW        = 5     # shadow offset in pixels

IS_WIN        = platform.system() == "Windows"
EMOJI_FAMILY  = "Segoe UI Emoji" if IS_WIN else "TkDefaultFont"


# ── Reusable font builders ────────────────────────────────────────────────────
def font(size, weight="normal", family="Helvetica"):
    return ctk.CTkFont(family=family, size=size, weight=weight)

def emoji_font(size):
    return ctk.CTkFont(family=EMOJI_FAMILY, size=size)


# ── Shadowed button ───────────────────────────────────────────────────────────
def shadowed_button(parent, text, fg_color, hover_color,
                    width, height, font_size=13, command=None):
    """
    Returns a transparent CTkFrame sized (width+SHADOW) × (height+SHADOW).
    Inside it, a dark CTkFrame is placed at (+SHADOW, +SHADOW) as the shadow,
    then a CTkButton is placed at (0, 0) — covering the top-left of the shadow
    and leaving only the bottom-right SHADOW pixels visible.
    """
    wrapper = ctk.CTkFrame(parent, fg_color="transparent",
                           width=width + SHADOW, height=height + SHADOW)
    wrapper.pack_propagate(False)
    wrapper.grid_propagate(False)

    # 1. Shadow layer — drawn first, visually behind button
    ctk.CTkFrame(
        wrapper, width=width, height=height,
        corner_radius=10, fg_color=SHADOW_CLR,
	bg_color="transparent"
    ).place(x=SHADOW, y=SHADOW)

    # 2. Button face — placed at origin, sits on top of shadow
    ctk.CTkButton(
        wrapper, text=text,
        width=width, height=height,
        corner_radius=10,
        fg_color=fg_color,
        hover_color=hover_color,
        text_color=WHITE,
        font=font(font_size, "bold"),
        command=command,
	bg_color="transparent"
    ).place(x=0, y=0)

    return wrapper


# ── Stat box ──────────────────────────────────────────────────────────────────
def stat_box(parent, symbol, icon_bg, count, label, label_color, sym_font=None):
    """
    Rounded CTkFrame containing:
      - circular icon  (CTkFrame with corner_radius = half its size)
      - count number   (CTkLabel)
      - category name  (CTkLabel)
    """
    box = ctk.CTkFrame(parent, fg_color=STAT_BOX_BG, corner_radius=10,
                       width=172, height=108)
    box.pack_propagate(False)

    # Circular icon — corner_radius = 21 on a 42×42 frame → perfect circle
    circle = ctk.CTkFrame(box, width=42, height=42,
                          corner_radius=21, fg_color=icon_bg, bg_color="transparent")
    circle.pack(pady=(12, 2))
    circle.pack_propagate(False)

    ctk.CTkLabel(
        circle, text=symbol,
        font=sym_font or font(15, "bold"),
        text_color=WHITE, fg_color="transparent",
	bg_color="transparent"
    ).place(relx=0.5, rely=0.5, anchor="center")

    # Count
    ctk.CTkLabel(
        box, text=str(count),
        font=font(19, "bold"),
        text_color=WHITE, fg_color="transparent",
	bg_color="transparent"
    ).pack()

    # Label
    ctk.CTkLabel(
        box, text=label,
        font=font(11, "bold"),
        text_color=label_color, fg_color="transparent",
	bg_color="transparent"
    ).pack(pady=(0, 8))

    return box


# ── Main App ──────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Game Result")
        self.resizable(False, False)
        self.configure(fg_color=BG_OUTER)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(
            f"{WIN_W}x{WIN_H}+{(sw - WIN_W) // 2}+{(sh - WIN_H) // 2}"
        )

        # Full-window outer container
        outer = ctk.CTkFrame(self, fg_color=BG_OUTER, corner_radius=0, bg_color="transparent")
        outer.pack(fill="both", expand=True)

        # ── Card shadow ───────────────────────────────────────────────────────
        # Same size as card, placed centre + SHADOW offset → peeks out bottom-right
        ctk.CTkFrame(
            outer, width=CARD_W, height=CARD_H,
            corner_radius=CARD_RADIUS, fg_color=SHADOW_CLR,
	    bg_color="transparent"
        ).place(relx=0.5, rely=0.5, anchor="center", x=SHADOW, y=SHADOW)

        # ── Card ──────────────────────────────────────────────────────────────
        card = ctk.CTkFrame(
            outer, width=CARD_W, height=CARD_H,
            corner_radius=CARD_RADIUS, fg_color=CARD_BG,
	    bg_color="transparent"
        )
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)

        self._build(card)

    # ── Card contents ─────────────────────────────────────────────────────────
    def _build(self, card):

        # ── HEADER ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(card, fg_color=HEADER_BG,
                              corner_radius=CARD_RADIUS)
        header.pack(fill="x")

        # Crown 👑  +  "Black Wins!"  +  👑  — all in one row
        title_row = ctk.CTkFrame(header, fg_color="transparent", bg_color="transparent")
        title_row.pack(pady=(20, 6))

        ctk.CTkLabel(title_row, text="👑",
                     font=emoji_font(22),
                     text_color=WHITE, fg_color="transparent",
		     bg_color="transparent"
                     ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(title_row, text="Black Wins!",
                     font=font(27, "bold"),
                     text_color=WHITE, fg_color="transparent",
		     bg_color="transparent"
                     ).pack(side="left")

        ctk.CTkLabel(title_row, text="👑",
                     font=emoji_font(22),
                     text_color=WHITE, fg_color="transparent",
		     bg_color="transparent"
                     ).pack(side="left", padx=(10, 0))

        # "by Resignation"
        ctk.CTkLabel(header, text="by Resignation",
                     font=font(14),
                     text_color=GRAY, fg_color="transparent",
		     bg_color="transparent"
                     ).pack(pady=(0, 18))

        # ── SEPARATOR ─────────────────────────────────────────────────────────
        ctk.CTkFrame(card, height=2, fg_color=SEPARATOR_CLR,
                     corner_radius=0, bg_color="transparent").pack(fill="x")

        # ── STAT BOXES ────────────────────────────────────────────────────────
        stats_row = ctk.CTkFrame(card, fg_color="transparent", bg_color="transparent")
        stats_row.pack(padx=20, pady=20, fill="x")
        stats_row.columnconfigure((0, 1, 2), weight=1)

        boxes = [
            # symbol  icon_bg      count  label        label_color   sym_font
            ("!",    ICON_BLUE,    1,     "Great",     BLUE_ACCENT,  font(17, "bold")),
            ("★",    ICON_YELLOW,  4,     "Best",      YELLOW_ACCENT,emoji_font(15)),
            ("👍",   ICON_GREEN,   2,     "Excellent", GREEN_ACCENT, emoji_font(15)),
        ]

        for col, (sym, ibg, cnt, lbl, lbl_clr, sfnt) in enumerate(boxes):
            b = stat_box(stats_row, sym, ibg, cnt, lbl, lbl_clr, sfnt)
            b.grid(row=0, column=col, padx=8, sticky="nsew")

        # ── BUTTONS ───────────────────────────────────────────────────────────
        btn_area = ctk.CTkFrame(card, fg_color="transparent", bg_color="transparent")
        btn_area.pack(fill="x", padx=22)

        BW = CARD_W - 44   # total usable button row width

        # Each shadowed_button wrapper is (face_w + SHADOW) wide.
        # Subtract SHADOW from face widths so wrappers + gaps == BW exactly.

        # -- Game Review: wrapper = (BW - SHADOW) + SHADOW = BW --
        shadowed_button(
            btn_area, "Game Review",
            BTN_GREEN, BTN_GREEN_HOV,
            width=BW - SHADOW, height=52, font_size=15
        ).pack(pady=(2, 12))

        # -- Rematch | New Game --
        # 2*(hw + SHADOW) + 10(gap) = BW  =>  hw = (BW - 10 - 2*SHADOW) // 2
        row2 = ctk.CTkFrame(btn_area, fg_color="transparent", bg_color="transparent")
        row2.pack(fill="x", pady=(0, 10))

        hw = (BW - 10 - 2 * SHADOW) // 2

        shadowed_button(row2, "Rematch",
                        BTN_DARK, BTN_DARK_HOV, hw, 46).pack(side="left")
        ctk.CTkFrame(row2, fg_color="transparent", bg_color="transparent",
                     width=10, height=1).pack(side="left")
        shadowed_button(row2, "New Game",
                        BTN_DARK, BTN_DARK_HOV, hw, 46).pack(side="left")

        # -- View Board | Copy PGN | Save PGN --
        # 3*(tw + SHADOW) + 20(gaps) = BW  =>  tw = (BW - 20 - 3*SHADOW) // 3
        row3 = ctk.CTkFrame(btn_area, fg_color="transparent", bg_color="transparent")
        row3.pack(fill="x")

        tw = (BW - 20 - 3 * SHADOW) // 3

        for i, lbl in enumerate(["View Board", "Copy PGN", "Save PGN"]):
            shadowed_button(row3, lbl,
                            BTN_DARK, BTN_DARK_HOV,
                            tw, 44, font_size=12).pack(side="left")
            if i < 2:
                ctk.CTkFrame(row3, fg_color="transparent", bg_color="transparent",
                             width=10, height=1).pack(side="left")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()