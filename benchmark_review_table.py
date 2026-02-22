"""
Calibration benchmark for review move-type matching across two games.

This script replays CP-only tables with:
- API-only book policy simulation (with opening break state)
- mover-perspective EP/CP loss conversion via ReviewSystem helpers
- hybrid move classification from review_system.py
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from review_system import (
    HYBRID_CP_EXCELLENT,
    HYBRID_EP_EXCELLENT,
    BoardAnalyzer,
    MoveClassifier,
    OpeningBook,
    ReviewSystem,
)


TARGET_COMBINED = 0.85
TARGET_PER_GAME = 0.75

TYPE_RANK = {
    "blunder": 0,
    "mistake": 1,
    "inaccuracy": 2,
    "miss": 3,
    "good": 4,
    "excellent": 5,
    "best": 6,
    "great": 7,
    "brilliant": 8,
    "book": 9,
}


@dataclass(frozen=True)
class MoveRow:
    move_no: int
    move: str
    color: str
    cp_after: int
    chess_com_type: str
    api_opening: bool


GAME1_ROWS: List[MoveRow] = [
    MoveRow(1, "e4", "white", -35, "book", True),
    MoveRow(2, "e5", "black", 39, "book", True),
    MoveRow(3, "Nf3", "white", -41, "book", True),
    MoveRow(4, "Nc6", "black", 42, "book", True),
    MoveRow(5, "Nc3", "white", -20, "book", True),
    MoveRow(6, "Nf6", "black", 18, "book", True),
    MoveRow(7, "d4", "white", -12, "book", True),
    MoveRow(8, "exd4", "black", 11, "book", True),
    MoveRow(9, "e5", "white", 195, "mistake", False),
    MoveRow(10, "dxc3", "black", -182, "best", False),
    MoveRow(11, "exf6", "white", 202, "best", False),
    MoveRow(12, "Qxf6", "black", -185, "best", False),
    MoveRow(13, "b3", "white", 255, "inaccuracy", False),
    MoveRow(14, "d5", "black", -227, "good", False),
    MoveRow(15, "Qe2+", "white", 282, "inaccuracy", False),
    MoveRow(16, "Be7", "black", -243, "good", False),
    MoveRow(17, "Bg5", "white", 248, "best", False),
    MoveRow(18, "Qd6", "black", -252, "excellent", False),
    MoveRow(19, "Bxe7", "white", 243, "best", False),
    MoveRow(20, "Nxe7", "black", -234, "excellent", False),
    MoveRow(21, "Qb5+", "white", 288, "good", False),
    MoveRow(22, "Bd7", "black", -249, "excellent", False),
    MoveRow(23, "Qxb7", "white", 318, "excellent", False),
    MoveRow(24, "Bc6", "black", -192, "mistake", False),
]

GAME2_ROWS: List[MoveRow] = [
    MoveRow(1, "d4", "white", -33, "book", True),
    MoveRow(2, "Nh6", "black", 116, "mistake", False),
    MoveRow(3, "e4", "white", -129, "best", False),
    MoveRow(4, "d5", "black", 140, "good", False),
    MoveRow(5, "exd5", "white", -144, "best", False),
    MoveRow(6, "Qxd5", "black", 156, "best", False),
    MoveRow(7, "Nc3", "white", -151, "best", False),
    MoveRow(8, "Qe6+", "black", 154, "inaccuracy", False),
    MoveRow(9, "Be2", "white", -161, "best", False),
    MoveRow(10, "Nc6", "black", 459, "mistake", False),
    MoveRow(11, "d5", "white", -476, "great", False),
    MoveRow(12, "Qg6", "black", 485, "best", False),
    MoveRow(13, "dxc6", "white", -492, "best", False),
    MoveRow(14, "bxc6", "black", 500, "excellent", False),
    MoveRow(15, "Nf3", "white", -504, "best", False),
    MoveRow(16, "Qg2", "black", 505, "best", False),
    MoveRow(17, "Rg1", "white", -526, "best", False),
    MoveRow(18, "Qh3", "black", 514, "best", False),
    MoveRow(19, "Bf4", "white", -511, "best", False),
    MoveRow(20, "Qd7", "black", 655, "good", False),
    MoveRow(21, "Ne5", "white", -647, "best", False),
]

DATASETS: Dict[str, List[MoveRow]] = {
    "game1": GAME1_ROWS,
    "game2": GAME2_ROWS,
}


def replay_dataset(
    rows: List[MoveRow],
    review: ReviewSystem,
    classifier: MoveClassifier,
) -> Tuple[List[Tuple[int, str, str, str, bool, float, float]], int]:
    opening_active = True
    results: List[Tuple[int, str, str, str, bool, float, float]] = []
    opening_false_book = 0

    for idx, row in enumerate(rows):
        ep_loss = 0.0
        cp_loss = 0.0

        if row.move_no == 1:
            predicted = "book"
            opening_active = True
        elif opening_active and row.api_opening:
            predicted = "book"
        else:
            opening_active = False
            prev = rows[idx - 1]
            mover_cp_before = review._to_mover_cp(prev.cp_after, "", row.color, is_after_move=False)
            mover_cp_after = review._to_mover_cp(row.cp_after, "", row.color, is_after_move=True)
            ep_loss = review._calculate_mover_ep_loss(mover_cp_before, mover_cp_after)
            cp_loss = review._calculate_mover_cp_loss(mover_cp_before, mover_cp_after)
            predicted = classifier.classify_from_hybrid_loss(
                ep_loss=ep_loss,
                cp_loss=cp_loss,
                mover_cp_before=mover_cp_before,
                mover_cp_after=mover_cp_after,
                move_count=row.move_no,
                last_pgn=row.move
            )

        matched = predicted == row.chess_com_type
        if predicted == "book" and row.chess_com_type != "book":
            opening_false_book += 1

        results.append((row.move_no, row.move, predicted, row.chess_com_type, matched, ep_loss, cp_loss))

    return results, opening_false_book


def print_dataset_result(name: str, rows: List[MoveRow], results: List[Tuple[int, str, str, str, bool, float, float]]) -> None:
    print(f"\n{name.upper()}")
    print("Move | SAN    | Predicted   | Chess.com   | Match | EP Loss | CP Loss")
    print("-----+--------+-------------+-------------+-------+---------+--------")
    for move_no, move, predicted, target, matched, ep_loss, cp_loss in results:
        print(
            f"{move_no:>4} | {move:<6} | {predicted:<11} | {target:<11} | "
            f"{str(matched):<5} | {ep_loss:>7.4f} | {cp_loss:>6.1f}"
        )

    matches = sum(1 for row in results if row[4])
    total = len(results)
    print(f"Match: {matches}/{total} ({matches/total:.2%})")


def run_benchmark() -> int:
    review = ReviewSystem()
    classifier = MoveClassifier(BoardAnalyzer(), OpeningBook())

    all_confusion = Counter()
    game_rates: Dict[str, float] = {}
    opening_false_book_by_game: Dict[str, int] = {}
    detailed_results: Dict[str, List[Tuple[int, str, str, str, bool, float, float]]] = {}

    for name, rows in DATASETS.items():
        results, opening_false_book = replay_dataset(rows, review, classifier)
        detailed_results[name] = results
        opening_false_book_by_game[name] = opening_false_book
        print_dataset_result(name, rows, results)

        matches = sum(1 for row in results if row[4])
        rate = matches / len(results)
        game_rates[name] = rate

        for _, _, predicted, target, *_ in results:
            all_confusion[(predicted, target)] += 1

    macro_avg = sum(game_rates.values()) / len(game_rates)
    print("\nSUMMARY")
    for name, rate in game_rates.items():
        print(f"- {name}: {rate:.2%}")
    print(f"- macro average: {macro_avg:.2%}")
    print(f"- opening false-book count (game2): {opening_false_book_by_game['game2']}")

    print("\nCONFUSION (predicted -> chess.com)")
    for (predicted, target), count in sorted(all_confusion.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {predicted:>10} -> {target:<10}: {count}")

    # Acceptance checks from plan
    failures: List[str] = []
    if macro_avg < TARGET_COMBINED:
        failures.append(f"macro average {macro_avg:.2%} < {TARGET_COMBINED:.0%}")

    for name, rate in game_rates.items():
        if rate < TARGET_PER_GAME:
            failures.append(f"{name} match {rate:.2%} < {TARGET_PER_GAME:.0%}")

    if opening_false_book_by_game["game2"] != 0:
        failures.append("game2 opening false-book count is not zero")

    # Targeted checks from plan
    game2_map = {move_no: row for move_no, *row in detailed_results["game2"]}
    for move_no, san in [(2, "Nh6"), (3, "e4"), (4, "d5")]:
        predicted = game2_map[move_no][1]
        if predicted == "book":
            failures.append(f"game2 move {move_no} ({san}) predicted as book")

    for move_no, san in [(15, "Nf3"), (17, "Rg1"), (18, "Qh3"), (21, "Ne5")]:
        _, predicted, _, _, ep_loss, cp_loss = game2_map[move_no]
        crossed = (ep_loss >= HYBRID_EP_EXCELLENT) or (cp_loss >= HYBRID_CP_EXCELLENT)
        if (not crossed) and TYPE_RANK.get(predicted, -1) < TYPE_RANK["best"]:
            failures.append(f"game2 move {move_no} ({san}) worse than best without threshold crossing")

    review.shutdown()

    print("\nCHECKS")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
