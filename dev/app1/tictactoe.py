"""Simple console TicTacToe game"""

from typing import List

Board = List[List[str]]


def print_board(board: Board) -> None:
    for row in board:
        print(" | ".join(row))
        print("-" * 9)


def check_winner(board: Board) -> str | None:
    lines = board + list(zip(*board)) + [
        [board[i][i] for i in range(3)],
        [board[i][2 - i] for i in range(3)],
    ]
    for line in lines:
        if line[0] != " " and all(cell == line[0] for cell in line):
            return line[0]
    return None


def main() -> None:
    board: Board = [[" "] * 3 for _ in range(3)]
    player = "X"
    while True:
        print_board(board)
        move = input(f"Player {player}, enter row and column (1-3 1-3): ")
        try:
            r, c = map(int, move.split())
            r -= 1
            c -= 1
            if board[r][c] != " ":
                print("Square taken, try again.")
                continue
        except Exception:
            print("Invalid input, try again.")
            continue
        board[r][c] = player
        winner = check_winner(board)
        if winner:
            print_board(board)
            print(f"Player {winner} wins!")
            break
        if all(cell != " " for row in board for cell in row):
            print("It's a draw!")
            break
        player = "O" if player == "X" else "X"


if __name__ == "__main__":
    main()
