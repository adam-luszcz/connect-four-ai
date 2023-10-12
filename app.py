from easyAI import TwoPlayerGame
import numpy as np


"""
How to set up the game:
---
Please install easyAI with pip3 install easyAI
and run thegame with: python3 connect_four.py


How to play:
---
The goal of the game is for the user to get 4 of his checkers
in a row—horizontally, vertically, or diagonally
before the AI does it.
The user sets the checker with giving the column number 0-6 in his turn.
The checker is always set at the first available space counting from bottom.
Links to offline game instructions:
https://www.wikiboardgames.com/rules-of-connect-four/
https://www.virginialawngames.com/wp-content/uploads/2019/10/Connect-Four-Rule-Guide-How-To-Play-Connect-4.pdf
https://instructions.hasbro.com/en-my/instruction/connect-4-game

Authors: Adam Łuszcz, Anna Rogala
"""


class ConnectFour(TwoPlayerGame):
    def __init__(self, players):
        self.players = players
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def possible_moves(self):
        moves = []
        for i in range(7):
            for row in self.board:
                if row[i] == 0:
                    moves.append(i)
                    break
        return moves

    def make_move(self, column):
        for row in range(6):
            if self.board[row, column] == 0:
                self.board[row, column] = self.current_player
                break

    def show(self):
        for row in range(6):
            row_str = ' '.join([['.', '1', '2'][self.board[5 - row][col]] for col in range(7)])
            print(row_str)
        print("-" * 13)
        print("0 1 2 3 4 5 6")

    def lose(self):
        return find_four(self.board, self.opponent_index)

    def is_over(self):
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0


def find_four(board, opponent_player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for i in range(6):
        for j in range(7):
            for dr, dc in directions:
                for k in range(4):
                    r, c = i + k * dr, j + k * dc
                    if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == opponent_player:
                        if k == 3:
                            return True
                    else:
                        break

    return False


if __name__ == '__main__':
    from easyAI import Human_Player, AI_Player, Negamax

    ai = Negamax(6)
    game = ConnectFour([Human_Player(), AI_Player(ai)])
    game.play()
    if game.lose():
        print('-----------------')
        print(f'Player {game.opponent_index} wins the game.')
    else:
        print('-----------------')
        print('It is a draw.')
