import chess
import chess.svg
# from IPython.display import SVG, display
# import cairosvg
# from PIL import Image, ImageTk
# import io
# import tkinter as tk
from chessboard import display 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
from time import sleep

def get_board(move_uci_array, move_number):
    board = chess.Board()
    for i in range(move_number):
        move = chess.Move.from_uci(move_uci_array[i])
        board.push(move)

    return board


# df = pd.read_csv("kingbase_processed_all.csv")

board = chess.Board()

# all_moves = df["Moves_UCI"]
# arr1 = all_moves[0].split(" ")
arr1 = ['g1f3', 'c7c5', 'g2g3', 'g7g6', 'f1g2', 'f8g7', 'c2c4', 'b8c6', 'b1c3', 'e7e6', 'd2d3', 'g8e7', 'c1d2', 'e8g8', 'a2a3', 'b7b6', 'a1b1', 'c8b7', 'b2b4', 'c5b4', 'a3b4', 'c6e5', 'f3h4', 'b7g2', 'h4g2', 'd7d5', 'f2f4', 'e5c6', 'b4b5', 'c6a5', 'c4d5', 'e7d5', 'c3d5', 'd8d5', 'e1g1', 'a5b7', 'd1b3', 'd5d7', 'f1c1', 'f8c8', 'b3a4', 'b7c5', 'a4a3', 'g7f8', 'a3a1', 'c5e4', 'd2e3', 'f8g7', 'c1c8', 'a8c8', 'a1a6', 'e4c3', 'b1e1', 'c3b5', 'a6a4', 'c8c7', 'g1f2', 'b5c3', 'a4d7', 'c7d7', 'g2h4', 'c3d5', 'h4f3', 'd5e3', 'f2e3', 'd7c7', 'e1b1', 'g8f8', 'd3d4', 'f8e8', 'e3d3', 'e8d8', 'e2e4', 'd8c8', 'b1b3', 'c8b7', 'f3e5', 'f7f6', 'e5c4', 'c7d7', 'c4a5', 'b7a8', 'a5c4', 'd7d8', 'h2h3', 'g7f8', 'd3e3', 'd8b8', 'e4e5', 'f6e5', 'f4e5', 'b6b5', 'c4a5', 'b5b4', 'e3e4', 'b8b5', 'b3f3', 'b5a5', 'f3f8', 'a8b7', 'f8f7', 'b7c8', 'f7f8', 'c8c7', 'f8f7', 'c7c6', 'f7e7', 'b4b3', 'e7e6', 'c6d7', 'e6d6', 'd7c7', 'd6f6', 'b3b2', 'f6f7', 'c7b6', 'f7f6', 'b6b5', 'f6f1', 'a5a1']


move = chess.Move.from_uci(arr1[0])
board.push(move)


board = get_board(arr1, 2)
print(board)

board = chess.Board()
display.start(board.fen())
counter = 0
while True:
    move = chess.Move.from_uci(arr1[counter])
    board.push(move)
    display.update(board.fen())
    sleep(1)
    counter += 1

    if display.check_for_quit():
        display.terminate()

# svg = chess.svg.board(
#     board,
#     fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
#     arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
#     squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
#     size=350,
# ) 


# # Convert SVG to PNG in memory
# png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
# img = Image.open(io.BytesIO(png_bytes))

# # Tkinter GUI
# root = tk.Tk()
# root.title("Chess Board")

# tk_img = ImageTk.PhotoImage(img)
# label = tk.Label(root, image=tk_img)
# label.pack()

# root.mainloop()
