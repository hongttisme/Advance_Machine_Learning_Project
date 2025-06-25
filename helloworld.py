import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from our_graph2_encode import move_to_index, encode_node_features, create_batch_from_boards, encode_edge_features, encode_global_node_features, static_edge_index, static_edge_map, base_graph_edges, index_to_move

from torch.utils.data import DataLoader
import torch.nn as nn
from our_model2 import ChessGNN
from torch.utils.data import Dataset, DataLoader
import random
import chess
from tqdm import tqdm
import numpy as np
from chessboard import display
from time import sleep



def uci_to_index(uci_move):
    try:
        move = chess.Move.from_uci(uci_move)
        return move_to_index[chess.Move(move.from_square, move.to_square)]
    except:
        print("error encode!!", chess.Move.from_uci(uci_move)) 
        return -1


board = chess.Board()



print(encode_node_features(board).shape, "flag")
print(encode_edge_features(board, base_graph_edges).shape, "flag")
print(encode_global_node_features(board).shape, "flag")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


NODE_IN_FEATURES = 12
EDGE_IN_FEATURES = 11
GNN_NODE_OUT_FEATURES = 56
GLOBAL_NODE_IN_FEATURES = 9
NUM_POSSIBLE_MOVES = 1792
BATCH_SIZE = 128
TEST_SIZE = 0.1
VAL_SIZE = 0.1

model = ChessGNN(
    node_in_features=NODE_IN_FEATURES,
    edge_in_features=EDGE_IN_FEATURES,
    global_node_in_features=GLOBAL_NODE_IN_FEATURES,
    gnn_hidden_features=GNN_NODE_OUT_FEATURES,
    num_possible_moves=NUM_POSSIBLE_MOVES,
).to(device)


script_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(script_dir, "checkpoint_epoch_41.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# display.start(board.fen())

while(True):
    node_features = torch.tensor(encode_node_features(board)).to(device) # [B * 64, 12]
    edge_features = torch.tensor(encode_edge_features(board, base_graph_edges)).to(device) # [B * 1792, 11]
    global_features = torch.tensor(encode_global_node_features(board).reshape(1,9)).to(device)# [B * 1, 9]
    edge_index = static_edge_index.to(device)             # [B * 1792, 11]
    edge_map = static_edge_map.to(device)                 # [1792]



    policy_logits, value_pred = model(
        node_feature_matrix=node_features,
        edge_feature_matrix=edge_features,
        global_node_vector=global_features,
        edge_index=edge_index,
        edge_map=edge_map,
        batch_size=1
    )

    # print(value_pred)
    policy_logits = policy_logits.to("cpu")
    index = np.argmax(policy_logits.detach().numpy())

    print(index_to_move[index], end="', '")
    board.push(index_to_move[index])

    # if display.check_for_quit():
    #     display.terminate()

    # display.update(fen=board, game_board=board)
    # sleep(1)