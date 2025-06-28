import os
import torch
from graph_encode import  encode_node_features, encode_edge_features, static_edge_index, static_edge_map, base_graph_edges, index_to_move

from our_model import ChessGNN
import chess
import numpy as np

    
def start_model(checkpoint_file_name="checkpoints/ffv1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NODE_IN_FEATURES = 21
    EDGE_IN_FEATURES = 11
    GNN_NODE_OUT_FEATURES = 56
    NUM_POSSIBLE_MOVES = 1792

    model = ChessGNN(
        node_in_features=NODE_IN_FEATURES,
        edge_in_features=EDGE_IN_FEATURES,
        gnn_hidden_features=GNN_NODE_OUT_FEATURES,
        num_possible_moves=NUM_POSSIBLE_MOVES,
    ).to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(script_dir, checkpoint_file_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def model_predict(model, board):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = torch.tensor(encode_node_features(board)).to(device) # [B * 64, 21]
    edge_features = torch.tensor(encode_edge_features(board, base_graph_edges)).to(device) # [B * 1792, 11]
    edge_index = static_edge_index.to(device)             # [B * 1792, 11]
    edge_map = static_edge_map.to(device)                 # [1792]

    policy_logits, value_pred = model(
        node_feature_matrix=node_features,
        edge_feature_matrix=edge_features,
        edge_index=edge_index,
        edge_map=edge_map,
        batch_size=1
    )

    policy_logits = policy_logits.to("cpu")
    policy_logits = policy_logits.detach().numpy()

    value_pred = value_pred.to("cpu")
    value_pred = value_pred.detach().numpy()

    return policy_logits, value_pred

if __name__ == '__main__':
    board = chess.Board()
    model = start_model()

    while(True):
        policy_logits, value_pred = model_predict(model, board)

        # print(value_pred)
        index = np.argmax(policy_logits)

        print(index_to_move[index], end="', '")
        board.push(index_to_move[index])