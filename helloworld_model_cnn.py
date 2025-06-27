import os
import torch
import chess
import numpy as np

from graph_encode import encode_node_features, index_to_move, move_to_index
from model_CNN import AlphaZeroLikeCNN




def start_model(checkpoint_file_name="checkpoint_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AlphaZeroLikeCNN().to(device)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, checkpoint_file_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() 
    return model

def state_to_tensor(board: chess.Board):
    tensor = encode_node_features(board)
    return tensor.reshape((21, 8, 8))


def model_predict(model, board):
    device = next(model.parameters()).device
    x = state_to_tensor(board)
    x_tensor = torch.from_numpy(x).float().to(device).unsqueeze(0)

    with torch.no_grad(): 
        value_pred, policy_logits = model(x_tensor)

    policy_logits = policy_logits.cpu().numpy().squeeze(0)
    value_pred = value_pred.cpu().numpy()

    return policy_logits, value_pred

if __name__ == '__main__':
    board = chess.Board()
    model = start_model()

    if model is None:
        print("Model could not be loaded. Exiting.")
        exit() 

    game_turn = 0
    while not board.is_game_over():
        if game_turn >= 10: 
            print("Reached move limit.")
            break
        
        game_turn += 1
        print(f"\n--- Turn {game_turn} ({'White' if board.turn else 'Black'}) ---")
        print(board)

        policy_logits, value_pred = model_predict(model, board)

        legal_moves = list(board.legal_moves)


        masked_policy_logits = np.full_like(policy_logits, -1e9)

        legal_move_indices = []
        for move in legal_moves:
            if move in move_to_index:
                idx = move_to_index[move]
                masked_policy_logits[idx] = policy_logits[idx]
                legal_move_indices.append(idx)
        

            
        best_move_index = np.argmax(masked_policy_logits)
        chosen_move = index_to_move[best_move_index]

      

        print(f"Model predicted value: {value_pred[0][0]:.4f}")
        print(f"Chosen move: {chosen_move}")
        
        board.push(chosen_move)

    print("\n--- Game Over ---")
    print(f"Result: {board.result()}")
    print(board)