import chess
import numpy as np
import time
import torch

SQUARES = chess.SQUARES
SQUARE_NAMES = chess.SQUARE_NAMES
BOARD_SIZE = 8

nodes = list(SQUARES)

def encode_global_node_features(board: chess.Board):
    """
    Encodes the global features of a board state into a single vector.
    
    Features:
    - Current player (1)
    - Total move count (1)
    - Half-move clock (no progress count) (1)
    - Castling rights (4)
    - Repetition counts (2)
    Total: 9 features
    """
    
    # Feature 1: Current player (1.0 for WHITE, 0.0 for BLACK)
    current_player = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Feature 2: Total move count
    total_move_count = board.fullmove_number
    
    # Feature 3: No progress count (for 50-move rule)
    no_progress_count = board.halfmove_clock
    
    # Features 4-7: Castling rights (WK, WQ, BK, BQ)
    castling_rights = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)

    # Features 8-9: Repetition info
    is_2_fold_repetition = 1.0 if board.is_repetition(2) else 0.0
    is_3_fold_repetition = 1.0 if board.is_repetition(3) else 0.0
    
    global_features = np.array([
        current_player,
        total_move_count,
        no_progress_count,
        *castling_rights, # Unpack the 4 castling rights
        is_2_fold_repetition,
        is_3_fold_repetition
    ], dtype=np.float32)

    return global_features

def get_move_type(move: chess.Move):
    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    is_knight_move = (abs(dx), abs(dy)) in [(1, 2), (2, 1)]
    is_bishop_move = abs(dx) == abs(dy)
    is_rook_move = dx == 0 or dy == 0
    is_king_move = max(abs(dx), abs(dy)) == 1
    
    is_pawn_move = abs(dx) <= 1 and dy == 1 

    return {
        'knight': is_knight_move,
        'bishop': is_bishop_move and not is_king_move,
        'rook': is_rook_move and not is_king_move,
        'queen': (is_bishop_move or is_rook_move) and not is_king_move,
        'king': is_king_move,
        'pawn': is_pawn_move
    }


base_graph_edges = []
move_to_index = {}
index_to_move = {}
adjacency_list = []

for i, start_square in enumerate(SQUARES):
    adjacency_list.append([])
    for j, end_square in enumerate(SQUARES):
        if start_square == end_square:
            continue
        type_dict = get_move_type(chess.Move(start_square, end_square))
        skip_flag = True
        for is_protential_move in type_dict.values():
            if is_protential_move:
                skip_flag = False
                break

        if skip_flag :
            continue

        new_index = len(base_graph_edges)
        adjacency_list[i].append((j,new_index))


        the_move = chess.Move(start_square, end_square)
        base_graph_edges.append(the_move)
        move_to_index[the_move] = new_index
        index_to_move[new_index] = the_move



print(f"created {len(nodes)} nodes")
print(f"created {len(base_graph_edges)} edge")



PIECE_MAP = {
    (chess.PAWN, chess.WHITE): 0, (chess.PAWN, chess.BLACK): 1,
    (chess.KNIGHT, chess.WHITE): 2, (chess.KNIGHT, chess.BLACK): 3,
    (chess.BISHOP, chess.WHITE): 4, (chess.BISHOP, chess.BLACK): 5,
    (chess.ROOK, chess.WHITE): 6, (chess.ROOK, chess.BLACK): 7,
    (chess.QUEEN, chess.WHITE): 8, (chess.QUEEN, chess.BLACK): 9,
    (chess.KING, chess.WHITE): 10, (chess.KING, chess.BLACK): 11,
}

def encode_node_features(board: chess.Board):
    """
    Encodes the local features for each of the 64 nodes (squares).
    
    Features:
    - One-hot encoding of the piece on the square (12 possibilities)
    Total: 12 features per node
    """
    
    # The feature vector for each node will have 12 elements,
    # one for each piece type and color.
    node_features = np.zeros((64, 12), dtype=np.float32)

    for i, square in enumerate(SQUARES):
        piece = board.piece_at(square)
        if piece:
            # Get the index for the piece (e.g., white pawn is 0, black pawn is 1, etc.)
            piece_idx = PIECE_MAP[(piece.piece_type, piece.color)]
            # Set the corresponding feature to 1.0
            node_features[i, piece_idx] = 1.0
            
    return node_features




def encode_edge_features(board: chess.Board, edges: list):
    edge_features = np.zeros((len(edges), 11), dtype=np.float32)

    for i, move in enumerate(edges):
        offset = 0
        
        if move in board.legal_moves:
            edge_features[i, offset] = 1.0
        offset += 1

        dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        edge_features[i, offset] = dx
        edge_features[i, offset + 1] = dy
        offset += 2

 

        move_type = get_move_type(move)
        edge_features[i, offset] = 1.0 if move_type['pawn'] else 0.0
        edge_features[i, offset + 1] = 1.0 if move_type['pawn'] else 0.0
        offset += 2
        edge_features[i, offset] = 1.0 if move_type['knight'] else 0.0
        edge_features[i, offset + 1] = 1.0 if move_type['bishop'] else 0.0
        edge_features[i, offset + 2] = 1.0 if move_type['rook'] else 0.0
        edge_features[i, offset + 3] = 1.0 if move_type['queen'] else 0.0
        offset += 4
        edge_features[i, offset] = 1.0 if move_type['king'] else 0.0
        edge_features[i, offset + 1] = 1.0 if move_type['king'] else 0.0
        offset += 2
        
    return edge_features



board = chess.Board()


moves_to_play = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
for move_uci in moves_to_play:
    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        board.push(move)

print("current state (FEN):", board.fen())


node_feature_matrix = encode_node_features(board)
edge_feature_matrix = encode_edge_features(board, base_graph_edges)


print("\nencode result:")
print(f"node matrix shape: {node_feature_matrix.shape}") 
print(f"edge matrix shape: {edge_feature_matrix.shape}") 

# display.start(board.fen())
# while True:
#     if display.check_for_quit():
#         display.terminate()


edge_list_source = []
edge_list_target = []

for i, start_square in enumerate(SQUARES):
    for end_square_idx, _ in adjacency_list[i]:
        edge_list_source.append(i) 
        edge_list_target.append(end_square_idx) 


static_edge_index = torch.tensor([edge_list_target, edge_list_source], dtype=torch.long)

static_edge_map = torch.arange(len(base_graph_edges), dtype=torch.long)


print("--- Static Graph Components ---")
print(f"static_edge_index shape: {static_edge_index.shape}")
print(f"static_edge_map shape: {static_edge_map.shape}")
print("-" * 30)



def create_batch_from_boards(board_list: list[chess.Board]):
    """
    Creates a batch of graph data from a list of board states,
    including the new global node vectors.
    """
    batch_node_features = []
    batch_edge_features = []
    # --- NEW: List to store global features for each board ---
    batch_global_features = []

    for board in board_list:
        node_features_np = encode_node_features(board)           # Shape: (64, 12)
        edge_features_np = encode_edge_features(board, base_graph_edges) # Shape: (1792, 11)
        # --- NEW: Encode and append the global features ---
        global_features_np = encode_global_node_features(board) # Shape: (9,)

        batch_node_features.append(node_features_np)
        batch_edge_features.append(edge_features_np)
        # --- NEW: Add to the batch list ---
        batch_global_features.append(global_features_np)

    # Concatenate node and edge features as before
    batch_node_features_np = np.concatenate(batch_node_features, axis=0)
    batch_edge_features_np = np.concatenate(batch_edge_features, axis=0)
    
    # --- NEW: Stack global features into a batch tensor ---
    # Use np.stack to create a (batch_size, num_global_features) tensor
    batch_global_features_np = np.stack(batch_global_features, axis=0)


    batch_data = {
        'node_feature_matrix': torch.from_numpy(batch_node_features_np).float(),
        'edge_feature_matrix': torch.from_numpy(batch_edge_features_np).float(),
        # --- NEW: Add the global node vector to the batch data dictionary ---
        'global_node_vector': torch.from_numpy(batch_global_features_np).float(),
        'edge_index': static_edge_index,
        'edge_map': static_edge_map,
    }
    
    return batch_data




