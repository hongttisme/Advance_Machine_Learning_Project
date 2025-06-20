import chess
from chessboard import display 
import numpy as np
import time

SQUARES = chess.SQUARES
SQUARE_NAMES = chess.SQUARE_NAMES
BOARD_SIZE = 8

nodes = list(SQUARES)


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
    
    node_features = np.zeros((64, 21), dtype=np.float32)

    current_player = 1.0 if board.turn == chess.WHITE else 0.0
    total_move_count = board.fullmove_number
    no_progress_count = board.halfmove_clock
    
    castling_rights = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)

    for i, square in enumerate(SQUARES):
        
        feature_offset = 0


        piece = board.piece_at(square)
        if piece:
            piece_idx = PIECE_MAP[(piece.piece_type, piece.color)]
            node_features[i, feature_offset + piece_idx] = 1.0
        feature_offset += 12
        
        if board.is_repetition(2):
            node_features[i, feature_offset] = 1.0
        if board.is_repetition(3):
            node_features[i, feature_offset + 1] = 1.0
        feature_offset += 2
        
        node_features[i, feature_offset] = current_player
        node_features[i, feature_offset + 1] = total_move_count
        node_features[i, feature_offset + 2: feature_offset + 6] = castling_rights
        node_features[i, feature_offset + 6] = no_progress_count

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

display.start(board.fen())
while True:
    if display.check_for_quit():
        display.terminate()
