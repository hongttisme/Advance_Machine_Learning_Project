import os
import pandas as pd
import chess.pgn
from tqdm import tqdm
import time


PGN_FOLDER_PATH = 'C:\\Users\\tan04\\Downloads\\KingBaseLite2019-pgn' 

OUTPUT_CSV_PATH = 'kingbase_processed_all.csv'


MIN_ELO = 2200         
MIN_MOVES = 10        
EXCLUDE_DRAWS = True   

FILE_LIMIT = None


def process_all_pgns_in_folder(folder_path, output_path, min_elo, min_moves, exclude_draws, file_limit=None):
    """
    扫描一个文件夹中的所有 PGN 文件，进行预处理，并合并成一个 CSV 文件。
    """
    print("--- 开始预处理流程 ---")
    
    print(f"正在扫描文件夹: {folder_path}")
    try:
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pgn')]
        if not all_files:
            print(f"错误: 在指定文件夹中未找到任何 .pgn 文件。请检查路径: '{folder_path}'")
            return
    except FileNotFoundError:
        print(f"错误: 文件夹不存在 -> '{folder_path}'。请确保路径正确。")
        return

    print(f"在文件夹中找到 {len(all_files)} 个 PGN 文件。")
    
    files_to_process = all_files
    if file_limit is not None:
        files_to_process = all_files[:file_limit]
        print(f"注意: 已设置文件处理上限，本次将只处理前 {len(files_to_process)} 个文件。")

    all_games_data = []
    total_games_scanned = 0

    for pgn_filename in tqdm(files_to_process, desc="正在处理PGN文件"):
        file_path = os.path.join(folder_path, pgn_filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break 
                        
                        total_games_scanned += 1
                        
                        headers = game.headers
                        result = headers.get('Result', '*')
                        if result == '*' or (exclude_draws and result == '1/2-1/2'):
                            continue
                        
                        try:
                            white_elo = int(headers.get('WhiteElo', 0))
                            black_elo = int(headers.get('BlackElo', 0))
                            if white_elo < min_elo or black_elo < min_elo:
                                continue
                        except ValueError:
                            continue

                        num_moves = game.end().board().fullmove_number
                        if num_moves < min_moves:
                            continue

                        moves = [move.uci() for move in game.mainline_moves()]
                        game_data = {
                            'Event': headers.get('Event', 'N/A'),
                            'Site': headers.get('Site', 'N/A'),
                            'Date': headers.get('Date', 'N/A'),
                            'White': headers.get('White', 'N/A'),
                            'Black': headers.get('Black', 'N/A'),
                            'Result': result,
                            'WhiteElo': white_elo,
                            'BlackElo': black_elo,
                            'ECO': headers.get('ECO', 'N/A'),
                            'TotalMoves': num_moves,
                            'Moves_UCI': ' '.join(moves)
                        }
                        all_games_data.append(game_data)
                    except (ValueError, KeyError, IndexError) as e:

                        continue
        except Exception as e:
            print(f"\n错误: 无法处理文件 {pgn_filename}。错误: {e}")

    print("\n--- 所有文件处理完毕 ---")
    print(f"总共扫描了 {total_games_scanned} 场对局。")
    
    if not all_games_data:
        print("没有找到任何符合条件的对局。请检查您的过滤参数或 PGN 文件。")
        return

    print(f"找到 {len(all_games_data)} 场符合条件的对局。正在整合并保存到 CSV 文件...")
    df = pd.DataFrame(all_games_data)
    
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 预处理成功完成！")
        print(f"所有数据已合并并保存至: {output_path}")
        print(f"总共保存了 {len(df)} 行数据。")
    except Exception as e:
        print(f"\n错误: 保存 CSV 文件失败。错误: {e}")

if __name__ == '__main__':
    start_time = time.time()
    
    process_all_pgns_in_folder(
        folder_path=PGN_FOLDER_PATH,
        output_path=OUTPUT_CSV_PATH,
        min_elo=MIN_ELO,
        min_moves=MIN_MOVES,
        exclude_draws=EXCLUDE_DRAWS,
        file_limit=FILE_LIMIT
    )
    
    end_time = time.time()
    print(f"\n总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    