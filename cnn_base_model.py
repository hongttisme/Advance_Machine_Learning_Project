import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import chess
import numpy as np
from tqdm import tqdm
import os
import random
import time
from sklearn.model_selection import train_test_split
from graph_encode import move_to_index, index_to_move, encode_node_features



CSV_FILE_PATH = 'C:\\Users\\tan04\\Documents\\codeplace\\AI\\advance ML\\Advance_Machine_Learning_Project\\kingbase_processed_all.csv' 
# CSV_FILE_PATH = 'D:\\programming\\github\\Advance_Machine_Learning_Project\\kingbase_processed_all.csv' 
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
MODEL_SAVE_PATH = 'simple_chess_cnn_v2.pth'
TEST_SIZE = 0.1
VAL_SIZE = 0.1

NUM_POSSIBLE_MOVES = len(index_to_move)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def uci_to_index(uci_move):
    try:
        move = chess.Move.from_uci(uci_move)
        return move_to_index[chess.Move(move.from_square, move.to_square)]
    except:
        print("error encode!!", chess.Move.from_uci(uci_move)) 
        return -1

def state_to_tensor(board: chess.Board):
    tensor = encode_node_features(board)
    return tensor.T.reshape((21, 8, 8))


def result_to_value(result: str):
    if result == '1-0': return 1.0
    elif result == '0-1': return -1.0
    return 0.0


class ChessDataset(Dataset):
    def __init__(self, dataframe): 
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        moves_uci = row['Moves_UCI'].split()
        result = row['Result']
        if len(moves_uci) < 2:
            return self.__getitem__(random.randint(0, len(self) - 1))
        move_idx_to_play = random.randint(0, len(moves_uci) - 1)
        board = chess.Board()
        for move_uci in moves_uci[:move_idx_to_play]:
            try: board.push_uci(move_uci)
            except: return self.__getitem__(random.randint(0, len(self) - 1))
        state_tensor = state_to_tensor(board)
        target_move_uci = moves_uci[move_idx_to_play]
        target_move_index = uci_to_index(target_move_uci)
        game_value = result_to_value(result)
        if board.turn == chess.BLACK: game_value = -game_value
        return state_tensor, target_move_index, game_value

class SimpleChessCNN(nn.Module):
    def __init__(self):
        super(SimpleChessCNN, self).__init__()
        num_input_channels = 21
        num_possible_moves = NUM_POSSIBLE_MOVES
        self.body = nn.Sequential(nn.Conv2d(num_input_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        flattened_size = 64 * 8 * 8
        self.value_head = nn.Sequential(nn.Linear(flattened_size, 512), nn.ReLU(), nn.Linear(512, 1), nn.Tanh())
        self.policy_head = nn.Sequential(nn.Linear(flattened_size, 1024), nn.ReLU(), nn.Linear(1024, num_possible_moves))
    def forward(self, x):
        x = self.body(x)
        x_flattened = x.view(x.size(0), -1)
        value = self.value_head(x_flattened)
        policy_logits = self.policy_head(x_flattened)
        return value, policy_logits


def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始在 {device} 上训练...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}\n\n")
    
    for epoch in range(epochs):
        # --- 训练部分 ---
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
        for states, target_moves, target_values in train_pbar:
            states, target_moves, target_values = states.to(device), target_moves.to(device), target_values.to(device).float()
            optimizer.zero_grad()
            pred_values, pred_policies = model(states)
            loss_v = criterion_value(pred_values.squeeze(), target_values)
            loss_p = criterion_policy(pred_policies, target_moves)
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

        # --- 验证部分 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [验证]")
            for states, target_moves, target_values in val_pbar:
                states, target_moves, target_values = states.to(device), target_moves.to(device), target_values.to(device).float()
                pred_values, pred_policies = model(states)
                loss_v = criterion_value(pred_values.squeeze(), target_values)
                loss_p = criterion_policy(pred_policies, target_moves)
                val_loss += (loss_v.item() + loss_p.item())
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} 结束 | 平均训练损失: {avg_train_loss:.4f} | 平均验证损失: {avg_val_loss:.4f}")

    print("训练完成！")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 开始在测试集上最终评估 ---")
    model.to(device)
    model.eval()
    test_loss = 0.0
    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[测试]")
        for states, target_moves, target_values in test_pbar:
            states, target_moves, target_values = states.to(device), target_moves.to(device), target_values.to(device).float()
            pred_values, pred_policies = model(states)
            loss_v = criterion_value(pred_values.squeeze(), target_values)
            loss_p = criterion_policy(pred_policies, target_moves)
            test_loss += (loss_v.item() + loss_p.item())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"最终测试损失: {avg_test_loss:.4f}")

if __name__ == '__main__':


    if not os.path.exists(CSV_FILE_PATH):
        print(f"错误: 数据文件 {CSV_FILE_PATH} 不存在。请先运行预处理脚本。")
    else:
        full_df = pd.read_csv(CSV_FILE_PATH)
        print(f"成功加载完整数据集，共 {len(full_df)} 条记录。")




        train_val_df, test_df = train_test_split(full_df, test_size=TEST_SIZE, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=(VAL_SIZE / (1 - TEST_SIZE)), random_state=42)
        
        print(f"数据划分完成:")
        print(f" - 训练集: {len(train_df)} 条")
        print(f" - 验证集: {len(val_df)} 条")
        print(f" - 测试集: {len(test_df)} 条")

        train_dataset = ChessDataset(dataframe=train_df)
        val_dataset = ChessDataset(dataframe=val_df)
        test_dataset = ChessDataset(dataframe=test_df)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        cnn_model = SimpleChessCNN()
        train_model(cnn_model, train_loader, val_loader, epochs=EPOCHS)
        
        test_model(cnn_model, test_loader)