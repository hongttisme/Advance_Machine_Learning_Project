import torch.nn as nn
from graph_encode import index_to_move


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class AlphaZeroLikeCNN(nn.Module):
    def __init__(self, num_input_channels=21, num_residual_blocks=24, num_filters=128):
        super(AlphaZeroLikeCNN, self).__init__()
        num_possible_moves = len(index_to_move)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8*8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh() 
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_possible_moves)
        )

    def forward(self, x):
        # x shape: (batch, 21, 8, 8)
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        return value, policy_logits