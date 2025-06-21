import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadGATEAULayer(nn.Module):
    """
    GATEAULayer with Multi-Head Attention.
    """
    def __init__(self, node_in_features, edge_in_features, node_out_features, num_heads=8):
        super(MultiHeadGATEAULayer, self).__init__()
        assert node_out_features % num_heads == 0, "node_out_features must be divisible by num_heads"

        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.node_out_features = node_out_features
        self.num_heads = num_heads
        self.head_dim = node_out_features // num_heads
        

        self.Wv = nn.Parameter(torch.randn(node_in_features, edge_in_features))
        self.Wu = nn.Parameter(torch.randn(node_in_features, edge_in_features))
        self.We = nn.Parameter(torch.randn(edge_in_features, edge_in_features))
        
        self.Wh = nn.Parameter(torch.randn(node_in_features, node_out_features))
        self.Wg = nn.Parameter(torch.randn(edge_in_features, node_out_features))
        self.W0 = nn.Parameter(torch.randn(node_in_features, node_out_features))

        self.a = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.W_out = nn.Linear(node_out_features, node_out_features)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wu)
        nn.init.xavier_uniform_(self.We)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.xavier_uniform_(self.Wg)
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)
        nn.init.xavier_uniform_(self.a)


    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index, edge_map):
        num_nodes = node_feature_matrix.shape[0]
        num_edges = edge_index.shape[1]
        target_node_idx, source_node_idx = edge_index[0], edge_index[1]


        h_nodes_v = node_feature_matrix @ self.Wv
        h_nodes_u = node_feature_matrix @ self.Wu
        h_edges_e = edge_feature_matrix @ self.We

        # These transformations produce the values to be aggregated, so they need to be split into heads.
        h_nodes_h = (node_feature_matrix @ self.Wh).view(-1, self.num_heads, self.head_dim)
        h_edges_g = (edge_feature_matrix @ self.Wg).view(-1, self.num_heads, self.head_dim)
        h_nodes_0 = (node_feature_matrix @ self.W0).view(-1, self.num_heads, self.head_dim)

        target_node_feats_for_attention = h_nodes_u[target_node_idx]
        source_node_feats_for_attention = h_nodes_v[source_node_idx]
        edge_feats_for_attention = h_edges_e[edge_map]
        
        new_edge_feature_for_attention = target_node_feats_for_attention + source_node_feats_for_attention + edge_feats_for_attention
        new_edge_feature_for_attention = new_edge_feature_for_attention.unsqueeze(1).repeat(1, self.num_heads, 1) # Shape: (num_edges, num_heads, edge_in_features)
        

        if not hasattr(self, 'a_proj'):
            self.a_proj = nn.Linear(self.edge_in_features, self.num_heads).to(node_feature_matrix.device)
            nn.init.xavier_uniform_(self.a_proj.weight)

        attention_logits = self.a_proj(new_edge_feature_for_attention[:, 0, :]).view(num_edges, self.num_heads) 
        attention_scores = self.leaky_relu(attention_logits) 
        max_scores = torch.full((num_nodes, self.num_heads), -1e9, device=attention_scores.device, dtype=attention_scores.dtype)
        max_scores.scatter_reduce_(0, target_node_idx.unsqueeze(-1).expand_as(attention_scores), attention_scores, reduce="amax", include_self=False)
        
        scores_max_per_edge = max_scores[target_node_idx]
        attention_scores_exp = torch.exp(attention_scores - scores_max_per_edge) # Shape: (num_edges, num_heads)

        sum_exp_scores = torch.zeros_like(max_scores) # Shape: (num_nodes, num_heads)
        sum_exp_scores.index_add_(0, target_node_idx, attention_scores_exp)
        
        sum_exp_per_edge = sum_exp_scores[target_node_idx]

        alpha = attention_scores_exp / (sum_exp_per_edge + 1e-10) # Shape: (num_edges, num_heads)


        source_node_values = h_nodes_h[source_node_idx] # Shape: (num_edges, num_heads, head_dim)
        edge_values = h_edges_g[edge_map]             # Shape: (num_edges, num_heads, head_dim)
        values = source_node_values + edge_values
        
        # alpha needs to be unsqueezed to match value dimensions for broadcasting
        weighted_values = values * alpha.unsqueeze(-1) # Shape: (num_edges, num_heads, head_dim)

        aggregated_messages = torch.zeros_like(h_nodes_0) # Shape: (num_nodes, num_heads, head_dim)
        aggregated_messages.index_add_(0, target_node_idx, weighted_values)

        # Combine with skip connection and concatenate heads
        new_h = h_nodes_0 + aggregated_messages # Shape: (num_nodes, num_heads, head_dim)
        
        # Concatenate heads
        concatenated_h = new_h.view(-1, self.node_out_features) # Shape: (num_nodes, node_out_features)
        
        # Apply final linear projection
        new_final = self.W_out(concatenated_h)


        new_edge_feature = target_node_feats_for_attention + source_node_feats_for_attention + edge_feats_for_attention
        
        return new_final, new_edge_feature


class BNR(nn.Module):
    def __init__(self, num_features):
        super(BNR, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResGATEAU(nn.Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features, num_heads=8):
        super(ResGATEAU, self).__init__()

        self.bnr1 = BNR(node_in_features)
        self.gateau1 = MultiHeadGATEAULayer(node_in_features, edge_in_features, node_out_features, num_heads=num_heads)
        
        self.bnr2 = BNR(node_out_features)
        # The edge features for the second layer are the output of the first layer's attention calculation logic.
        # The dimension is `edge_in_features`.
        self.gateau2 = MultiHeadGATEAULayer(node_out_features, edge_in_features, node_out_features, num_heads=num_heads)
        
        if node_in_features != node_out_features:
            self.residual_transform = nn.Linear(node_in_features, node_out_features)
        else:
            self.residual_transform = nn.Identity()

    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index, edge_map):
        residual = self.residual_transform(node_feature_matrix)

        x = self.bnr1(node_feature_matrix)
        
        # The first layer uses the original edge features
        x, e1 = self.gateau1(x, edge_feature_matrix, edge_index, edge_map)
        
        x = self.bnr2(x)

        # The second layer uses the new edge features `e1` from the first layer
        x, e2 = self.gateau2(x, e1, edge_index, edge_map)

        output_node_features = residual + x
        
        return output_node_features, e2


class ChessGNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, gnn_hidden_features, num_possible_moves, num_res_layers=10, num_heads=8):
        super(ChessGNN, self).__init__()
        
        self.gnn_layers = nn.ModuleList()

        # First ResGATEAU layer
        self.gnn_layers.append(ResGATEAU(
            node_in_features=node_in_features,
            edge_in_features=edge_in_features,
            node_out_features=gnn_hidden_features,
            num_heads=num_heads
        ))

        # Subsequent ResGATEAU layers
        for _ in range(num_res_layers - 1):
            self.gnn_layers.append(ResGATEAU(
                node_in_features=gnn_hidden_features,
                edge_in_features=edge_in_features, # Assuming edge feature dimension remains constant
                node_out_features=gnn_hidden_features,
                num_heads=num_heads
            ))

        self.policy_head = nn.Sequential(
            nn.Linear(64 * gnn_hidden_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_possible_moves)
        )

        self.value_head = nn.Sequential(
            nn.Linear(64 * gnn_hidden_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() 
        )

    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index, edge_map, batch_size):
        x = node_feature_matrix
        e = edge_feature_matrix
        for layer in self.gnn_layers:
            x, e = layer(x, e, edge_index, edge_map) # Pass the updated edge features 'e' to the next layer

        processed_node_features = x
        
        # Reshape to (batch_size, num_nodes * features) for the policy and value heads
        graph_representation = processed_node_features.view(batch_size, -1)

        policy_logits = self.policy_head(graph_representation)
        value = self.value_head(graph_representation)

        return policy_logits, value