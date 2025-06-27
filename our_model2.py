import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadGATEAULayer(nn.Module):

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
        self.W_global_edge = nn.Parameter(torch.randn(node_in_features, edge_in_features))
        
        self.Wh = nn.Parameter(torch.randn(node_in_features, node_out_features))
        self.Wg = nn.Parameter(torch.randn(edge_in_features, node_out_features))
        self.W0 = nn.Parameter(torch.randn(node_in_features, node_out_features))
        self.W_global_node = nn.Parameter(torch.randn(node_in_features, node_out_features))

        self.a = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.W_out = nn.Linear(node_out_features, node_out_features)

        self.a_proj = nn.Linear(self.edge_in_features, self.num_heads)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wu)
        nn.init.xavier_uniform_(self.We)
        nn.init.xavier_uniform_(self.W_global_edge)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.xavier_uniform_(self.Wg)
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W_global_node)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)
        nn.init.xavier_uniform_(self.a)

    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index, edge_map, global_node_features):
        num_nodes = node_feature_matrix.shape[0]
        num_edges = edge_index.shape[1]
        target_node_idx, source_node_idx = edge_index[0], edge_index[1]

        h_nodes_v = node_feature_matrix @ self.Wv
        h_nodes_u = node_feature_matrix @ self.Wu
        h_edges_e = edge_feature_matrix @ self.We
        h_global = global_node_features @ self.W_global_edge

        h_nodes_h = (node_feature_matrix @ self.Wh).view(-1, self.num_heads, self.head_dim)
        h_edges_g = (edge_feature_matrix @ self.Wg).view(-1, self.num_heads, self.head_dim)
        h_nodes_0 = (node_feature_matrix @ self.W0).view(-1, self.num_heads, self.head_dim)

        target_node_feats_for_attention = h_nodes_u[target_node_idx]
        source_node_feats_for_attention = h_nodes_v[source_node_idx]
        edge_feats_for_attention = h_edges_e[edge_map]
        global_feats_for_attention = h_global[target_node_idx]
        
        new_edge_feature_for_attention = target_node_feats_for_attention + source_node_feats_for_attention + edge_feats_for_attention + global_feats_for_attention
        new_edge_feature_for_attention = new_edge_feature_for_attention.unsqueeze(1).repeat(1, self.num_heads, 1)
        
        if not hasattr(self, 'a_proj'):
            nn.init.xavier_uniform_(self.a_proj.weight)

        attention_logits = self.a_proj(new_edge_feature_for_attention[:, 0, :]).view(num_edges, self.num_heads) 
        attention_scores = self.leaky_relu(attention_logits) 
        max_scores = torch.full((num_nodes, self.num_heads), -1e9, device=attention_scores.device, dtype=attention_scores.dtype)
        max_scores.scatter_reduce_(0, target_node_idx.unsqueeze(-1).expand_as(attention_scores), attention_scores, reduce="amax", include_self=False)
        
        scores_max_per_edge = max_scores[target_node_idx]
        attention_scores_exp = torch.exp(attention_scores - scores_max_per_edge)

        sum_exp_scores = torch.zeros_like(max_scores)
        sum_exp_scores.index_add_(0, target_node_idx, attention_scores_exp)
        
        sum_exp_per_edge = sum_exp_scores[target_node_idx]
        alpha = attention_scores_exp / (sum_exp_per_edge + 1e-10)

        source_node_values = h_nodes_h[source_node_idx]
        edge_values = h_edges_g[edge_map]
        values = source_node_values + edge_values
        
        weighted_values = values * alpha.unsqueeze(-1)

        aggregated_messages = torch.zeros_like(h_nodes_0)
        aggregated_messages.index_add_(0, target_node_idx, weighted_values)
        
        projected_global_for_update = (global_node_features @ self.W_global_node).view(-1, self.num_heads, self.head_dim)

        new_h = h_nodes_0 + aggregated_messages + projected_global_for_update
        
        concatenated_h = new_h.view(-1, self.node_out_features)
        new_final = self.W_out(concatenated_h)
        new_edge_feature = target_node_feats_for_attention + source_node_feats_for_attention + edge_feats_for_attention + global_feats_for_attention
        
        return new_final, new_edge_feature

class GATEAUTransformerBlock(nn.Module):
    def __init__(self, node_features, edge_features, num_heads, ff_dim, dropout=0.1):
        super(GATEAUTransformerBlock, self).__init__()
        
        self.attention = MultiHeadGATEAULayer(
            node_in_features=node_features,
            edge_in_features=edge_features,
            node_out_features=node_features, 
            num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(node_features)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(
            model_dim=node_features, 
            ff_dim=ff_dim, 
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(node_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index, edge_map, global_node_features):
        residual = node_feature_matrix
        x_norm = self.norm1(node_feature_matrix)
        
        attn_output, new_edge_features = self.attention(
            x_norm, edge_feature_matrix, edge_index, edge_map, global_node_features
        )
        
        x = residual + self.dropout1(attn_output)

        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        output_node_features = residual + self.dropout2(ffn_output)

        return output_node_features, new_edge_features

class GlobalNodeUpdater(nn.Module):
    def __init__(self, hidden_features):
        super(GlobalNodeUpdater, self).__init__()
        self.gate_linear = nn.Linear(hidden_features * 2, hidden_features)
        self.candidate_linear = nn.Linear(hidden_features * 2, hidden_features)
        self.hidden_features = hidden_features

    def forward(self, global_node, node_features, batch_size):
        num_nodes_per_graph = node_features.shape[0] // batch_size
        
        # Reshape to (batch_size, num_nodes, features) and aggregate
        node_features_reshaped = node_features.view(batch_size, num_nodes_per_graph, self.hidden_features)
        mean_node_features = node_features_reshaped.mean(dim=1) # Shape: (batch_size, hidden_features)

        combined = torch.cat([global_node, mean_node_features], dim=-1)

        update_gate = torch.sigmoid(self.gate_linear(combined))
        candidate_state = torch.tanh(self.candidate_linear(combined))

        new_global_node = (1 - update_gate) * global_node + update_gate * candidate_state
        return new_global_node

class ChessGNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, global_node_in_features, gnn_hidden_features, num_possible_moves, 
                 num_res_layers=10, num_heads=8, ff_dim=512, dropout=0.1):
        super(ChessGNN, self).__init__()
        
        self.input_proj_node = nn.Linear(node_in_features, gnn_hidden_features)
        self.input_proj_edge = nn.Linear(edge_in_features, gnn_hidden_features)
        self.input_proj_global = nn.Linear(global_node_in_features, gnn_hidden_features)

        self.gnn_layers = nn.ModuleList()
        self.global_updaters = nn.ModuleList()
        
        for _ in range(num_res_layers):
            self.gnn_layers.append(GATEAUTransformerBlock(
                node_features=gnn_hidden_features,
                edge_features=gnn_hidden_features,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ))
            self.global_updaters.append(GlobalNodeUpdater(gnn_hidden_features))

        self.final_norm = nn.LayerNorm(gnn_hidden_features)

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

    def forward(self, node_feature_matrix, edge_feature_matrix, global_node_vector, edge_index, edge_map, batch_size):
        x = self.input_proj_node(node_feature_matrix)
        e = self.input_proj_edge(edge_feature_matrix)
        
        current_global_node = self.input_proj_global(global_node_vector)
        
        num_nodes_per_graph = x.shape[0] // batch_size
        
        for i, layer in enumerate(self.gnn_layers):
            expanded_global_node = current_global_node.unsqueeze(1).repeat(1, num_nodes_per_graph, 1).view(batch_size * num_nodes_per_graph, -1)
            
            x, e = layer(x, e, edge_index, edge_map, expanded_global_node)
            
            updater = self.global_updaters[i]
            current_global_node = updater(current_global_node, x, batch_size)

        processed_node_features = self.final_norm(x)
        graph_representation = processed_node_features.view(batch_size, -1)
        policy_logits = self.policy_head(graph_representation)
        value = self.value_head(graph_representation)

        return policy_logits, value
    