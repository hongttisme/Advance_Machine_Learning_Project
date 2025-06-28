
import torch
import torch.nn as nn
import torch.nn.functional as F



class GATEAULayer(nn.Module):

    def __init__(self, node_in_features, edge_in_features, node_out_features):

        super(GATEAULayer, self).__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.node_out_features = node_out_features

  
        self.Wv = nn.Parameter(torch.randn(node_in_features, edge_in_features))
        self.Wu = nn.Parameter(torch.randn(node_in_features, edge_in_features))
        self.We = nn.Parameter(torch.randn(edge_in_features, edge_in_features))
        
        self.Wh = nn.Parameter(torch.randn(node_in_features, node_out_features))
        self.Wg = nn.Parameter(torch.randn(edge_in_features, node_out_features))
        self.W0 = nn.Parameter(torch.randn(node_in_features, node_out_features))

        self.a = nn.Parameter(torch.randn(edge_in_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wu)
        nn.init.xavier_uniform_(self.We)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.xavier_uniform_(self.Wg)
        nn.init.xavier_uniform_(self.W0)
        nn.init.zeros_(self.a)


    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index):

        num_nodes = node_feature_matrix.shape[0]
        target_node_idx, source_node_idx = edge_index[0], edge_index[1]

        h_nodes_v = node_feature_matrix @ self.Wv
        h_nodes_u = node_feature_matrix @ self.Wu
        h_nodes_0 = node_feature_matrix @ self.W0
        h_nodes_h = node_feature_matrix @ self.Wh
        h_edges_e = edge_feature_matrix @ self.We
        h_edges_g = edge_feature_matrix @ self.Wg
        
        target_node_feats_for_attention = h_nodes_u[target_node_idx]
        source_node_feats_for_attention = h_nodes_v[source_node_idx]
        edge_feats_for_attention = h_edges_e


        g_prime = target_node_feats_for_attention + source_node_feats_for_attention + edge_feats_for_attention
        
        attention_scores = self.leaky_relu(g_prime @ self.a)
        max_scores = torch.full((num_nodes,), -1e9, device=attention_scores.device, dtype=attention_scores.dtype)
        max_scores.scatter_reduce_(0, target_node_idx, attention_scores, reduce="amax", include_self=False)
        
        scores_max_per_edge = max_scores[target_node_idx]
        attention_scores_exp = torch.exp(attention_scores - scores_max_per_edge)

        sum_exp_scores = torch.zeros(num_nodes, device=attention_scores.device, dtype=attention_scores.dtype)
        sum_exp_scores.index_add_(0, target_node_idx, attention_scores_exp)
        
        sum_exp_per_edge = sum_exp_scores[target_node_idx]

        alpha = attention_scores_exp / (sum_exp_per_edge + 1e-10)

        source_node_values = h_nodes_h[source_node_idx]
        edge_values = h_edges_g
        values = source_node_values + edge_values
        
        weighted_values = values * alpha.unsqueeze(-1)

        aggregated_messages = torch.zeros_like(h_nodes_0)
        aggregated_messages.index_add_(0, target_node_idx, weighted_values)

        new_final = h_nodes_0 + aggregated_messages
        
        return new_final, g_prime
    




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

    def __init__(self, node_in_features, edge_in_features, node_out_features):
        super(ResGATEAU, self).__init__()

        self.bnr1 = BNR(node_in_features)
        self.gateau1 = GATEAULayer(node_in_features, edge_in_features, node_out_features)
        

        self.bnr2 = BNR(node_out_features)
        self.gateau2 = GATEAULayer(node_out_features, edge_in_features, node_out_features)
        self.bnr3 = BNR(edge_in_features)
        self.bnr4 = BNR(edge_in_features)


        if node_in_features != node_out_features:
            self.residual_transform = nn.Linear(node_in_features, node_out_features)
        else:
            self.residual_transform = nn.Identity()

    def forward(self, node_feature_matrix, edge_feature_matrix, edge_index):
        node_residual = self.residual_transform(node_feature_matrix)
        edge_residual = edge_feature_matrix 

        x = self.bnr1(node_feature_matrix)
        e = self.bnr3(edge_feature_matrix) 
        x, e = self.gateau1(x, e, edge_index) 
        
        x = self.bnr2(x)
        e = self.bnr4(e) 
        x, e = self.gateau2(x, e, edge_index) 

        output_node_features = node_residual + x
        output_edge_features = edge_residual + e 
        
        return output_node_features, output_edge_features

class ChessGNN(nn.Module):

    def __init__(self, node_in_features, edge_in_features, gnn_hidden_features, num_possible_moves, num_res_layers=10):
        super(ChessGNN, self).__init__()
        

        self.gnn_layers = nn.ModuleList()

        self.gnn_layers.append(ResGATEAU(
            node_in_features=node_in_features,
            edge_in_features=edge_in_features,
            node_out_features=gnn_hidden_features
        ))


        for _ in range(num_res_layers - 1):
            self.gnn_layers.append(ResGATEAU(
                node_in_features=gnn_hidden_features,
                edge_in_features=edge_in_features,
                node_out_features=gnn_hidden_features  
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

        x_node = node_feature_matrix
        x_edge = edge_feature_matrix[edge_map]
        for layer in self.gnn_layers:

            x_node, x_edge = layer(x_node, x_edge, edge_index)

        processed_node_features = x_node
        

        graph_representation = processed_node_features.view(batch_size, -1)

        policy_logits = self.policy_head(graph_representation)

        value = self.value_head(graph_representation)

        return policy_logits, value