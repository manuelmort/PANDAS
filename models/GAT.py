# models/GAT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    def __init__(self, n_class, n_features=768, hidden_dim=64, heads=4, dropout=0.1, class_weights=None):
        super(GATClassifier, self).__init__()
        
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        
        # GAT layers
        self.gat1 = GATConv(n_features, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        self.classifier = nn.Linear(hidden_dim, n_class)
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_feat, labels, adj, mask, return_attention=False):
        """
        Args:
            node_feat: (batch, num_nodes, n_features)
            labels: (batch,)
            adj: (batch, num_nodes, num_nodes) - adjacency matrix
            mask: (batch, num_nodes) - node mask
            return_attention: bool - whether to return attention weights
        """
        batch_size = node_feat.size(0)
        
        all_preds = []
        all_attns = []
        total_loss = 0
        
        # Process each graph in batch separately (GAT expects single graph)
        for b in range(batch_size):
            # Get single graph
            x = node_feat[b]  # (num_nodes, n_features)
            a = adj[b]        # (num_nodes, num_nodes)
            m = mask[b] if mask is not None else None  # (num_nodes,)
            
            # Apply mask to remove padded nodes
            if m is not None:
                valid_nodes = m.bool()
                x = x[valid_nodes]
                a = a[valid_nodes][:, valid_nodes]
            
            # Convert dense adjacency to edge_index
            edge_index = a.nonzero(as_tuple=False).t().contiguous()
            
            # Skip if no edges
            if edge_index.size(1) == 0:
                continue
            
            # GAT forward pass
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = self.dropout(x)
            
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = self.dropout(x)
            
            x, attn3 = self.gat3(x, edge_index, return_attention_weights=True)
            
            # Global mean pooling
            graph_embed = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            
            # Classification
            out = self.classifier(graph_embed)  # (1, n_class)
            
            all_preds.append(out)
            
            if return_attention:
                all_attns.append((attn1, attn2, attn3))
        
        # Stack predictions
        if len(all_preds) == 0:
            # Handle edge case
            out = torch.zeros(batch_size, self.n_class, device=node_feat.device)
        else:
            out = torch.cat(all_preds, dim=0)  # (batch, n_class)
        
        # Compute loss
        loss = self.criterion(out, labels)
        
        # Get predictions
        pred = out.argmax(dim=1)
        
        if return_attention:
            return pred, labels, loss, all_attns
        
        return pred, labels, loss
    
    def get_attention_weights(self, node_feat, adj, mask):
        """Extract attention weights for visualization"""
        self.eval()
        with torch.no_grad():
            x = node_feat.squeeze(0)
            a = adj.squeeze(0)
            m = mask.squeeze(0) if mask is not None else None
            
            if m is not None:
                valid_nodes = m.bool()
                x = x[valid_nodes]
                a = a[valid_nodes][:, valid_nodes]
            
            edge_index = a.nonzero(as_tuple=False).t().contiguous()
            
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            
            x, attn3 = self.gat3(x, edge_index, return_attention_weights=True)
            
        return {
            'layer1': attn1,
            'layer2': attn2,
            'layer3': attn3,
            'edge_index': edge_index
        }