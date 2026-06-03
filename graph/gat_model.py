import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for learning structural priors
    from medical image segmentation graphs
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=32, num_heads=4):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output (structural latent code) dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph Attention Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        
        self.relu = nn.ReLU()
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object with x (node features) and edge_index
            
        Returns:
            structural_code: Structural latent representation (B, output_dim)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        # Global pooling (aggregate node features into graph-level representation)
        if batch is None:
            struct_code = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        else:
            struct_code = global_mean_pool(x, batch)
        
        return struct_code


class StructuralLatentEncoder(nn.Module):
    """
    Encodes structural information from graphs
    """
    
    def __init__(self, input_dim=5, output_dim=32):
        super().__init__()
        self.gat = GraphAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_heads=4
        )
    
    def forward(self, graph_batch):
        """Encode graph to structural code"""
        return self.gat(graph_batch)


# Test
if __name__ == "__main__":
    from torch_geometric.data import DataLoader, Batch
    
    # Create dummy graphs
    graphs = []
    for _ in range(4):
        x = torch.randn(5, 5)  # 5 nodes, 5 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        data = torch.geometric.data.Data(x=x, edge_index=edge_index)
        graphs.append(data)
    
    # Create batch
    batch = Batch.from_data_list(graphs)
    
    # Model
    model = GraphAttentionNetwork(input_dim=5, output_dim=32)
    output = model(batch)
    
    print(f"Output shape: {output.shape}")  # Should be (4, 32)