"""
Graph Attention Network (GAT) for Medical Image Structure Encoding
Learns anatomical relationships from segmentation graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import MODEL_CONFIG


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for learning structural priors
    from medical image segmentation graphs
    
    Architecture:
    - Input: Node features [N, input_dim] + Edge indices [2, E]
    - GAT Layer 1: Multi-head attention
    - GAT Layer 2: Single-head attention
    - Global pooling: Aggregate to graph-level
    - Output: Structural latent code [output_dim]
    
    Args:
        input_dim: Input node feature dimension (5)
        hidden_dim: Hidden layer dimension (64)
        output_dim: Output (structural code) dimension (32)
        num_heads: Number of attention heads (8)
        num_layers: Number of GAT layers (2)
        dropout: Dropout rate (0.2)
        pooling: Pooling method ('mean' or 'add')
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=32, 
                 num_heads=8, num_layers=2, dropout=0.2, pooling='mean'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        
        print(f"\n{'='*60}")
        print("🧠 Graph Attention Network (GAT)")
        print(f"{'='*60}")
        print(f"\n   Architecture:")
        print(f"      Input dim:  {input_dim}")
        print(f"      Hidden dim: {hidden_dim}")
        print(f"      Output dim: {output_dim}")
        print(f"      Num heads:  {num_heads}")
        print(f"      Num layers: {num_layers}")
        print(f"      Dropout:    {dropout}")
        print(f"      Pooling:    {pooling}\n")
        
        # ===== GAT Layers =====
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer: input_dim → hidden_dim * num_heads
        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                concat=True,  # Concatenate head outputs
                dropout=dropout,
                add_self_loops=True
            )
        )
        
        # Middle GAT layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True
                )
            )
        
        # Final GAT layer: hidden_dim * num_heads → output_dim (single head)
        self.gat_layers.append(
            GATConv(
                hidden_dim * num_heads,
                output_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                add_self_loops=True
            )
        )
        
        # ===== Activation & Normalization =====
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Batch normalization for intermediate layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads) 
            for _ in range(num_layers - 1)
        ])
        
        # ===== Pooling =====
        self.pooling_fn = global_mean_pool if pooling == 'mean' else global_add_pool
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ GAT initialized\n")
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, GATConv) or isinstance(module, GATv2Conv):
                # GAT weights initialized in the module itself
                pass
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        """
        Forward pass: Process graph and return structural code
        
        Args:
            data: PyTorch Geometric Data/Batch object
                  - x: Node features [N, input_dim]
                  - edge_index: Edge connectivity [2, E]
                  - batch: Graph indices for batches (optional) [N]
        
        Returns:
            struct_code: Structural latent code
                        - Single graph: [output_dim]
                        - Batch: [batch_size, output_dim]
        """
        x = data.x
        edge_index = data.edge_index
        
        # Get batch indices if available (for batched graphs)
        batch_idx = data.batch if hasattr(data, 'batch') else None
        
        # ===== GAT Layers =====
        for layer_idx, gat_layer in enumerate(self.gat_layers[:-1]):
            # Forward through GAT
            x = gat_layer(x, edge_index)
            
            # Batch normalization
            if layer_idx < len(self.bn_layers):
                x = self.bn_layers[layer_idx](x)
            
            # Activation and dropout
            x = self.relu(x)
            x = self.dropout(x)
        
        # Final GAT layer (no batch norm or dropout after)
        x = self.gat_layers[-1](x, edge_index)
        
        # ===== Global Pooling =====
        # Aggregate node-level features to graph-level
        if batch_idx is None:
            # Single graph: create batch tensor of zeros
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        struct_code = self.pooling_fn(x, batch_idx)
        
        return struct_code


class StructuralLatentEncoder(nn.Module):
    """
    Wrapper for GAT: Encodes anatomical structure from graphs
    
    Provides:
    - Clean interface for graph encoding
    - Consistent parameter handling
    - Support for single graphs and batches
    """
    
    def __init__(self, input_dim=5, output_dim=32, config=None):
        """
        Args:
            input_dim: Node feature dimension
            output_dim: Structural code dimension
            config: Config dict with GAT parameters (optional)
        """
        super().__init__()
        
        if config is None:
            config = MODEL_CONFIG['gat']
        
        self.gat = GraphAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=output_dim,
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 2),
            dropout=0.2,
            pooling='mean'
        )
    
    def forward(self, graph_batch):
        """
        Encode graph batch to structural codes
        
        Args:
            graph_batch: PyTorch Geometric Batch
        
        Returns:
            struct_codes: [batch_size, output_dim]
        """
        return self.gat(graph_batch)
    
    def encode_single(self, graph):
        """
        Encode single graph
        
        Args:
            graph: PyTorch Geometric Data
        
        Returns:
            struct_code: [output_dim]
        """
        with torch.no_grad():
            # Wrap in batch
            batch = Batch.from_data_list([graph])
            codes = self.gat(batch)
            return codes[0]


def main():
    """Test GAT model"""
    import numpy as np
    from torch_geometric.data import DataLoader
    
    print("\n" + "="*60)
    print("Testing Graph Attention Network (GAT)")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # ===== Test 1: Single Graph =====
    print("="*60)
    print("Test 1: Single Graph Processing")
    print("="*60 + "\n")
    
    # Create a simple graph
    x = torch.randn(5, 5, device=device)  # 5 nodes, 5 features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ], dtype=torch.long, device=device)
    
    graph = Data(x=x, edge_index=edge_index)
    
    model = GraphAttentionNetwork(input_dim=5, output_dim=32).to(device)
    
    print(f"Input: {x.shape}")
    print(f"Edges: {edge_index.shape}")
    
    output = model(graph)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✅ Test passed\n")
    
    # ===== Test 2: Batch Processing =====
    print("="*60)
    print("Test 2: Batch Processing")
    print("="*60 + "\n")
    
    # Create batch of graphs
    graphs = []
    for i in range(4):
        num_nodes = np.random.randint(3, 8)
        x = torch.randn(num_nodes, 5, device=device)
        
        # Random edges
        if num_nodes > 1:
            edge_list = []
            for j in range(num_nodes):
                targets = np.random.choice(num_nodes, size=min(2, num_nodes), replace=False)
                for t in targets:
                    if j != t:
                        edge_list.append([j, t])
            
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        graphs.append(Data(x=x, edge_index=edge_index))
    
    # Batch graphs
    batch = Batch.from_data_list(graphs)
    
    print(f"Batch info:")
    print(f"  Graphs: {len(graphs)}")
    print(f"  Total nodes: {batch.num_nodes}")
    print(f"  Total edges: {batch.num_edges}")
    print(f"  Batch tensor: {batch.batch.shape}")
    
    output = model(batch)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✅ Test passed\n")
    
    # ===== Test 3: StructuralLatentEncoder =====
    print("="*60)
    print("Test 3: StructuralLatentEncoder Wrapper")
    print("="*60 + "\n")
    
    encoder = StructuralLatentEncoder(
        input_dim=5,
        output_dim=32,
        config=MODEL_CONFIG['gat']
    ).to(device)
    
    output = encoder(batch)
    print(f"Encoder output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✅ Test passed\n")
    
    # ===== Test 4: Different Input Dimensions =====
    print("="*60)
    print("Test 4: Different Input Dimensions")
    print("="*60 + "\n")
    
    for input_dim in [3, 5, 10]:
        x_test = torch.randn(5, input_dim, device=device)
        graph_test = Data(x=x_test, edge_index=edge_index)
        
        model_test = GraphAttentionNetwork(input_dim=input_dim, output_dim=32).to(device)
        output_test = model_test(graph_test)
        
        print(f"Input dim {input_dim}: Output shape {output_test.shape}")
    
    print("✅ Test passed\n")
    
    # ===== Test 5: Attention Visualization =====
    print("="*60)
    print("Test 5: Feature Analysis")
    print("="*60 + "\n")
    
    # Analyze output statistics
    batch = Batch.from_data_list(graphs)
    output = model(batch)
    
    print(f"Output statistics:")
    print(f"  Mean:   {output.mean():.4f}")
    print(f"  Std:    {output.std():.4f}")
    print(f"  Min:    {output.min():.4f}")
    print(f"  Max:    {output.max():.4f}")
    print("✅ Test passed\n")
    
    # ===== Count Parameters =====
    print("="*60)
    print("Model Statistics")
    print("="*60 + "\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:          {total_params * 4 / 1e6:.2f} MB")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
