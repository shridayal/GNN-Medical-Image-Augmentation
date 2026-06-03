import numpy as np
import networkx as nx
from scipy import ndimage
from skimage import measure
import torch
from torch_geometric.data import Data

class GraphBuilder:
    """
    Convert medical image segmentation masks to graph representation
    """
    
    def __init__(self):
        pass
    
    def mask_to_graph(self, mask, use_weights=True):
        """
        Convert binary/multi-class segmentation mask to graph
        
        Args:
            mask: 2D numpy array (H, W) with integer labels
            use_weights: Whether to weight edges by distance
            
        Returns:
            PyTorch Geometric Data object (graph)
        """
        
        # Label connected components
        labeled_array, num_features = ndimage.label(mask > 0)
        
        if num_features == 0:
            print("Warning: Empty mask!")
            return None
        
        # Create nodes (one per connected region)
        nodes_data = []
        region_labels = []
        
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id)
            
            # Compute node features
            area = region_mask.sum()
            centroid = ndimage.center_of_mass(region_mask)
            
            # Get region properties
            props = measure.regionprops(region_mask.astype(int))[0]
            
            node_features = [
                area,                          # Area
                centroid[0],                   # Centroid Y
                centroid[1],                   # Centroid X
                props.eccentricity,            # Eccentricity
                props.solidity,                # Solidity
            ]
            
            nodes_data.append(node_features)
            region_labels.append(region_id)
        
        # Create edges (adjacency between regions)
        edges = []
        edge_weights = []
        
        for i in range(num_features):
            for j in range(i+1, num_features):
                region_i = (labeled_array == region_labels[i])
                region_j = (labeled_array == region_labels[j])
                
                # Check if regions are adjacent (touching)
                # Dilate region_i and check overlap with region_j
                from scipy.ndimage import binary_dilation
                dilated_i = binary_dilation(region_i)
                
                if np.any(dilated_i & region_j):
                    # Regions are adjacent
                    cent_i = np.array(ndimage.center_of_mass(region_i))
                    cent_j = np.array(ndimage.center_of_mass(region_j))
                    
                    distance = np.linalg.norm(cent_i - cent_j)
                    
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirectional
                    
                    weight = 1.0 / (distance + 1e-5) if use_weights else 1.0
                    edge_weights.append(weight)
                    edge_weights.append(weight)
        
        # Convert to tensors
        node_features = torch.tensor(nodes_data, dtype=torch.float32)
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            # No edges - create self-loops
            edge_index = torch.arange(len(nodes_data), dtype=torch.long)
            edge_index = torch.stack([edge_index, edge_index])
            edge_weight = torch.ones(len(nodes_data))
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weight,
            mask=torch.tensor(mask, dtype=torch.float32)
        )
        
        return graph
    
    def batch_mask_to_graph(self, masks):
        """
        Convert batch of masks to graphs
        
        Args:
            masks: List of masks or tensor (B, H, W)
            
        Returns:
            List of graph Data objects
        """
        graphs = []
        for mask in masks:
            graph = self.mask_to_graph(mask.numpy() if isinstance(mask, torch.Tensor) else mask)
            if graph is not None:
                graphs.append(graph)
        return graphs


# Test
if __name__ == "__main__":
    # Create dummy mask
    mask = np.zeros((256, 256))
    mask[50:100, 50:100] = 1  # Region 1
    mask[150:200, 150:200] = 2  # Region 2
    
    builder = GraphBuilder()
    graph = builder.mask_to_graph(mask)
    
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    print(f"Node features shape: {graph.x.shape}")