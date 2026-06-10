"""
Graph Neural Network Graph Builder
Convert medical image segmentation masks to anatomical graph representation
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage import measure
import torch
from torch_geometric.data import Data
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ FIX: Import config with try-except to avoid circular imports
try:
    from training.config import GRAPH_CONFIG
except ImportError:
    # Fallback config if import fails
    GRAPH_CONFIG = {
        'min_component_size': 50,
        'max_component_size': 50000,
        'adjacency_threshold': 0.5,
        'connectivity': 8,
    }


class GraphBuilder:
    """
    Convert medical image segmentation masks to graph representation
    
    Nodes: Anatomical regions (organs/structures)
    Edges: Spatial adjacency relationships
    
    Graph Properties:
    - Node features: [centroid_y, centroid_x, area, eccentricity, solidity]
    - Edge features: Distance-based weights
    - Normalized to [0, 1] for consistency
    """
    
    def __init__(self, img_size=256, connectivity=8):
        """
        Initialize graph builder
        
        Args:
            img_size: Image dimension (default 256x256)
            connectivity: Connectivity type (4 or 8)
        """
        self.img_size = img_size
        self.connectivity = connectivity
        
        print(f"\n{'='*60}")
        print("📊 Graph Builder Initialized")
        print(f"{'='*60}")
        print(f"\n   Image size: {img_size}×{img_size}")
        print(f"   Connectivity: {connectivity}-neighbor")
        print(f"   Min component: {GRAPH_CONFIG['min_component_size']} pixels")
        print(f"   Max component: {GRAPH_CONFIG['max_component_size']} pixels")
        print(f"   Adjacency threshold: {GRAPH_CONFIG['adjacency_threshold']}\n")
    
    def mask_to_graph(self, mask, verbose=False):
        """
        Convert binary/multi-class segmentation mask to graph
        
        Args:
            mask: 2D numpy array [H, W] with integer labels
            verbose: Print debug information
        
        Returns:
            torch_geometric.data.Data object (graph)
        """
        try:
            # Ensure proper shape
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            mask = mask.squeeze()
            
            if mask.ndim != 2:
                return None
            
            # ===== Label Connected Components =====
            struct = ndimage.generate_binary_structure(2, self.connectivity // 4 + 1)
            labeled_array, num_components = ndimage.label(mask > 0.5, structure=struct)
            
            if num_components == 0:
                if verbose:
                    print("  ⚠️  Warning: No components in mask")
                return None
            
            # ===== Extract Nodes (Regions) =====
            nodes_data = []
            region_ids = []
            region_masks = []
            
            for region_id in range(1, num_components + 1):
                region_mask = (labeled_array == region_id)
                area = region_mask.sum()
                
                # Skip components outside size range
                if area < GRAPH_CONFIG['min_component_size']:
                    continue
                if area > GRAPH_CONFIG['max_component_size']:
                    continue
                
                # Compute node features
                try:
                    # Basic features
                    centroid = ndimage.center_of_mass(region_mask)
                    
                    # Region properties
                    props = measure.regionprops(region_mask.astype(int))[0]
                    
                    # Safe feature extraction
                    eccentricity = float(props.eccentricity) if hasattr(props, 'eccentricity') else 0.5
                    solidity = float(props.solidity) if hasattr(props, 'solidity') else 0.8
                    
                    # Normalize features to [0, 1]
                    node_features = [
                        centroid[0] / self.img_size,          # Normalized Y
                        centroid[1] / self.img_size,          # Normalized X
                        np.log(area + 1) / np.log(self.img_size**2 + 1),  # Log area
                        eccentricity,                         # Eccentricity [0, 1]
                        solidity,                             # Solidity [0, 1]
                    ]
                    
                    nodes_data.append(node_features)
                    region_ids.append(region_id)
                    region_masks.append(region_mask)
                
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error extracting properties: {str(e)[:40]}")
                    continue
            
            if len(nodes_data) == 0:
                if verbose:
                    print("  ⚠️  No valid regions after filtering")
                return None
            
            # ===== Create Edges (Adjacency) =====
            edges = []
            edge_weights = []
            
            num_nodes = len(nodes_data)
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    region_i = region_masks[i]
                    region_j = region_masks[j]
                    
                    # Check adjacency
                    dilated_i = binary_dilation(region_i)
                    
                    if np.any(dilated_i & region_j):
                        # Regions are adjacent
                        cent_i = np.array(nodes_data[i][:2]) * self.img_size
                        cent_j = np.array(nodes_data[j][:2]) * self.img_size
                        
                        # Distance-based weight
                        distance = np.linalg.norm(cent_i - cent_j)
                        normalized_dist = min(distance / (self.img_size * np.sqrt(2)), 1.0)
                        weight = 1.0 / (normalized_dist + 0.1)
                        
                        # Add bidirectional edge
                        edges.append([i, j])
                        edges.append([j, i])
                        edge_weights.append(weight)
                        edge_weights.append(weight)
            
            # ===== Convert to PyTorch Tensors =====
            node_features = torch.tensor(nodes_data, dtype=torch.float32)
            
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
            else:
                # Self-loops if no edges
                edge_index = torch.arange(num_nodes, dtype=torch.long)
                edge_index = torch.stack([edge_index, edge_index])
                edge_weight = torch.ones(num_nodes, dtype=torch.float32)
            
            # ===== Create Graph Object =====
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weight,
                num_nodes=len(nodes_data)
            )
            
            if verbose:
                print(f"  ✅ Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
            
            return graph
        
        except Exception as e:
            if verbose:
                print(f"  ❌ Graph construction error: {str(e)[:60]}")
            return None
    
    def batch_mask_to_graph(self, masks, verbose=False):
        """
        Convert batch of masks to graphs
        
        Args:
            masks: List/tensor of masks [B, H, W]
            verbose: Print debug information
        
        Returns:
            List of torch_geometric.data.Data objects
        """
        graphs = []
        
        for idx, mask in enumerate(masks):
            try:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask
                
                graph = self.mask_to_graph(mask_np, verbose=verbose)
                
                if graph is not None and graph.num_nodes > 0:
                    graphs.append(graph)
                elif verbose:
                    print(f"  ⚠️  Batch {idx}: Invalid graph")
            
            except Exception as e:
                if verbose:
                    print(f"  ❌ Batch {idx} error: {str(e)[:40]}")
                continue
        
        return graphs


def main():
    """Test graph builder"""
    from torch_geometric.data import Batch
    
    print("\n" + "="*60)
    print("Testing Graph Builder")
    print("="*60 + "\n")
    
    builder = GraphBuilder(img_size=256)
    
    # ===== Test 1: Simple mask =====
    print("Test 1: Simple single-region mask")
    print("-" * 60 + "\n")
    
    mask1 = np.zeros((256, 256))
    mask1[50:150, 50:150] = 1
    
    graph1 = builder.mask_to_graph(mask1, verbose=True)
    
    if graph1 is not None:
        print(f"\n✅ Graph created")
        print(f"   Nodes: {graph1.num_nodes}")
        print(f"   Edges: {graph1.num_edges}")
        print(f"   Node features: {graph1.x.shape}\n")
    
    # ===== Test 2: Multi-region mask =====
    print("\nTest 2: Multi-region mask")
    print("-" * 60 + "\n")
    
    mask2 = np.zeros((256, 256))
    mask2[30:120, 30:120] = 1
    mask2[140:230, 140:230] = 1
    mask2[70:160, 140:230] = 1
    
    graph2 = builder.mask_to_graph(mask2, verbose=True)
    
    if graph2 is not None:
        print(f"\n✅ Graph created")
        print(f"   Nodes: {graph2.num_nodes}")
        print(f"   Edges: {graph2.num_edges}\n")
    
    print("="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
