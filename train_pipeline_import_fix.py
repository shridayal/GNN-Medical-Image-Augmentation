"""Quick fix for circular imports"""

# Instead of:
# from data.data_loader import get_dataloader
# from graph.graph_builder import GraphBuilder
# from training.train_gnn_vae import TrainerGNNVAE

# Use lazy imports:
def get_dataloader(*args, **kwargs):
    from data.data_loader import get_dataloader as _get_dataloader
    return _get_dataloader(*args, **kwargs)

def GraphBuilder(*args, **kwargs):
    from graph.graph_builder import GraphBuilder as _GraphBuilder
    return _GraphBuilder(*args, **kwargs)

def TrainerGNNVAE(*args, **kwargs):
    from training.train_gnn_vae import TrainerGNNVAE as _TrainerGNNVAE
    return _TrainerGNNVAE(*args, **kwargs)

