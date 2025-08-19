"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/sage_conv.py
"""


from torch_scatter import scatter_add
import torch


def inci_norm(edge_index, edge_weight):
    """
    Normalize edge weights using GCN-style normalization
    """

    num_nodes = edge_index.max().item() + 1
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]

    # Compute node degrees
    # deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    # Compute D^(-1/2)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Normalize edge weights
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


