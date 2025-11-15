import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layer.gps_layer import GPSLayer



class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, args, dim_in, dim_out):
        super().__init__()

        self.pre_mp = torch.nn.Linear(dim_in, args.hid_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.layer):
            self.layers.append(GPSLayer(
                dim_h=args.hid_dim,
                local_gnn_type=args.local_gnn_type,
                num_heads=args.heads,
                dropout = args.dropout
            ))

        self.post_mp = torch.nn.Linear(args.hid_dim, dim_out)
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.pre_mp.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, edge_index):
        if self.pre_mp is not None:
            x = self.pre_mp(x)
        for layer in self.layers:
            x, edge_index = layer(x, edge_index)
        
        x = self.post_mp(x)
        
        return x
