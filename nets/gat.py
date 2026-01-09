"""
Pytorch Geometric
Ref: ttps://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gat_conv.py
"""

from typing import Union, Tuple, Optional

from torch_geometric.nn import GATConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_batch

from torch_geometric.nn.inits import reset, glorot, zeros


class StandGATXBN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3, head=8):
        super().__init__()
        self.Conv = nn.Conv1d(nhid, nclass, kernel_size=1)
        self._cached_adj_t = None

        self.layer = nlayer

        num_head = 1
        head_dim = nhid//num_head
        head_nclass = nclass//num_head

        self.conv1 = GATConv(nfeat, head_dim, heads=head)
        self.conv2 = GATConv(nhid, head_dim, heads=head)
        self.convx = nn.ModuleList([GATConv(nhid, head_dim, heads=head) for _ in range(nlayer-2)])
        self.dropout_p = dropout
        self.is_add_self_loops = True

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).t()

        edge_index = self._cached_adj_t
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        if self.layer == 1:
            return self.tran_lin(x)

        x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                x = F.dropout(x, p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index)
                x = self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x,p= self.dropout_p,  training=self.training)
        x= self.conv2(x, edge_index)
        x = self.batch_norm2(x)

        return self.tran_lin(x)

    def tran_lin(self, x):
        x = x.unsqueeze(0)  # Qin Jun22
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

