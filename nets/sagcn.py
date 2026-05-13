from typing import Optional

import torch
import torch_geometric
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.nn.dense.linear import Linear

from nets.SAGCN2 import SAGCN2


def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


import torch
import torch.nn.functional as F
from typing import Optional
from torch_sparse import SparseTensor
import torch.nn as nn


class SAGCNXBN(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        is_add_self_loops = args.add_selfloop
        self.is_add_self_loops = is_add_self_loops
        self.BN_model = args.BN_model
        norm = args.gcn_norm
        nhid = args.hid_dim
        dropout = args.dropout
        nlayer = args.layer

        GCNConv = SAGCN2
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.layer = nlayer

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index, edge_weight)
        if self.layer == 1:
            return x
        if self.BN_model:
            x = self.batch_norm1(x)
        x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index, edge_weight)
                if self.BN_model:
                    x= self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if self.BN_model:
            x = self.batch_norm2(x)
        return x