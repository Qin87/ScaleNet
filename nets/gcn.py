"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gcn_conv.py
"""
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy
import numpy as np

from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import reset, glorot, zeros

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[col]

class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.temp_weight = torch.nn.Linear(in_channels, out_channels, bias=False)
        # bias false.
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.temp_weight.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, is_add_self_loops: bool = True) -> Tensor:
        original_size = edge_index.shape[1]

        x = self.temp_weight(x)

        if self.normalize:        # without this, telegram is better!
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out, edge_index

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class StandGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(StandGCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass, cached=False, normalize=True)
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()
        self.is_add_self_loops = True

    def forward(self, x, adj, edge_weight=None):

        edge_index = adj
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)

        return x


class StandGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(StandGCN2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=True)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=True)
        self.dropout_p = dropout

        self.is_add_self_loops = True

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, edge_index, edge_weight=None):
        # # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.conv1(x, edge_index, edge_weight))  # no BN here is better
        # # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        # # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        #
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = F.relu(x)

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x, edge_index = self.conv2(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)

        return x


class StandGCNX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(StandGCNX, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=True)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=True)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.is_add_self_loops = True
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = F.relu(x)

        for iter_layer in self.convx:
            x = F.dropout(x,p= self.dropout_p, training=self.training)
            x, edge_index = iter_layer(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            x = F.relu(x)

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x, edge_index = self.conv2(x, edge_index, edge_weight,is_add_self_loops=self.is_add_self_loops)
        return x

class StandGCN1BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1, norm=True):
        super(StandGCN1BN, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass, cached=False, normalize=norm)
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()
        self.is_add_self_loops = True
        self.batch_norm1 = nn.BatchNorm1d(nclass)
        self.dropout_p = dropout

    def forward(self, x, adj, edge_weight=None):

        edge_index = adj
        x = F.dropout(x, p=self.dropout_p, training=self.training)  # the best arrangement of dropout and BN
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = self.batch_norm1(x)
        # x = F.dropout(x, p=self.dropout_p, training=self.training)  # the best arrangement of dropout and BN

        return x


class StandGCN2BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2, norm=True):
        super(StandGCN2BN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm)
        self.conv2 = GCNConv(nhid, nhid, cached=False, normalize=norm)
        self.dropout_p = dropout
        self.Conv = nn.Conv1d(nhid, nclass, kernel_size=1)

        self.is_add_self_loops = True
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = F.relu(x)
        # x = self.batch_norm1(x)     # Qin add May23

        x, edge_index = self.conv2(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)      # best arrange for dropout and BN

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x


class StandGCNXBN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super(StandGCNXBN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.is_add_self_loops = True  # Qin TODO True is the original
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = self.batch_norm1(x)
        x = F.relu(x)

        for iter_layer in self.convx:
            x = F.dropout(x,p= self.dropout_p, training=self.training)
            x, edge_index = iter_layer(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            x= self.batch_norm3(x)
            x = F.relu(x)

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x, edge_index = self.conv2(x, edge_index, edge_weight,is_add_self_loops=self.is_add_self_loops)
        x = self.batch_norm2(x)
        # x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x


def create_gcn(nfeat, nhid, nclass, dropout, nlayer, norm=True):
    if nlayer == 1:
        model = StandGCN1BN(nfeat, nhid, nclass, dropout,nlayer, norm)
    elif nlayer == 2:
        model = StandGCN2BN(nfeat, nhid, nclass, dropout,nlayer,norm)
    else:
        model = StandGCNXBN(nfeat, nhid, nclass, dropout,nlayer,norm)

    return model
