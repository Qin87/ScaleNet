"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gcn_conv.py
"""
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
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

from nets.SAGCN2 import SAGCN2
from nets.sagcn import SAGCN
from nets.sage import SAGEConv_SHA
from nets.src2.sage_qin import SAGEConv_QinNov


def gcn_norm0(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=0, dtype=None):

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

def norm0(edge_index, edge_weight=None, num_nodes=None, improved=False,
                  add_self_loops=0, norm='dir'):

    if norm == 'sym':
        # row normalization
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif norm == 'dir':
        # type 1: conside different inci-norm
        row, col = edge_index
        deg_row = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_col = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

        row_deg_inv_sqrt = deg_row.pow(-0.5)
        row_deg_inv_sqrt[row_deg_inv_sqrt == float('inf')] = 0

        col_deg_inv_sqrt = deg_col.pow(-0.5)
        col_deg_inv_sqrt[col_deg_inv_sqrt == float('inf')] = 0

        edge_weight = row_deg_inv_sqrt[row] * edge_weight * col_deg_inv_sqrt[col]
    return edge_index, edge_weight

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=0, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops == 1:
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

        if add_self_loops == 1:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[col]

class GCNConv_SHA(MessagePassing):
    r"""GraphSHA
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

        kwargs.setdefault('aggr', 1)
        super(GCNConv_SHA, self).__init__(**kwargs)

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

        # if self.normalize:        # without this, telegram is better!
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                if self.normalize:
                    edge_index, edge_weight = gcn_norm0(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                if self.normalize:
                    edge_index = gcn_norm0(  # yapf: disable
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

        return  out, edge_index

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
        self.conv1 = GCNConv_SHA(nfeat, nclass, cached=False, normalize=True)
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
        self.conv1 = GCNConv_SHA(nfeat, nhid, cached= False, normalize=True)
        self.conv2 = GCNConv_SHA(nhid, nclass, cached=False, normalize=True)
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
        self.conv1 = GCNConv_SHA(nfeat, nhid, cached= False, normalize=True)
        self.conv2 = GCNConv_SHA(nhid, nclass, cached=False, normalize=True)
        self.convx = nn.ModuleList([GCNConv_SHA(nhid, nhid) for _ in range(nlayer-2)])
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
        super().__init__()
        self.conv1 = GCNConv_SHA(nfeat, nclass, cached=False, normalize=norm)
        self.mlp1 = torch.nn.Linear(nclass, nclass)
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()
        self.is_add_self_loops = True
        self.batch_norm1 = nn.BatchNorm1d(nclass)
        self.dropout_p = dropout

    def forward(self, x, adj, edge_weight=None):

        edge_index = adj
        x = F.dropout(x, p=self.dropout_p, training=self.training)  # the best arrangement of dropout and BN
        x, edge_index = self.conv1(x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        # x= self.mlp1(x)
        x = self.batch_norm1(x)
        # x = F.dropout(x, p=self.dropout_p, training=self.training)  # the best arrangement of dropout and BN

        return x


class StandGCN2BN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2, norm=True):
        super().__init__()
        self.conv1 = GCNConv_SHA(nfeat, nhid, cached= False, normalize=norm)
        self.conv2 = GCNConv_SHA(nhid, nhid, cached=False, normalize=norm)
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

from torch_geometric.nn.dense.linear import Linear
class StandGCNXBN_Ak(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        nhid = args.feat_dim
        dropout = args.dropout
        nlayer = args.Ak        # Differ
        is_add_self_loops = args.First_self_loop
        norm = args.gcn_norm
        self.is_add_self_loops = is_add_self_loops  # Qin True is the original
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.conv2 = Linear(nhid, nclass, bias=True, weight_initializer='glorot')

        self.convx = nn.ModuleList([Linear(nhid, nhid, bias=True, weight_initializer='glorot') for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.layer = nlayer

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index)
        if self.layer == 1:
            return x

        if self.layer>2:
            for iter_layer in self.convx:
                x = iter_layer(x)
                x = F.relu(x)
                if self.dropout_p:
                    x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)

        return x


class StandGCNXBN(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        nhid = args.feat_dim
        dropout = args.dropout
        nlayer = args.layer
        is_add_self_loops = args.First_self_loop
        norm = args.gcn_norm
        self.is_add_self_loops = is_add_self_loops  # Qin True is the original
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)
            # self.conv1 = SAGEConv_QinNov(nfeat, nclass)      #  delete
            # self.conv1 = SAGEConv(nfeat, nclass)      #  delete
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())  # no effect to layer=1,
        # self.non_reg_params = self.conv2.parameters()

        self.layer = nlayer

    def forward(self, x, adj, args, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index)
        # x = self.mlp1(x)
        if self.layer == 1:
            return x
        # if args.BN_model:
        #     x = self.batch_norm1(x)
        x = F.relu(x)
        # if args.dropout:
        #     x = F.dropout(x, p=args.dropout, training=self.training)
        # if args.normalize:
        #     x = F.normalize(x, p=2, dim=1)

        if self.layer>2:
            for iter_layer in self.convx:
                # x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index)
                # if args.BN_model:
                #     x= self.batch_norm3(x)
                x = F.relu(x)
                if args.dropout:
                    x = F.dropout(x, p=args.dropout, training=self.training)
                # if args.normalize:
                #     x = F.normalize(x, p=2, dim=1)

        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        # if args.BN_model:
        #     x = self.batch_norm2(x)
        # x = F.relu(x)
        # if args.dropout:
        #     x = F.dropout(x, p=args.dropout, training=self.training)
        # if args.normalize:
        #     x = F.normalize(x, p=2, dim=1)

        # x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x

class StandGCNX_noRelu(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        nhid = args.feat_dim
        dropout = args.dropout
        nlayer = args.layer
        is_add_self_loops = args.First_self_loop
        norm = args.gcn_norm
        self.is_add_self_loops = is_add_self_loops  # Qin True is the original
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)
            # self.conv1 = SAGEConv_QinNov(nfeat, nclass)      #  delete
            # self.conv1 = SAGEConv(nfeat, nclass)      #  delete
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached= False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())  # no effect to layer=1,
        # self.non_reg_params = self.conv2.parameters()

        self.layer = nlayer

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index)
        # x = self.mlp1(x)
        if self.layer == 1:
            return x
        # if args.BN_model:
        #     x = self.batch_norm1(x)
        # x = F.relu(x)
        # if args.dropout:
        #     x = F.dropout(x, p=args.dropout, training=self.training)
        # if args.normalize:
        #     x = F.normalize(x, p=2, dim=1)

        if self.layer>2:
            for iter_layer in self.convx:
                # x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index)
                # if args.BN_model:
                #     x= self.batch_norm3(x)
                # x = F.relu(x)
                if self.dropout_p:
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                # if args.normalize:
                #     x = F.normalize(x, p=2, dim=1)

        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        # if args.BN_model:
        #     x = self.batch_norm2(x)
        # x = F.relu(x)
        # if args.dropout:
        #     x = F.dropout(x, p=args.dropout, training=self.training)
        # if args.normalize:
        #     x = F.normalize(x, p=2, dim=1)

        # x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x



class GraphSAGEXBatNorm(nn.Module):
    def __init__(self,  nfeat, nclass, args):
        super().__init__()
        self.dropout_p = args.dropout
        nhid = args.feat_dim
        nlayer= args.layer
        # self.Conv = nn.Conv1d(nhid*2 , nclass, kernel_size=1)
        # SAGEConv(input_dim, output_dim, root_weight=False)
        # SAGEConv = NormalizedSAGEConv  #  Qin
        # SAGEConv= SAGEConv_SHA
        # SAGEConv= SAGEConv_Qin
        # SAGEConv= GCNConv
        # SAGEConv= SAGEConv_QinNov
        self.conv1 = SAGEConv(nfeat, nhid)
        # self.conv1_1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        if nlayer >2:
            self.convx = nn.ModuleList([SAGEConv(nhid, nhid) for _ in range(nlayer-2)])
            # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        if nlayer==1:
            # self.batch_norm1 = nn.BatchNorm1d(nclass)

            self.conv1 = SAGEConv(nfeat, nclass)

            # self.conv1 = SAGEConv(nfeat, nhid)        #  delete after test Qin
            self.mlp1 = torch.nn.Linear(nhid, nhid)
            self.mlp2 = torch.nn.Linear(nhid, nhid)

        #     self.reg_params =[]
        #     self.non_reg_params = self.conv2.parameters()
        # else:
        #     self.non_reg_params = self.conv2.parameters()

        self.layer = nlayer
        self.BN = args.BN_model

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj
        x = self.conv1(x, edge_index)
        # x2 = self.conv1_1(x, edge_index, edge_weight)
        # x= torch.cat((x1, x2), dim=-1)
        # x = self.mlp1(x1) + self.mlp2(x2)
        # if self.BN:
        #     x = self.batch_norm1(x)
        if self.layer == 1:
            # x = x.unsqueeze(0)  # can't simplify, because the input of Conv1d is 3D
            # x = x.permute((0, 2, 1))
            # x = self.Conv(x)
            # x = F.log_softmax(x, dim=1)  # transforms the raw output scores (logits) into log probabilities, which are more numerically stable for computation and training
            # x = x.permute(2, 1, 0).squeeze()
            return x

        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index,edge_weight)
                if self.BN:
                    x = self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        if self.BN:
            x = self.batch_norm2(x)

        return x

class GATLikeLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformation for input features
        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data)

        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(1, num_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        src, dst = edge_index
        h = torch.mm(x, self.W).view(-1, self.num_heads, self.out_features)

        # Compute attention coefficients
        edge_h = torch.cat((h[src], h[dst]), dim=-1)
        edge_e = self.leakyrelu(torch.sum(self.a * edge_h.unsqueeze(1), dim=-1))

        # Normalize attention coefficients
        alpha = F.softmax(edge_e, dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Apply attention and aggregate
        out = scatter_add(alpha.unsqueeze(-1) * h[src], dst, dim=0, dim_size=x.size(0))

        if self.concat:
            out = out.view(-1, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        return out


class ParaGCNXBN00(nn.Module):
    def __init__(self, num_edges, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super().__init__()
        self.num_heads = 8
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(GATLikeLayer(nfeat, nhid, self.num_heads, dropout, concat=True))
        for _ in range(nlayer - 2):
            self.layers.append(GATLikeLayer(nhid * self.num_heads, nhid, self.num_heads, dropout, concat=True))
        self.layers.append(GATLikeLayer(nhid * self.num_heads, nclass, 1, dropout, concat=False))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(nhid * self.num_heads) for _ in range(nlayer - 1)])
        self.layer_norms.append(nn.LayerNorm(nclass))

    def forward(self, x, edge_index):
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = norm(x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x

class ParaGCNXBN2(nn.Module):
    def __init__(self, num_edges, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super().__init__()
        self.num_heads = 8
        self.out_features = nclass

        # Multi-head edge weights
        self.edge_weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_edges)) for _ in range(self.num_heads)
        ])

        # Dynamic weight computation
        self.weight_func = nn.Sequential(
            nn.Linear(nfeat * 2, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_heads)
        )

        # Output projection
        self.proj = nn.Linear(nfeat * self.num_heads, nclass)

        # Layer norm
        self.layer_norm = nn.LayerNorm(nclass)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        src, dst = edge_index

        # Compute dynamic weights
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        dynamic_weights = self.weight_func(edge_features).sigmoid()

        outputs = []
        for head in range(self.num_heads):
            # Combine static and dynamic weights
            edge_weight = self.edge_weights[head] * dynamic_weights[:, head]

            # Apply nonlinearity and normalization
            edge_weight = F.leaky_relu(edge_weight)
            edge_weight = F.softmax(edge_weight, dim=0)

            # Message passing
            out = scatter_add(x[src] * edge_weight.unsqueeze(1), dst, dim=0, dim_size=x.size(0))
            outputs.append(out)

        # Combine multi-head outputs
        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)
        out = self.layer_norm(out)

        return out


class ParaGCNXBN1(nn.Module):
    def __init__(self,num_node, num_edges, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super(ParaGCNXBN1, self).__init__()

        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=False)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=False) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached=False, normalize=False)
            self.batch_norm1 = nn.BatchNorm1d(nclass)
        elif nlayer > 1:
            self.conv1 = GCNConv(nfeat, nhid, cached=False, normalize=False)
            self.batch_norm1 = nn.BatchNorm1d(nhid)
            self.batch_norm3 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)

        self.is_add_self_loops = True
        if self.is_add_self_loops:
            num_edges = num_edges + num_node
        # self.edge_weight = nn.Parameter(torch.ones(size=(num_edges,)), requires_grad=True)
        self.edge_weight = nn.Parameter(torch.FloatTensor())
        self.edge_weight_src = nn.Parameter(torch.ones(size=(num_edges,)), requires_grad=True)
        self.edge_weight_dst = nn.Parameter(torch.ones(size=(num_edges,)), requires_grad=True)
        self.feature_scale = nn.Linear(nfeat, 1, bias=False)
        self.norm = norm

        if nlayer == 1:
            self.reg_params = list(self.conv1.parameters())
            self.non_reg_params = []
        else :
            self.reg_params = list(self.conv1.parameters())
            self.non_reg_params = list(self.conv2.parameters())
            if nlayer >2:
                self.reg_params += list(self.convx.parameters())

        self.layer = nlayer
        self.current_epoch = 0
        self.edge_mask = torch.ones_like(self.edge_weight, dtype=torch.bool, device=self.edge_weight.device)
        self.non_zero = 0

    def forward(self, x, adj):

        self.current_epoch += 1
        with torch.no_grad():  # Ensures this operation doesn't track gradients
            self.edge_weight[torch.isnan(self.edge_weight)] = 1

            self.edge_weight.data[self.edge_weight.data < 0] = 0
            self.edge_weight.data[self.edge_weight.data > 1] = 1

            self.edge_mask = (self.edge_mask).to(self.edge_weight.device)
            self.edge_mask = self.edge_mask & (self.edge_weight > 0)

        num_zeros1 = torch.sum(self.edge_weight.data == 0).item()

        if num_zeros1:
            if num_zeros1>self.non_zero:
                # print(f"After, Number of zeros in edge_weight: {num_zeros1}", str(int(self.current_epoch/3)))
                self.non_zero = num_zeros1

        # self.edge_weight.data = self.edge_weight * self.edge_mask
        # edge_weight = self.edge_weight
        edge_weight_src = self.edge_weight_src * self.edge_mask
        edge_weight_dst = self.edge_weight_dst * self.edge_mask
        self.edge_weight = edge_weight_src + edge_weight_dst
        edge_weight = binary_approx(self.edge_weight)
        edge_index = adj
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # edge_index = adj.flip(0)

        # self.edge_weight = edge_weight

        non_zero_indices = edge_weight != 0

        # Filter edge_index and edge_weight using non-zero indices
        edge_index = edge_index[:, non_zero_indices]
        edge_weight = edge_weight[non_zero_indices]

        if self.norm:
            edge_index, edge_weight = norm0(edge_index, edge_weight)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.batch_norm1(x)
        if self.layer == 1:
            return x
        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x= iter_layer(x, edge_index, edge_weight)
                x = self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x= self.conv2(x, edge_index, edge_weight)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class ParaGCNXBN(nn.Module):
    def __init__(self,num_node, num_edges, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super(ParaGCNXBN, self).__init__()

        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=False)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=False) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached=False, normalize=False)
            self.batch_norm1 = nn.BatchNorm1d(nclass)
        elif nlayer > 1:
            self.conv1 = GCNConv(nfeat, nhid, cached=False, normalize=False)
            self.batch_norm1 = nn.BatchNorm1d(nhid)
            self.batch_norm3 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)

        self.is_add_self_loops = True
        if self.is_add_self_loops:
            num_edges = num_edges + num_node
        self.edge_weight = nn.Parameter(torch.ones(size=(num_edges,)), requires_grad=True)
        self.norm = norm

        if nlayer == 1:
            self.reg_params = list(self.conv1.parameters())
            self.non_reg_params = []
        else :
            self.reg_params = list(self.conv1.parameters())
            self.non_reg_params = list(self.conv2.parameters())
            if nlayer >2:
                self.reg_params += list(self.convx.parameters())

        self.layer = nlayer
        self.current_epoch = 0
        self.edge_mask = torch.ones_like(self.edge_weight, dtype=torch.bool, device=self.edge_weight.device)
        self.non_zero = 0
        self.num_node = num_node

    def forward(self, x, adj):
        self.current_epoch += 1
        with torch.no_grad():  # Ensures this operation doesn't track gradients
            self.edge_weight[torch.isnan(self.edge_weight)] = 1

            self.edge_weight.data[self.edge_weight.data < 0] = 0
            self.edge_weight.data[self.edge_weight.data > 1] = 1

            self.edge_mask = (self.edge_mask).to(self.edge_weight.device)
            self.edge_mask = self.edge_mask & (self.edge_weight > 0)

        num_zeros1 = torch.sum(self.edge_weight.data == 0).item()

        if num_zeros1:
            if num_zeros1>self.non_zero:
                # print(f"After, Number of zeros in edge_weight: {num_zeros1}", str(int(self.current_epoch/3)))
                self.non_zero = num_zeros1

        # self.edge_weight.data = self.edge_weight * self.edge_mask
        self.edge_weight.data = self.edge_weight
        edge_weight = self.edge_weight
        edge_weight = binary_approx(edge_weight)
        edge_index = adj
        # edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_node)
        # edge_index = adj.flip(0)

        non_zero_indices = edge_weight != 0

        # Filter edge_index and edge_weight using non-zero indices
        edge_index = edge_index[:, non_zero_indices]
        edge_weight = edge_weight[non_zero_indices]

        if self.norm:
            edge_index, edge_weight = norm0(edge_index, edge_weight)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.batch_norm1(x)
        if self.layer == 1:
            return x
        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x= iter_layer(x, edge_index, edge_weight)
                x = self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x= self.conv2(x, edge_index, edge_weight)
        x = self.batch_norm2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

class BinaryEdgeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
def binary_approx(edge_weight, temperature=10.0):
    return torch.sigmoid(temperature * (edge_weight - 0.5))
