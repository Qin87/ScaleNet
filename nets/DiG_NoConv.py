from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, get_laplacian
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing, JumpingKnowledge, GATConv
from torch_geometric.nn.inits import glorot, zeros

from typing import  Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch import Tensor

from torch_geometric.utils import spmm

from nets.gat import UnifiedGATRATConv


class InceptionBlock_Di(torch.nn.Module):
    def __init__(self, m, in_dim, out_dim, args):
        super().__init__()
        head = args.heads
        self.dropout = args.dropout

        tuple_num = 2
        self.ln = Linear(in_dim, out_dim)
        if m in ['RiGib', 'UiGib', 'DiGib']:
            self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(tuple_num)])
        elif m in ['AiGib']:
            num_head = 1
            head_dim = out_dim // num_head
            # self.convx = nn.ModuleList([GATConv(in_dim, head_dim, heads=head) for _ in range(tuple_num)])
            self.convx = nn.ModuleList([UnifiedGATRATConv(in_dim, head_dim, heads=head,  args=args, concat=False) for _ in range(tuple_num)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        # self.lin_src_to_dst = nn.ModuleList([Linear(input_dim, output_dim) for _ in range(20)])

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.convx.reset_parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):

        x0 = self.ln(x)
        for i in range(len(edge_index_tuple)):
            x0 += F.dropout(self.convx[i](x, edge_index_tuple[i], edge_weight_tuple[i]), p=0.6, training=self.training)
            torch.cuda.empty_cache()

        return x0


class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




def Conv_Out(x, Conv):
    x = x.unsqueeze(0)
    x = x.permute((0, 2, 1))
    x = Conv(x)
    x = x.permute((0, 2, 1)).squeeze()

    return x


class DiSAGE_xBN_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self._cached_adj_t = None
        self.BN_model = args.BN_model
        self.dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.layer = args.layer
        head = args.heads
        K = args.K
        self.jk = args.jk
        # out1 = out_dim  if layer == 1 else nhid

        if self.jk is not None:
            in_dim_jk = nhid * self.layer if self.jk == "cat" else nhid
            # self.lin = Linear(input_dim, out_dim)
            self.lin = Linear(in_dim_jk, nhid)
            if self.jk:
                self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, nhid)
            self.conv2 = DiSAGEConv(nhid, nhid)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m in ['RiG', 'UiG']:
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, nhid)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, nhid, K)
            self.conv2 = DIChebConv(nhid, nhid, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m in ['AiG']:
            num_head = 1
            head_dim = nhid // num_head
            self.conv1 = UnifiedGATRATConv(input_dim, head_dim, heads=head,  args= args,concat=False)
            self.conv2 = UnifiedGATRATConv(nhid, head_dim, heads=head,  args= args, concat=False)
            self.convx = nn.ModuleList([UnifiedGATRATConv(nhid, head_dim, heads=head, args= args, concat=False) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        if self.layer == 1:
            self.reg_params = []
            self.non_reg_params = self.conv1.parameters()
        elif self.layer == 2:
            self.reg_params = list(self.conv1.parameters())
            self.non_reg_params = self.conv2.parameters()
        else:
            self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
            self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).t()

        edge_index = self._cached_adj_t

        xs = []
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)
        xs += [x]
        if self.layer == 1:
            x = Conv_Out(x, self.Conv)
            return x

        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(iter_layer(x, edge_index, edge_weight))
                xs += [x]

        x = F.dropout(x, self.dropout, training=self.training)
        x=self.conv2(x, edge_index, edge_weight)
        if self.BN_model:
            x = self.batch_norm2(x)
        xs += [x]

        if self.jk:
            x = self.jump(xs)
            x = self.lin(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)

        return x


class DiSAGE_x_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self.dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.layer = args.layer
        head = args.heads
        K = args.K
        if self.layer > 1:
            n_change = nhid
        else:
            n_change = out_dim
        self.jk = args.jk
        out1 = out_dim if layer == 1 else nhid

        if self.jk not in (0, None):
            in_dim_jk = nhid * self.layer if self.jk == "cat" else nhid
            # self.lin = Linear(input_dim, out_dim)
            self.lin = Linear(in_dim_jk, nhid)
            self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, n_change)
            self.conv2 = DiSAGEConv(nhid, out1)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m in ['RiG', 'UiG']:
            # self.conv1 = DIGCNConv(input_dim, n_change)
            self.conv1 = DIGCNConv(input_dim, nhid)  # Qin temp
            self.conv2 = DIGCNConv(nhid, out1)
            # self.conv2 = Linear(nhid, out1)     # Qin temp
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, n_change, K)
            self.conv2 = DIChebConv(nhid, out1, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m in ['AiG']:
            num_head = 1
            head_dim = nhid // num_head
            self.conv1 = UnifiedGATRATConv(input_dim, head_dim, heads=head, args=args, concat=False)
            self.conv2 = UnifiedGATRATConv(nhid, head_dim, heads=head, args=args, concat=False)
            self.convx = nn.ModuleList([UnifiedGATRATConv(nhid, head_dim, heads=head, args=args, concat=False) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        if self.jk not in (0, None):
            in_dim_jk = nhid * self.layer if self.jk == "cat" else nhid
            self.lin = Linear(in_dim_jk, out_dim)
            # self.lin = Linear(in_dim_jk, nhid)
            self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        # self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        xs = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        xs += [x]
        if self.layer == 1:
            x = F.dropout(x, self.dropout, training=self.training)
            return x

        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(iter_layer(x, edge_index, edge_weight))
                xs += [x]

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        xs += [x]

        if self.jk is not None and bool(self.jk):
            x = self.jump(xs)
            x = self.lin(x)

        return x  # log softmax operation, has the same dimension


class Di_IB_1_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, n_cls, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, n_cls, args)
        self.Conv = nn.Conv1d(nhid, n_cls, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(n_cls)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class Di_IB_2_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, out_dim, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters()) + list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        return x

class Di_IB_XBN_nhid_ConV(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self.nonlinear = args.nonlinear
        self._cached_adj_t = None
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.layer = args.layer
        self.BN_model = args.BN_model

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        # num_nodes = x.size(0)
        # if self._cached_adj_t is None:
        #     cached_list = []
        #     for edge_index, edge_weight in zip(edge_index_tuple, edge_weight_tuple):
        #         adj_t = SparseTensor(
        #             row=edge_index[0],  # target
        #             col=edge_index[1],  # source
        #             value=edge_weight,  # normalized weights
        #             sparse_sizes=(num_nodes, num_nodes),
        #         )
        #         cached_list.append(adj_t)
        #         self._cached_adj_t = tuple(cached_list)
        #
        # edge_index_tuple = self._cached_adj_t
        # layer Normalization best only one at last layer, good for telegram
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = F.dropout(x, p=self._dropout, training=self.training)

        if self.layer == 1:
            if self.BN_model:
                x = self.batch_norm1(x)
            x = Conv_Out(x, self.Conv)
            return x

        if self.nonlinear:
            x = F.relu(x)
        if self.layer > 2:
            for iter_layer in self.ibx:
                x = F.dropout(x, p=self._dropout, training=self.training)
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)
                if self.nonlinear:
                    x = F.relu(x)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        if self.BN_model:
            x = self.batch_norm2(x)
        x = Conv_Out(x, self.Conv)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class Di_IB_X_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, out_dim, args)
        self.layer = args.layer
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = F.dropout(x, p=self._dropout, training=self.training)

        for iter_layer in self.ibx:
            x = F.dropout(x, p=self._dropout, training=self.training)
            x = iter_layer(x, edge_index_tuple, edge_weight_tuple)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x



def create_Di_IB_nhid(m, nfeat, nclass, args):
    if args.layer == 1:
        model = Di_IB_1_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = Di_IB_2_nhid(m, nfeat, nclass, args)
    else:
        model = Di_IB_X_nhid(m, nfeat, nclass, args)
    return model
