from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, get_laplacian
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing, JumpingKnowledge, GATConv
from torch_geometric.nn.inits import glorot, zeros

from nets.edge_data import normalize_row_edges
from nets.Sym_Reg import DGCNConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch import Tensor


class InceptionBlock_Qinlist(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = Linear(in_dim, out_dim)
        # self.conv1 = DIGCNConv(in_dim, out_dim)
        # self.conv2 = DIGCNConv(in_dim, out_dim)
        self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(20)])

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.convx.reset_parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        x0 = self.ln(x)
        x_list = [x0]
        for i in range(len(edge_index_tuple)):
            x_list.append(self.convx[i](x, edge_index_tuple[i], edge_weight_tuple[i]))
        return x_list

class InceptionBlock_Di_list(torch.nn.Module):
    def __init__(self, m, in_dim, out_dim,args):
        super().__init__()
        self.ln = Linear(in_dim, out_dim)
        head = args.heads
        K = args.K

        if m == 'S':
            self.convx = nn.ModuleList([DiSAGEConv(in_dim, out_dim) for _ in range(20)])
        elif m == 'G':
            self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(20)])
        elif m == 'C':
            self.convx = nn.ModuleList([DIChebConv(in_dim, out_dim, K) for _ in range(20)])
        elif m == 'A':
            num_head = 1
            head_dim = out_dim // num_head
            self.convx = nn.ModuleList([GATConv(in_dim, head_dim, heads=head) for _ in range(20)])
        else:
            raise ValueError(f"Model '{m}' not implemented")
        self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(20)])

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.convx.reset_parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        x0 = self.ln(x)
        x_list = [x0]
        for i in range(len(edge_index_tuple)):
            x_list.append(self.convx[i](x, edge_index_tuple[i], edge_weight_tuple[i]))
        return x_list

class InceptionBlock_Di(torch.nn.Module):
    def __init__(self, m, in_dim, out_dim, args):
        super().__init__()
        head = args.heads
        K = args.K
        self.dropout = args.dropout
        self.fusion_mode = args.fs
        alpha_dir = args.alphaDir

        self.ln = Linear(in_dim, out_dim)
        if m == 'S':
            self.convx = nn.ModuleList([DiSAGEConv(in_dim, out_dim) for _ in range(20)])
        elif m == 'G':
            self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(20)])
            # self.convx = nn.ModuleList([DirGCNConv(in_dim, out_dim, alpha_dir) for _ in range(20)])
            # self.convx = nn.ModuleList([DirGCNConv(in_dim, out_dim) for _ in range(20)])
            # self.convx = DirGCNConv(in_dim, out_dim)
        elif m == 'C':
            self.convx = nn.ModuleList([DIChebConv(in_dim, out_dim, K) for _ in range(20)])
        elif m == 'A':
            num_head = 1
            head_dim = out_dim // num_head
            self.convx = nn.ModuleList([GATConv(in_dim, head_dim,  heads=head) for _ in range(20)])
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


def union_edges(num_node, edge_index_tuple, device, mode):
    if mode == 'union':
        concatenated_tensor = torch.cat(edge_index_tuple, dim=1)
        edges_tuples = list(set(zip(concatenated_tensor[0].tolist(), concatenated_tensor[1].tolist())))
        edges = torch.tensor(edges_tuples).T
    else:
        edges = edge_index_tuple[-1]
    weights = normalize_row_edges(edge_index=edges, num_nodes= num_node)

    return edges.to(device), weights.to(device)



class DIChebConv(MessagePassing):
    r"""The Chebyshev graph convolutional operator for directed graphs.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of Chebyshev polynomials. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K, normalization: Optional[str] = 'sym',
                 bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        from torch_geometric.nn.dense.linear import Linear
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        # self.cached_result = None

    def __norm__(
            self,
            edge_index: Tensor,
            num_nodes: Optional[int],
            edge_weight: OptTensor,
            normalization: Optional[str],
            lambda_max: OptTensor = None,
            dtype: Optional[int] = None,
            batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            # lambda_max,
            dtype=x.dtype,
            # batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  # Dummy assignment for Tx_1
        out = self.lins[0](Tx_0)
        # out = torch.matmul(Tx_0, self.weight[0])

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
    # def __repr__(self):
    #     return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels,
    #                                      self.out_channels, self.K)


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

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class DiSAGEConv(MessagePassing):      #    Qin from Claude
    r"""The Directed GraphSAGE operator.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, normalize=False,  cached=True, bias=True, **kwargs):
        super().__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_neighbor = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_neighbor)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
            # else:
            #     norm = edge_weight
            self.cached_result = edge_index, edge_weight

        edge_index, edge_weight = self.cached_result

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j

    def update(self, aggr_out, x):
        out = torch.matmul(x, self.weight) + torch.matmul(aggr_out, self.weight_neighbor)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

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
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
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
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, nhid)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, nhid, K)
            self.conv2 = DIChebConv(nhid, nhid, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv(input_dim, head_dim,  heads=head)
            self.conv2 = GATConv(nhid, head_dim,  heads=head)
            self.convx = nn.ModuleList([GATConv(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
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
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
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
    def __init__(self, m, input_dim,  out_dim, args):
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
        out1 = out_dim  if layer == 1 else nhid

        if self.jk not in (0, None):
            in_dim_jk = nhid * self.layer if self.jk == "cat" else nhid
            # self.lin = Linear(input_dim, out_dim)
            self.lin = Linear(in_dim_jk, nhid)
            self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, n_change)
            self.conv2 = DiSAGEConv(nhid, out1)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            # self.conv1 = DIGCNConv(input_dim, n_change)
            self.conv1 = DIGCNConv(input_dim, nhid)  # Qin temp
            self.mlp1 = Linear(nhid, nhid)      # Qin temp
            self.mlp12 = Linear(nhid, nhid)      # Qin temp
            self.mlp13 = Linear(nhid, nhid)      # Qin temp
            self.mlp11 = Linear(nhid, out1)      # Qin temp
            self.conv2 = DIGCNConv(nhid, out1)
            self.mlp21 = Linear(out1, out1)
            self.mlp23 = Linear(out1, out1)
            self.mlp2 = Linear(out1, out1)
            self.mlp22 = Linear(out1, out1)
            # self.conv2 = Linear(nhid, out1)     # Qin temp
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, n_change, K)
            self.conv2 = DIChebConv(nhid, out1, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv(input_dim, n_change // num_head,  heads=head)
            self.conv2 = GATConv(nhid, out1//num_head,  heads=head)
            self.convx = nn.ModuleList([GATConv(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
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
        # x = self.mlp1(x)    # Qin temp
        # x = self.mlp12(x)    # Qin temp
        # x = self.mlp13(x)    # Qin temp
        # x = self.mlp11(x)    # Qin temp  # ######  using this
        # x = F.relu(x)  # Qin temp
        xs += [x]
        if self.layer == 1:
            x = F.dropout(x, self.dropout, training=self.training)
            return x

        x = F.relu(x)

        if self.layer > 2:
            for iter_layer in self.convx:
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(iter_layer(x, edge_index, edge_weight))
                # x = self.mlp23(x)  # Qin temp
                # x = self.mlp21(x)  # Qin temp
                # x = self.mlp2(x)  # Qin temp
                xs += [x]

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = self.mlp2(x)  # Qin temp
        # x = self.mlp23(x)  # Qin temp
        # x = self.mlp21(x)  # Qin temp
        # x = self.mlp22(x)  # Qin temp
        # x = F.relu(x)  # Qin temp
        # x = self.conv2(x)       # Qin temp
        xs += [x]

        if self.jk is not None and bool(self.jk):
            x = self.jump(xs)
            x = self.lin(x)

        return x   # log softmax operation, has the same dimension


class DiG_SimpleXBN_nhid_Pan(torch.nn.Module):
    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)
        self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer-2)])
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight, w_layer):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        out_list = []
        out_list.append(x)

        for conv in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            out_list.append(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.batch_norm2(x)
        out_list.append(x)

        # Sum up the contributions of all layers based on w_layer
        output = sum(weight * layer_out for weight, layer_out in zip(w_layer, out_list))

        x = output.unsqueeze(0).permute(0, 2, 1)
        x = self.Conv(x)
        x = x.permute(0, 2, 1).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)

        return x


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
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, out_dim, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters())+list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        return x


class DiGCN_IB_1BN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        dropout = args.dropout
        layer = args.layer
        
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx

        x = self.batch_norm1(x)     # keep it is better performance

        x= x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = F.dropout(x, p=self._dropout, training=self.training)   # only dropout during training   keep this is better
        return x

class DiGIB_1BN_Sym_nhid_para(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self._dropout = args.dropout
        self.ib1 = InceptionBlock_Qinlist(input_dim, nhid)
        self.ib2 = InceptionBlock_Qinlist(nhid, nhid)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])        # coef for ib1
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs =  self.coef1

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        DiGx = x_list[0]
        for i in range(1, len(x_list)):
            DiGx += self.coef1[i-1] * x_list[i]
        x = DiGx + symx
        x = self.batch_norm1(x)     # keep it is better performance

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = F.dropout(x, p=self._dropout, training=self.training)   # only dropout during training   keep this is better
        return x


class DiGIB_2BN_Sym_nhid_para(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Qinlist(input_dim, nhid)
        self.ib2 = InceptionBlock_Qinlist(nhid, nhid)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])        # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])        # coef for ib1

        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs = list(self.coef1) + list(self.coef2)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        DiGx = x_list[0]
        for i in range(1, len(x_list)):
            DiGx += self.coef1[i - 1] * x_list[i]
        x = DiGx + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x_list = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        DiGx = x_list[0]
        for i in range(1, len(x_list)):
            DiGx += self.coef1[i - 1] * x_list[i]
        x = DiGx + symx
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_Sym_nhid(torch.nn.Module):
    def __init__(self,m, input_dim, out_dim,args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        
        
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        # if self._dropout > 0:
        x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layer in self.linx:
            symx = iter_layer(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            x = x + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGIB_XBN_Sym_nhid_para(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Qinlist(input_dim, nhid)
        self.ib2 = InceptionBlock_Qinlist(nhid, nhid)
        self.ibx = InceptionBlock_Qinlist(nhid, nhid)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])  # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])  # coef for ib1
        self.coef3 = nn.ParameterList([nn.Parameter(torch.tensor(1.0, requires_grad=True)) for _ in range(20)])  # coef for ib1

        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(args.layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs = list(self.coef1) + list(self.coef2)+ list(self.coef3)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        DiGx = x_list[0]
        for i in range(1, len(x_list)):
            DiGx += self.coef1[i - 1] * x_list[i]
        x = DiGx + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layer in self.linx:
            symx = iter_layer(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x_list = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            DiGx = x_list[0]
            for i in range(1, len(x_list)):
                DiGx += self.coef3[i - 1] * x_list[i]
            x = DiGx + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x_list = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        DiGx = x_list[0]
        for i in range(1, len(x_list)):
            DiGx += self.coef2[i - 1] * x_list[i]
        x = DiGx + symx
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2BN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    all ib ends with nhid
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)     # worse with this
        # symx = F.relu(symx)

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        # x = self.batch_norm1(x)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        # x = self.batch_norm1(x)       # interpret it got no improvement
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3
        # symx = self.batch_norm1(symx)    # interpret this gets better!(BN only for the sum, not for the specific symx or digx)
        # symx = F.relu(symx)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        # x = self.batch_norm1(x)
        x = torch.cat((x, symx), dim=-1)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)

        # x = self.Conv2(x)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        return x


class DiGCN_IB_XBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        self.layer= args.layer
        layer = args.layer
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])
        self.batch_norm2 = nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])
        # self.linx = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)       # without this is faster
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_lin, iter_ib in zip(self.linx, self.ibx):
            symx = iter_lin(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            DiGx = iter_ib(x, edge_index_tuple, edge_weight_tuple)
            x = torch.cat((DiGx, symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)  # with this block or without, almost the same result
            x = x.permute((0, 2, 1)).squeeze()
            # x = self.batch_norm1(x)       # without this is better
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = self.batch_norm2(x)     # keep this is better
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_SymCat_1ibx_nhid(torch.nn.Module):
    '''
    revised for edge_index confusionx
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self.layer= args.layer
        self._dropout = args.dropout
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = InceptionBlock_Di(m, nhid, nhid, args)
        # self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(args.layer - 2)])
        # self.linx = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)       # without this is faster
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # for iter_lin, iter_ib in zip(self.linx, self.ibx):
        for iter_lin in self.linx:
            symx = iter_lin(x)
            symx1 = self.gconv(symx, edge_index)
            symx2 = self.gconv(symx, edge_in, in_w)
            symx3 = self.gconv(symx, edge_out, out_w)
            symx = symx1 + symx2 + symx3

            x = self.ibx(x, edge_index_tuple, edge_weight_tuple)
            x = torch.cat((x, symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)  # with this block or without, almost the same result
            x = x.permute((0, 2, 1)).squeeze()
            # x = self.batch_norm1(x)       # without this is better
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)

        x = self.batch_norm2(x)     # keep this is better
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_1BN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index comfusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        # self.Conv1 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)     # very bad
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        symx = self.batch_norm1(symx)

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        DiGx = self.batch_norm1(DiGx)
        x = torch.cat((DiGx, symx), dim=-1)
        x = self.batch_norm2(x)     # with this and BN1 is better
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)     # keep this is better!

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        return x


class DiGCN_IB_2MixBN_SymCat_nhid(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self._dropout = args.dropout
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)     # keep both it and the endBN is better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2MixBN_SymCat_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    nhid
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters()) + list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)

        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.Conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x


class DiGCN_IB_3MixBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, m, input_dim,  out_dim, args):
        super().__init__()
        self.layer = args.layer
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)
        self.Convx = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(args.layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((x, symx), dim=-1)
        #
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --DiGib
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                symx = iter_layer(x)
                symx1 = self.gconv(symx, edge_index)
                symx2 = self.gconv(symx, edge_in, in_w)
                symx3 = self.gconv(symx, edge_out, out_w)
                x = symx1 + symx2 + symx3

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3
        x = self.batch_norm3(x)  # keep this is better performance

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Convx(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()


        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x


class DiGCN_IB_3MixBN_SymCat_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self,m,  input_dim,  out_dim, args):
        super().__init__()
        self.layer = args.layer
        nhid = args.hid_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(args.layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(args.layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG
        symx = self.lin1(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3

        DiGx = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = torch.cat((DiGx, symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --addSym
        symx = self.lin2_(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        symx = symx1 + symx2 + symx3
        x=symx
        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.ibx:
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)

                # x = self.batch_norm2(x)
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        symx = self.lin2(x)
        symx1 = self.gconv(symx, edge_index)
        symx2 = self.gconv(symx, edge_in, in_w)
        symx3 = self.gconv(symx, edge_out, out_w)
        x = symx1 + symx2 + symx3

        x = self.batch_norm3(x)     # with this, much better than without
        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x


class Di_IB_XBN_nhid_ConV(torch.nn.Module):
    def __init__(self, m, input_dim,   out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer
        self.layer = args.layer
        self.BN_model = args.BN_model

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])

        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        # layer Normalization best only one at last layer, good for telegram
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = F.dropout(x, p=self._dropout, training=self.training)

        if self.layer == 1:
            if self.BN_model:
                x = self.batch_norm1(x)
            x = Conv_Out(x, self.Conv)
            return x

        # x = F.relu(x)
        if self.layer > 2:
            for iter_layer in self.ibx:
                x = F.dropout(x, p=self._dropout, training=self.training)
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        if self.BN_model:
            x = self.batch_norm2(x)
        x = Conv_Out(x, self.Conv)
        x = F.dropout(x, p=self._dropout, training=self.training)

        return x

class Di_IB_XBN_nhid_ConV_JK(torch.nn.Module):
    def __init__(self, m, input_dim,   out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        self.jk= args.jk
        self.BNorm = args.BN_model

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.layer = args.layer
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(self.layer - 2)])

        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        if self.jk is not None:
            input_dim = nhid * self.layer if self.jk == "cat" else nhid
            self.lin = Linear(input_dim, out_dim)
            # self.lin = Linear(input_dim, nhid)
            self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        xs = []
        # layer Normalization best only one at last layer, good for telegram
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        # x = F.dropout(x, p=self._dropout, training=self.training)

        if self.layer == 1:
            if self.BNorm:
                x = self.batch_norm1(x)
            x = Conv_Out(x, self.Conv)
            return x

        x = F.relu(x)
        xs += [x]
        if self.layer > 2:
            for iter_layer in self.ibx:
                # x = F.dropout(x, p=self._dropout, training=self.training)
                x = iter_layer(x, edge_index_tuple, edge_weight_tuple)
                x = F.relu(x)
                xs += [x]

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = F.relu(x)
        if self.BNorm:
            x = self.batch_norm2(x)
        xs += [x]


        if self.jk is not None:
            x = self.jump(xs)
            x = self.lin(x)
        # x = Conv_Out(x, self.Conv)
        # x = F.dropout(x, p=self._dropout, training=self.training)


        return x        #
        # return torch.nn.functional.log_softmax(x, dim=1)

class Di_IB_X_nhid(torch.nn.Module):
    def __init__(self, m, input_dim,   out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        layer = args.layer

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, out_dim, args)
        self.layer = args.layer
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])

        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

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


class DiGCN_IB_XBN_nhid_para(torch.nn.Module):
    def __init__(self, m, num_features, out_dim, args):
        super().__init__()
        nhid = args.hid_dim
        if args.layer==1:
            self.ib1 = InceptionBlock_Di_list(m, num_features, out_dim, args)
        else:
            self.ib1 = InceptionBlock_Di_list(m, num_features, nhid, args)
        self.ib2 = InceptionBlock_Di_list(m, nhid, out_dim, args)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib2
        self._dropout = args.dropout
        self.BN_model = args.BN_model

        self.layer = args.layer
        layer = args.layer
        self.ibx=nn.ModuleList([InceptionBlock_Di_list(m, nhid,nhid, args) for _ in range(layer-2)])
        self.coefx = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)]) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs = list(self.coef1)+list(self.coef2)+list(self.coefx.parameters())
        # self.coefs = [self.coef1, self.coef2, self.coefx]     # wrong

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef1[i] * x_list[i]
        x = F.dropout(x, p=self._dropout, training=self.training)
        if self.layer == 1:
            if self.BN_model:
                x = self.batch_norm1(x)
            return x

        if self.layer > 2:
            for iter_layer, iter_coef in zip(self.ibx, self.coefx):
                x_list = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
                x = x_list[0]
                for i in range(1, len(x_list)):
                    x += iter_coef[i] * x_list[i]
                x = F.dropout(x, p=self._dropout, training=self.training)

        x_list = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
        if self.BN_model:
            x = self.batch_norm2(x)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef2[i] * x_list[i]

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_X_nhid_para_Jk(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super().__init__()
        self._dropout = args.dropout
        nhid = args.hid_dim
        self.jumping_knowledge = args.jk
        num_layers = args.layer
        output_dim = nhid if self.jumping_knowledge else out_dim
        self.BNorm = args.BN_model
        if args.layer==1:
            self.ib1 = InceptionBlock_Di_list(m, input_dim, output_dim, args)
        else:
            self.ib1 = InceptionBlock_Di_list(m, input_dim, nhid, args)

        self.ib2 = InceptionBlock_Di_list(m, nhid, output_dim, args)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib2
        self._dropout = args.dropout

        self.batch_norm2 = nn.BatchNorm1d(nhid)

        if self.jumping_knowledge:
            input_dim = nhid * num_layers if self.jumping_knowledge == "cat" else nhid
            self.lin = Linear(input_dim, out_dim)
            self.jump = JumpingKnowledge(mode=self.jumping_knowledge, channels=nhid, num_layers=num_layers)

        self.layer = args.layer
        layer = args.layer
        self.ibx=nn.ModuleList([InceptionBlock_Di_list(m, nhid, nhid, args) for _ in range(layer-2)])
        self.coefx = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)]) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs = list(self.coef1)+list(self.coef2)+list(self.coefx.parameters())

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        xs = []
        x = features
        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef1[i] * x_list[i]
        x = F.dropout(x, p=self._dropout, training=self.training)
        xs += [x]
        if self.layer == 1:
            return x

        if self.layer > 2:
            for iter_layer, iter_coef in zip(self.ibx, self.coefx):
                x_list = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
                x = x_list[0]
                for i in range(1, len(x_list)):
                    x += iter_coef[i] * x_list[i]
                x = F.dropout(x, p=self._dropout, training=self.training)
                xs += [x]

        x_list = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef2[i] * x_list[i]
        if self.BNorm:
            x = self.batch_norm2(x)
        xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x



def create_Di_IB_nhid(m, nfeat, nclass, args):
    if args.layer == 1:
        model = Di_IB_1_nhid(m, nfeat,  nclass, args)
    elif args.layer == 2:
        model = Di_IB_2_nhid(m, nfeat,  nclass, args)
    else:
        model = Di_IB_X_nhid(m, nfeat,  nclass, args)
    return model


def create_DiG_IB_Sym_nhid(m, nfeat,  nclass, args):
    '''
    revised for edge_index confusion
    '''
    nlayer = args.layer
    if args.layer == 1:
        model = DiGCN_IB_1BN_Sym_nhid(m, nfeat,  nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_Sym_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_Sym_nhid(m, nfeat, nclass, args)
    return model

def create_DiG_IB_Sym_nhid_para(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    '''
    if args.layer == 1:
        model = DiGIB_1BN_Sym_nhid_para(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGIB_2BN_Sym_nhid_para(m, nfeat, nclass, args)
    else:
        model = DiGIB_XBN_Sym_nhid_para(m, nfeat, nclass, args)
    return model


def create_DiG_IB_SymCat_nhid(m, nfeat, nclass, args, ibx1):
    '''
    revised for edge_index confusion
    only has nhid version,
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if args.layer == 1:
        model = DiGCN_IB_1BN_SymCat_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_SymCat_nhid(m, nfeat, nclass, args)
    else:
        if ibx1:
            model = DiGCN_IB_XBN_SymCat_1ibx_nhid(m, nfeat, nclass, args)
        else:
            model = DiGCN_IB_XBN_SymCat_nhid(m, nfeat, nclass, args)
    return model


def create_DiG_MixIB_SymCat_nhid(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_nhid(m, nfeat, nclass, args)
    return model


def create_DiG_MixIB_SymCat_Sym_nhid(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    all main layer end with nhid
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:

    Returns:

    '''
    nlayer = args.layer
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_nhid(m, nfeat, nclass, args)
    return model
