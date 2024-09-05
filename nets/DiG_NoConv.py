import sys

from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, get_laplacian
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter, Linear
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

from edge_nets.edge_data import normalize_row_edges
from nets.Sym_Reg import DGCNConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor

from nets.geometric_baselines import DirGCNConv, DirGCNConv_Qin


class InceptionBlock_Qinlist(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock_Qinlist, self).__init__()
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
        super(InceptionBlock_Di_list, self).__init__()
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
            self.convx = nn.ModuleList([GATConv_Qin(in_dim, head_dim, heads=head) for _ in range(20)])
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
        super(InceptionBlock_Di, self).__init__()
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
            self.convx = nn.ModuleList([GATConv_Qin(in_dim, head_dim,  heads=head) for _ in range(20)])
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



class InceptionBlock_Si(torch.nn.Module):
    def __init__(self, m, in_dim, out_dim, args):
        super(InceptionBlock_Si, self).__init__()
        head = args.heads
        K = args.K

        self.ln = Linear(in_dim, out_dim)
        if m == 'S':
            self.convx = DiSAGEConv(in_dim, out_dim)
        elif m == 'G':
            self.convx = DIGCNConv(in_dim, out_dim)
        elif m == 'C':
            self.convx = DIChebConv(in_dim, out_dim, K)
        elif m == 'A':
            num_head = 1
            head_dim = out_dim // num_head
            self.convx = GATConv_Qin(in_dim, head_dim,  heads=head)
        else:
            raise ValueError(f"Model '{m}' not implemented")

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.convx.reset_parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):
        device = x.device
        # print(device)
        x0 = self.ln(x)

        x0 += F.dropout(self.convx(x, edge_index_tuple, edge_weight_tuple), p=0.6, training=self.training)
        torch.cuda.empty_cache()
        return x0


class InceptionBlock_Di0(torch.nn.Module):
    def __init__(self, m, in_dim, out_dim, head=8):
        super(InceptionBlock_Di0, self).__init__()

        self.ln = Linear(in_dim, out_dim)
        if m == 'S':
            self.convx = nn.ModuleList([DiSAGEConv(in_dim, out_dim) for _ in range(20)])
        elif m == 'G':
            self.convx = nn.ModuleList([DIGCNConv(in_dim, out_dim) for _ in range(20)])
        # elif m == 'C':
        #     self.convx = nn.ModuleList([DIChebConv(in_dim, out_dim, K) for _ in range(20)])
        elif m == 'A':
            num_head = 1
            head_dim = out_dim // num_head
            self.convx = nn.ModuleList([GATConv_Qin(in_dim, head_dim, heads=head) for _ in range(20)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.convx.reset_parameters()

    def forward(self, x, edge_index_tuple, edge_weight_tuple):

        x0 = self.ln(x)
        for i in range(len(edge_index_tuple)):
            x0 += self.convx[i](x, edge_index_tuple[i], edge_weight_tuple[i])
            # torch.cuda.empty_cache()
        return x0


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
        super(DIChebConv, self).__init__(aggr='add', **kwargs)

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
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

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

class GATConv_Qin(MessagePassing):       # change from GAT
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = False,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 bias: bool = True, **kwargs):      # concat should be False. true case haven't try
        kwargs.setdefault('aggr', 'add')
        super(GATConv_Qin, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        if isinstance(in_channels, int):
            self.temp_weight = torch.nn.Linear(in_channels, heads*out_channels, bias=False)
            self.lin_l = self.temp_weight #Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            try:
                self.lin_l = torch.nn.Linear(in_channels[0],  out_channels, False)
                self.lin_r = torch.nn.Linear(in_channels[1],  out_channels, False)
            except:     # Ben for IndexError: invalid index of a 0-dim tensor
                self.lin_l = torch.nn.Linear(in_channels, out_channels, False)
                self.lin_r = torch.nn.Linear(in_channels, out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight=None,
                size: Size = None, return_attention_weights=None, is_add_self_loops: bool = True):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        H, C = self.heads, self.out_channels
        original_size = edge_index.shape[1]
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            #x_lyy = x_r = self.lin_l(x).view(-1, H, C)
            x = self.lin_l(x)   # .view(-1, H, C)
            x_l = x_r = x.view(-1,H,C)

            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if is_add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                # if x_r is not None:       # Qin delete
                #     num_nodes = min(num_nodes, x_r.size(0))
                # if size is not None:
                #     num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index,x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        '''
        -> Tensor indicates the expected return type of a function
        Args:
            x_j:
            alpha_j:
            alpha_i:
            index:
            ptr:
            size_i:

        Returns:

        '''
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,self.in_channels,self.out_channels, self.heads)


class DiGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=8, concat=False,
                 negative_slope=0.2, dropout=0.6, bias=True, **kwargs):
        super(DiGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # self.lin = nn.Linear(heads, heads*out_channels)

        self.lin = Linear(in_channels,  out_channels, bias=False)
        self.att_src = Parameter(torch.Tensor(1, heads,  out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads,  out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        x = self.lin(x)     # Shape: [N, out_channels]
        x = x.unsqueeze(1)
        x = x.expand(-1, self.heads, -1)  # Now x has shape (N, H, 64)
        # x= x.view(-1, self.heads, self.out_channels)  # Shape: [N, heads, out_channels]
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, x_i, x_j, edge_weight, size):
        alpha_i = (x_i * self.att_src).sum(dim=-1)
        alpha_j = (x_j * self.att_dst).sum(dim=-1)
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size[0])

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        alpha = alpha * edge_weight.view(-1, 1)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

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
        super(DiSAGEConv, self).__init__(aggr='mean', **kwargs)

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

class DiG_Simple1(nn.Module):
    def __init__(self, input_dim, nhid, out_dim,  dropout, layer=1):
        super(DiG_Simple1, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

        # # # type2
        # self.reg_params = list(self.conv1.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple2(nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout, layer=2):
        super(DiG_Simple2, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

        # # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_SimpleX(torch.nn.Module):
    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super(DiG_SimpleX, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, out_dim)
        self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer-2)])
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

        # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.relu(self.conv2(x, edge_index))      # I should desert this line
        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple1BN(nn.Module):
    def __init__(self, input_dim, nhid, out_dim,  dropout, layer=1):
        super(DiG_Simple1BN, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        # x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class DiG_Simple1BN_nhid(nn.Module):
    def __init__(self, input_dim, nhid, out_dim,  dropout, layer=1):
        super(DiG_Simple1BN_nhid, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, nhid)
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        x = self.conv1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

def Conv_Out(x, Conv):
    x = x.unsqueeze(0)
    x = x.permute((0, 2, 1))
    x = Conv(x)
    x = x.permute((0, 2, 1)).squeeze()

    return x

class DiSAGE_1_nhid(nn.Module):
    def __init__(self, m, in_dim, out_dim,  args):
        super(DiSAGE_1_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads

        if m == 'S':
            self.conv1 = DiSAGEConv(in_dim, out_dim)
        elif m == 'G':
            self.conv1 = DIGCNConv(in_dim, out_dim)
        elif m == 'A':
            num_head = 1
            head_dim = out_dim // num_head

            self.conv1 = GATConv_Qin(in_dim, head_dim, heads=head)
            # self.conv1 = DiGATConv(in_dim, nhid, nhid)     # little difference from GATConv_Qin
        elif m == 'C':
            self.conv1 = DIChebConv(in_dim, out_dim, args.K)
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)


        return x

class DiSAGE_1BN_nhid0(nn.Module):
    def __init__(self, m, in_dim, nhid, out_dim,  args, layer=1,  head=8):
        super(DiSAGE_1BN_nhid0, self).__init__()
        self.dropout = args.dropout
        if m == 'S':
            self.conv1 = DiSAGEConv(in_dim, nhid)
        elif m == 'G':
            self.conv1 = DIGCNConv(in_dim, nhid)
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(in_dim, head_dim, heads=head)

            # self.conv1 = DiGATConv(in_dim, nhid, nhid)     # little difference from GATConv_Qin
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        x = self.conv1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Simple2BN_nhid(nn.Module):
    def __init__(self, input_dim, nhid, ncls, dropout, layer=2):
        super(DiG_Simple2BN_nhid, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)
        self.Conv = nn.Conv1d(nhid, ncls, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))      # no BN here is better
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()


        return x

# class DiSAGE_2BN_nhid0(nn.Module):
#     def __init__(self, m, input_dim, nhid, ncls, dropout, layer=2, head=8):
#         super(DiSAGE_2BN_nhid0, self).__init__()
#         self.dropout = dropout
#
#         self.Conv = nn.Conv1d(nhid, ncls, kernel_size=1)
#
#         self.batch_norm1 = nn.BatchNorm1d(nhid)
#         self.batch_norm2 = nn.BatchNorm1d(nhid)
#
#         if m == 'S':
#             self.conv1 = DiSAGEConv(input_dim, nhid)
#             self.conv2 = DiSAGEConv(nhid, nhid)
#             self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
#         elif m == 'G':
#             self.conv1 = DIGCNConv(input_dim, nhid)
#             self.conv2 = DIGCNConv(nhid, nhid)
#             self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
#         elif m == 'A':
#             num_head = 1
#             head_dim = nhid // num_head
#
#             self.conv1 = GATConv_Qin(input_dim, head_dim, heads=head)
#             self.conv2 = GATConv_Qin(nhid, head_dim, heads=head)
#             self.batch_norm1 = nn.BatchNorm1d(head_dim)
#             self.batch_norm2 = nn.BatchNorm1d(head_dim)
#         elif m == 'C':
#             self.conv1 = DIChebConv(input_dim, nhid)
#             self.conv2 = DIChebConv(nhid, nhid)
#             self.convx = nn.ModuleList([DIChebConv(nhid, nhid) for _ in range(layer - 2)])
#         else:
#             raise ValueError(f"Model '{m}' not implemented")
#
#         self.reg_params = list(self.conv1.parameters())
#         self.non_reg_params = self.conv2.parameters()
#
#     def forward(self, x, edge_index, edge_weight):
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.conv1(x, edge_index, edge_weight))  # no BN here is better
#         # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
#
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = x.unsqueeze(0)
#         x = x.permute((0, 2, 1))
#         x = self.Conv(x)
#         x = x.permute((0, 2, 1)).squeeze()
#
#         return x

class DiSAGE_2BN_nhid(nn.Module):
    def __init__(self, m, input_dim,  ncls, args):
        super(DiSAGE_2BN_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads
        K = args.K

        self.Conv = nn.Conv1d(nhid, ncls, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, nhid)
            self.conv2 = DiSAGEConv(nhid, nhid)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, nhid)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(input_dim, head_dim, heads=head)
            self.conv2 = GATConv_Qin(nhid, head_dim, heads=head)
            self.batch_norm1 = nn.BatchNorm1d(head_dim)
            self.batch_norm2 = nn.BatchNorm1d(head_dim)
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, nhid, K)
            self.conv2 = DIChebConv(nhid, nhid, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))  # no BN here is better
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiSAGE_2_nhid(nn.Module):
    def __init__(self, m, input_dim,  ncls, args):
        super(DiSAGE_2_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads
        K = args.K

        self.Conv = nn.Conv1d(nhid, ncls, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, nhid)
            self.conv2 = DiSAGEConv(nhid, ncls)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, ncls)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(input_dim, head_dim, heads=head)
            self.conv2 = GATConv_Qin(nhid, ncls//num_head, heads=head)
            self.batch_norm1 = nn.BatchNorm1d(head_dim)
            self.batch_norm2 = nn.BatchNorm1d(head_dim)
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, nhid, K)
            self.conv2 = DIChebConv(nhid, ncls, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):  # this is original DiGCN
        x = F.relu(self.conv1(x, edge_index, edge_weight))   # no BN here is better
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

class DiSAGE_xBN_nhid0(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args, layer=3, head=8):
        super(DiSAGE_xBN_nhid0, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads
        K = args.K

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, nhid)
            self.conv2 = DiSAGEConv(nhid, nhid)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, nhid)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(input_dim, head_dim,  heads=head)
            self.conv2 = GATConv_Qin(nhid, head_dim,  heads=head)
            self.convx = nn.ModuleList([GATConv_Qin(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)

        return x

class DiSAGE_xBN_nhid(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiSAGE_xBN_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
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

            self.conv1 = GATConv_Qin(input_dim, head_dim,  heads=head)
            self.conv2 = GATConv_Qin(nhid, head_dim,  heads=head)
            self.convx = nn.ModuleList([GATConv_Qin(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
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

class DiSAGE_1BN_nhid(nn.Module):
    def __init__(self, m, in_dim, out_dim,  args):
        super(DiSAGE_1BN_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads

        if m == 'S':
            self.conv1 = DiSAGEConv(in_dim, nhid)
        elif m == 'G':
            self.conv1 = DIGCNConv(in_dim, nhid)
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(in_dim, head_dim, heads=head)
            # self.conv1 = DiGATConv(in_dim, nhid, nhid)     # little difference from GATConv_Qin
        elif m == 'C':
            self.conv1 = DIChebConv(in_dim, nhid, args.K)
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index, edge_weight):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.batch_norm1(self.conv1(x, edge_index, edge_weight))
        x = self.conv1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x


class DiSAGE_x_nhid(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiSAGE_x_nhid, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        self.layer = args.layer
        head = args.heads
        K = args.K
        if self.layer >1 :
            n_change = nhid
        else:
            n_change = out_dim
        self.jk = args.jk
        out1 = out_dim  if layer == 1 else nhid

        if self.jk is not None:
            in_dim_jk = nhid * self.layer if self.jk == "cat" else nhid
            # self.lin = Linear(input_dim, out_dim)
            self.lin = Linear(in_dim_jk, nhid)
            self.jump = JumpingKnowledge(mode=self.jk, channels=nhid, num_layers=out_dim)

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, n_change)
            self.conv2 = DiSAGEConv(nhid, out1)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, n_change)
            self.conv2 = DIGCNConv(nhid, out1)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, n_change, K)
            self.conv2 = DIChebConv(nhid, out1, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(input_dim, n_change // num_head,  heads=head)
            self.conv2 = GATConv_Qin(nhid, out1//num_head,  heads=head)
            self.convx = nn.ModuleList([GATConv_Qin(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        if self.jk is not None:
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

        if self.jk is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return x   # log softmax operation, has the same dimension


class DiSAGE_xBN_nhid_BN(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiSAGE_xBN_nhid_BN, self).__init__()
        self.dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        head = args.heads
        K = args.K

        if m == 'S':
            self.conv1 = DiSAGEConv(input_dim, nhid)
            self.conv2 = DiSAGEConv(nhid, out_dim)
            self.convx = nn.ModuleList([DiSAGEConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'G':
            self.conv1 = DIGCNConv(input_dim, nhid)
            self.conv2 = DIGCNConv(nhid, out_dim)
            self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer - 2)])
        elif m == 'C':
            self.conv1 = DIChebConv(input_dim, nhid, K)
            self.conv2 = DIChebConv(nhid, out_dim, K)
            self.convx = nn.ModuleList([DIChebConv(nhid, nhid, K) for _ in range(layer - 2)])
        elif m == 'A':
            num_head = 1
            head_dim = nhid // num_head

            self.conv1 = GATConv_Qin(input_dim, head_dim,  heads=head)
            self.conv2 = GATConv_Qin(nhid, out_dim//num_head,  heads=head)
            self.convx = nn.ModuleList([GATConv_Qin(nhid, head_dim,  heads=head) for _ in range(layer - 2)])
        else:
            raise ValueError(f"Model '{m}' not implemented")

        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(iter_layer(x, edge_index, edge_weight))
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))      # good for telegram
        # x = self.conv2(x, edge_index, edge_weight)
        return x   # log softmax operation, has the same dimension

class DiG_SimpleXBN_nhid(torch.nn.Module):
    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super(DiG_SimpleXBN_nhid, self).__init__()
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

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)

        return x

class DiG_SimpleXBN_nhid_Pan(torch.nn.Module):
    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super(DiG_SimpleXBN_nhid_Pan, self).__init__()
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

    #     for iter_layer in self.convx:
    #         x = F.dropout(x, self.dropout, training=self.training)
    #         x = iter_layer(x, edge_index, edge_weight)
    #         x = F.relu(x)
    #
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.conv2(x, edge_index, edge_weight)
    #     x = self.batch_norm2(x)
    #
    #     x = x.unsqueeze(0)
    #     x = x.permute((0, 2, 1))
    #     x = self.Conv(x)
    #     x = x.permute((0, 2, 1)).squeeze()
    #
    #     x = F.dropout(x, self.dropout, training=self.training)
    #
    #     return x
    # def forward(self, x, edge_index):
    #     out_list = []
    #     for i in range(self.num_layers):
    #         x = self.convs[i](x, edge_index)
    #         if i < self.num_layers - 1:
    #             x = F.relu(x)
    #         out_list.append(F.log_softmax(x, dim=1))
    #     return out_list

class DiG_Simple2BN(nn.Module):
    def __init__(self, input_dim, nhid, out_dim, dropout, layer=2):
        super(DiG_Simple2BN, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, out_dim)
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        # type1
        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))      # no BN here is better
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)


        return x
class DiG_SimpleXBN(torch.nn.Module):
    def __init__(self, input_dim,  nhid, out_dim, dropout, layer=3):
        super(DiG_SimpleXBN, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, nhid)
        self.conv2 = DIGCNConv(nhid, out_dim)
        self.convx = nn.ModuleList([DIGCNConv(nhid, nhid) for _ in range(layer-2)])
        # self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        # type1
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

        # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_weight)))

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            # x = F.relu(self.batch_norm3(iter_layer(x, edge_index, edge_weight)))
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.batch_norm2(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

def create_DiGSimple(m, nfeat, nclass, args):
    if args.layer == 1:
        model = DiG_Simple1BN(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiG_Simple2BN(m, nfeat, nclass, args)
    else:
        model = DiG_SimpleXBN(m, nfeat, nclass, args)
    return model

def create_DiGSimple_nhid(m, nfeat, nclass, args):
    if args.layer == 1:
        model = DiG_Simple1BN_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiG_Simple2BN_nhid(m, nfeat, nclass, args)
    else:
        model = DiG_SimpleXBN_nhid(m, nfeat, nclass, args)
    return model

def create_DiSAGESimple_nhid0(m, nfeat, nclass, args):
    if args.layer == 1:
        model = DiSAGE_1BN_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiSAGE_2BN_nhid(m, nfeat, nclass, args)
    else:
        model = DiSAGE_xBN_nhid(m, nfeat, nclass, args)
    return model

def create_DiSAGESimple_nhid(m, nfeat, n_cls, args):
    # if args.layer == 1:
    #     model = DiSAGE_1_nhid(m, nfeat,  n_cls, args)
    # elif args.layer == 2:
    #     model = DiSAGE_2_nhid(m, nfeat,  n_cls, args)
    # else:
    model = DiSAGE_x_nhid(m, nfeat,  n_cls, args)
    return model

def create_DiGSimple_batch_nhid(m, nfeat, nclass, args):

    if args.layer == 1:
        model = DiG_Simple1BN_batch_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiG_Simple2BN_batch_nhid(m, nfeat, nclass, args)
    else:
        model = DiG_SimpleXBN_batch_nhid(m, nfeat, nclass, args)
    return model

class Di_IB_1_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, n_cls, args):
        super(Di_IB_1_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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

class Di_IB_1BN_nhid0(torch.nn.Module):
    def __init__(self, m, input_dim,  n_cls, args):
        super(Di_IB_1BN_nhid0, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.Conv = nn.Conv1d(nhid, n_cls, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(n_cls)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_nhid(torch.nn.Module):
    def __init__(self, m, input_dim,  n_cls, args):
        super(DiGCN_IB_1BN_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(n_cls)
        self.Conv = nn.Conv1d(nhid, n_cls, kernel_size=1)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_nhid_para(torch.nn.Module):
    def __init__(self, input_dim, nhid, n_cls, dropout=0.5, layer=1):
        super(DiGCN_IB_1BN_nhid_para, self).__init__()
        self.ib1 = InceptionBlock_Qinlist(input_dim, nhid)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib1
        self._dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid, n_cls, kernel_size=1)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()
        self.coefs = self.coef1

    def forward(self, features, edge_index_tuple, edge_weight_tuple):   # TODO BN and dropout change position and dimension
        x = features
        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef1[i] * x_list[i]

        x = self.batch_norm1(x)     # better
        # x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        # x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        return x

class DiGCN_IB_1BN_batch(torch.nn.Module):
    '''
    for large dataset, using small batches not the whole graph
    '''
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_1BN_batch, self).__init__()
        self.ib1 = InceptionBlock_Di(m, input_dim, out_dim, args)
        self._dropout = args.dropout
        self.batch_size = args.batch_size
        self.batch_norm1 = nn.BatchNorm1d( out_dim)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_1BN_batch_nhid(torch.nn.Module):
    '''
    for large dataset, using small batches not the whole graph
    '''

    def __init__(self, m, input_dim,   out_dim, args):
        super(DiGCN_IB_1BN_batch_nhid, self).__init__()

        self._dropout = args.dropout
        self.batch_size = args.batch_size
        nhid = args.feat_dim

        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm = nn.BatchNorm1d(nhid)
        self.Conv1 = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.reg_params = []
        self.non_reg_params = self.ib1.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm(x_batch)

            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()

            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_2BN_nhid(torch.nn.Module):
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_2BN_nhid, self).__init__()
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters())+list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class Di_IB_2_nhid(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super(Di_IB_2_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim

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

class Di_IB_2BN_nhid0(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super(Di_IB_2BN_nhid0, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters())+list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)
        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_nhid_para(torch.nn.Module):
    def __init__(self, input_dim, out_dim, args):
        super(DiGCN_IB_2BN_nhid_para, self).__init__()
        self.ib1 = InceptionBlock_Qinlist(input_dim, nhid)
        self.ib2 = InceptionBlock_Qinlist(nhid, nhid)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib2
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.reg_params = list(self.ib1.parameters())+list(self.ib2.parameters())
        self.non_reg_params = []
        # self.coefs = [self.coef1, self.coef2] + self.coefx
        # self.coefs = self.coef1+ self.coef2       # wrong
        self.coefs = list(self.coef1) + list(self.coef2)
        # self.coefs = nn.ParameterList()       # this is OK too, but above list is simpler
        # self.coefs.extend(self.coef1)
        # self.coefs.extend(self.coef2)

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef1[i] * x_list[i]
        x = self.batch_norm1(x)

        x_list = self.ib2(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef2[i] * x_list[i]

        x = self.batch_norm2(x)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2BN_batch(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super(DiGCN_IB_2BN_batch, self).__init__()
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid,  out_dim)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d( out_dim)
        self.batch_size = args.batch_size

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        # batch_size = 1000  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x


class DiGCN_IB_2BN_batch_nhid(torch.nn.Module):
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_2BN_batch_nhid, self).__init__()
        self._dropout = args.dropout
        self.batch_size = args.batch_size
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        # batch_size = 1000  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm2(x_batch)

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x

class DiGCN_IB_2BN_samebatch(torch.nn.Module):
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_2BN_samebatch, self).__init__()
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid,  out_dim)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d( out_dim)
        self.batch_size = args.batch_size

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batatch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batatch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        return x

class DiGCN_IB_1BN_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_1BN_Sym, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, out_dim, args)
        self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, out_dim, bias=False)
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
        # x= x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv(x)
        # x = x.permute((0, 2, 1))
        # x = x.squeeze(0)
        x = self.batch_norm1(x)     # keep it is better performance

        x = F.dropout(x, p=self._dropout, training=self.training)   # only dropout during training   keep this is better
        return x
class DiGCN_IB_1BN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim, out_dim, args):
        super(DiGCN_IB_1BN_Sym_nhid, self).__init__()
        nhid = args.feat_dim
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
        super(DiGIB_1BN_Sym_nhid_para, self).__init__()
        nhid = args.feat_dim
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

class DiGCN_IB_1BN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion

    '''
    def __init__(self,m,  input_dim,  out_dim, args, batch_size=1024):
        super(DiGCN_IB_1BN_Sym_batch_nhid, self).__init__()
        nhid = args.feat_dim
        self.batch_size = args.batch_size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                    (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                       (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)



            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)
        x_batch = DiGx + symx
        x_batch = x_batch.unsqueeze(0)  # ?
        x_batch = x_batch.permute((0, 2, 1))
        x_batch = self.Conv(x_batch)
        x_batch = x_batch.permute((0, 2, 1))
        x_batch = x_batch.squeeze(0)
        x = self.batch_norm2(x_batch)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_1BN_Sym_batchConvOut(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''

    def __init__(self,m,  input_dim,  out_dim, args):
        super(DiGCN_IB_1BN_Sym_batchConvOut, self).__init__()
        self.batch_size = args.batch_size
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        # self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        # x = self.batch_norm1(x)        # without this is better performance
        x = x.unsqueeze(0)  # ?
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = self.batch_norm2(x)  # ?

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2BN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index1 confusion

    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2BN_Sym_batch_nhid, self).__init__()
        nhid = args.feat_dim
        self.batch_size = args.batch_size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        # x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)


        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(x_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = DiGx + symx
        x = self.batch_norm2(x)
        x = x.unsqueeze(0)  # ?
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)
        x = F.dropout(x, p=self._dropout, training=self.training)       # keep is better performance
        return x
class DiGCN_IB_2BN_Sym(torch.nn.Module):
    def __init__(self,m,  input_dim,  out_dim, args):
        super(DiGCN_IB_2BN_Sym, self).__init__()
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

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
        # symx = self.batch_norm1(symx)
        # symx = F.relu(symx)

        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x + symx
        # x = self.batch_norm1(x)
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

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x
class DiGIB_2BN_Sym_nhid_para(torch.nn.Module):
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGIB_2BN_Sym_nhid_para, self).__init__()
        nhid = args.feat_dim
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
        super(DiGCN_IB_2BN_Sym_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        
        
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

class DiGCN_IB_XBN_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self,m,  input_dim,  out_dim, args):
        super(DiGCN_IB_XBN_Sym, self).__init__()
        nhid = args.feat_dim
        self._dropout = args.dropout
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
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

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_XBN_Sym_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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
        super(DiGIB_XBN_Sym_nhid_para, self).__init__()
        nhid = args.feat_dim
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
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

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

class DiGCN_IB_XBN_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion

    '''
    def __init__(self, m,  input_dim, out_dim, args):
        super(DiGCN_IB_XBN_Sym_batch_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        self.batch_size = args.batch_size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer-2)])
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

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            # symx1 = self.gconv(symx, edge_index)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)
        x = DiGx + symx
        x = self.batch_norm1(x)
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_layerlinx,  iter_layeribx in zip(self.linx, self.ibx):
            outputs = []
            sym_outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                        (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                edge_index_batch = edge_index[:, mask] - start_idx

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                           (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                edge_in_batch = edge_in[:, mask_in]
                edge_in_batch = edge_in_batch - start_idx
                in_w_batch = in_w[mask_in]
                mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                            (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                edge_out_batch = edge_out[:, mask_out]
                edge_out_batch = edge_out_batch - start_idx
                out_w_batch = out_w[mask_out]

                # Forward pass for the current batch
                symx = iter_layerlinx(batch_x)
                # symx1 = self.gconv(symx, edge_index)
                symx1 = self.gconv(symx, edge_index_batch)
                symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                symx_batch = symx1 + symx2 + symx3
                sym_outputs.append(symx_batch)

                x_batch = iter_layeribx(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                
                # x_batch = self.batch_norm1(x_batch)       # without is better
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            DiGx = torch.cat(outputs, dim=0)
            symx = torch.cat(sym_outputs, dim=0)
            x = DiGx + symx
            # x = self.batch_normx(x)
            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        outputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        DiGx = torch.cat(outputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)
        x = self.batch_norm2(x)

        x = DiGx + symx
        x = x.unsqueeze(0)  # ?
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
        super(DiGCN_IB_2BN_SymCat_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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

class DiGCN_IB_2BN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    only has nhid version
    '''
    def __init__(self, m, input_dim, nhid, out_dim, args):
        super(DiGCN_IB_2BN_SymCat_batch_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        layer = args.layer
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

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
        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)


        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm2(symx_batch)

            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm2(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv2(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        # x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_XBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_XBN_SymCat_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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
        super(DiGCN_IB_XBN_SymCat_1ibx_nhid, self).__init__()
        self.layer= args.layer
        self._dropout = args.dropout
        nhid = args.feat_dim
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


class DiGCN_IB_XBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    only has nhid version
    '''
    def __init__(self,m, input_dim,  out_dim, args):
        super(DiGCN_IB_XBN_SymCat_batch_nhid, self).__init__()
        self.batch_size = args.batch_size  # Define your batch size
        nhid = args.feat_dim
        self._dropout = args.dropout
        
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm2 = nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        DiGx = []
        Symx =[]
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            Symx.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx.append(DiGx_batch)

        Symx = torch.cat(Symx, dim=0)
        DiGx = torch.cat(DiGx, dim=0)
        x = torch.cat((DiGx, Symx), dim=-1)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        for iter_lin in self.linx:
            Symx_x = []
            DiGx_x = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                        (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                edge_index_batch = edge_index[:, mask] - start_idx

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                           (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                edge_in_batch = edge_in[:, mask_in]
                edge_in_batch = edge_in_batch - start_idx
                in_w_batch = in_w[mask_in]
                mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                            (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                edge_out_batch = edge_out[:, mask_out]
                edge_out_batch = edge_out_batch - start_idx
                out_w_batch = out_w[mask_out]

                symx = iter_lin(batch_x)
                symx1 = self.gconv(symx, edge_index_batch)
                symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                symx_batch = symx1 + symx2 + symx3
                Symx_x.append(symx_batch)

                DiGx_batch = self.ibx(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                DiGx_x.append(DiGx_batch)
            Symx = torch.cat(Symx_x, dim=0)
            DiGx = torch.cat(DiGx_x, dim=0)
            x = torch.cat((DiGx, Symx), dim=-1)

            x = x.unsqueeze(0)
            x = x.permute((0, 2, 1))
            x = self.Convx(x)
            x = x.permute((0, 2, 1)).squeeze()

            x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

        Symx_2 = []
        DiGx_2 = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            Symx_2.append(symx_batch)

            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_2.append(DiGx_batch)

        Symx = torch.cat(Symx_2, dim=0)
        DiGx = torch.cat(DiGx_2, dim=0)
        x = torch.cat((DiGx, Symx), dim=-1)
        x = self.batch_norm2(x)     # keep this is better

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv2(x)
        x = x.permute((0, 2, 1)).squeeze()

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index comfusion
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_1BN_SymCat_nhid, self).__init__()
        nhid = args.feat_dim
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

class DiGCN_IB_1BN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index comfusion

    '''
    def __init__(self, m, input_dim,  out_dim, args,  batch_size=1024):
        super(DiGCN_IB_1BN_SymCat_batch_nhid, self).__init__()
        nhid = args.feat_dim
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)
        # self.batch_normx= nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm2(x_batch)

            x_batch = self.batch_norm2(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)

            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv1(x_batch)
            x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = F.relu(x_batch)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        # x = self.batch_normx(x)

        # x = x.unsqueeze(0)
        # x = x.permute((0, 2, 1))
        # x = self.Conv1(x)  # with this block or without, almost the same result
        # x = x.permute((0, 2, 1)).squeeze()
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        return x

class DiGCN_IB_1BN_SymCat_batchConvOut(torch.nn.Module):
    '''
    revised for edge_index comfusion
    move all conv outside of batch
    '''
    def __init__(self, m, input_dim,  out_dim, args,  batch_size=1024):
        super(DiGCN_IB_1BN_SymCat_batchConvOut, self).__init__()
        nhid = args.feat_dim
        self._dropout = args.dropout
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2= nn.BatchNorm1d(2*nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            symx_batch = self.batch_norm1(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx_batch = self.batch_norm1(DiGx_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)
            # x_batch = x_batch.unsqueeze(0)
            # x_batch = x_batch.permute((0, 2, 1))
            # x_batch = self.Conv1(x_batch)
            # x_batch = x_batch.permute((0, 2, 1)).squeeze()
            # x_batch = self.batch_norm1(x_batch)
            # x_batch = F.relu(x_batch)
            # if self._dropout > 0:
            #     x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)     # is this necessary?
        return x

class DiGCN_IB_2MixBN_SymCat_nhid(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2MixBN_SymCat_nhid, self).__init__()
        nhid = args.feat_dim
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


class DiGCN_IB_2MixBN_SymCat(torch.nn.Module):
    '''
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2MixBN_SymCat, self).__init__()
        nhid = args.feat_dim
        self._dropout = args.dropout
        
        self.ib1 = InceptionBlock_Di(m, input_dim, out_dim, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * nhid, out_dim, kernel_size=1)

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
        x = self.batch_norm1(x)  # keep both it and the endBN is better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        x = self.ib2(x, edge_index_tuple, edge_weight_tuple)

        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self,m,  input_dim,  out_dim, args, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_batch, self).__init__()
        nhid = args.feat_dim
        self._dropout = args.dropout
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, out_dim)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        DiGx = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)
            
            DiGx_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            DiGx.append(DiGx_batch)
        x= torch.cat(DiGx, dim=0)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x


class DiGCN_IB_2MixBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, m, input_dim,  out_dim, args, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_batch_nhid, self).__init__()
        nhid = args.feat_dim
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, 2*nhid, 2*nhid)
        # self.ib2 = InceptionBlock_Qin(nhid, nhid)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)
        self.Conv = nn.Conv1d(2 * nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better

            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        DiGx = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm2(x_batch)
            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            DiGx.append(x_batch)
        x = torch.cat(DiGx, dim=0)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2MixBN_SymCat_Sym, self).__init__()
        nhid = args.feat_dim
        self._dropout = args.dropout
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

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
        return x

class DiGCN_IB_2MixBN_SymCat_Sym_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    nhid
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2MixBN_SymCat_Sym_nhid, self).__init__()
        nhid = args.feat_dim
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

class DiGCN_IB_2MixBN_SymCat_Sym_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, m, input_dim,  out_dim, args, batch_size=1024):
        super(DiGCN_IB_2MixBN_SymCat_Sym_batch, self).__init__()
        nhid = args.feat_dim
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            # x_batch = self.batch_norm1(x_batch)
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in] - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid, self).__init__()
        self.batch_size = args.batch_size  # Define your batch size
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Convx = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, nhid))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters()) + list(self.ib2.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask]- start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
        # x = torch.cat(outputs, dim=0)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in] - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        # x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Convx(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()

        return x


class DiGCN_IB_3MixBN_SymCat_Sym_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''

    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_3MixBN_SymCat_Sym_batch_nhid, self).__init__()
        self.layer = args.layer
        nhid = args.feat_dim
        self.batch_size = args.batch_size  # Define your batch size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        # self.batch_norm = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.lin2_ = torch.nn.Linear(2*nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer is cat(Sym, DiGib),
        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better

            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # second layer --addSym
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2_(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)

        # more than 3 layer
        DiGx = []
        if self.layer > 3:
            for iter_layer in self.ibx:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                x_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

                x_batch = F.relu(x_batch)
                if self._dropout > 0:
                    x_batch = F.dropout(x_batch, self._dropout, training=self.training)
                DiGx.append(x_batch)

            x = torch.cat(DiGx, dim=0)
            x = F.dropout(x, p=self._dropout, training=self.training)
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()


        x = F.dropout(x, p=self._dropout, training=self.training)
        return x
class DiGCN_IB_3MixBN_SymCat_Sym_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_3MixBN_SymCat_Sym_batch, self).__init__()
        self.layer = args.layer
        self.batch_size = args.batch_size  # Define your batch size
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2*nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2*out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])


        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3

            DiGx_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = torch.cat((DiGx_batch, symx_batch), dim=-1)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            
            x_batch = F.relu(x_batch)
            if self._dropout > 0:
                x_batch = F.dropout(x_batch, self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # second layer --addSym
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2_(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)

        # more than 3 layer
        DiGx =[]
        if self.layer > 3:
            for iter_layer in self.ibx:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                x_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                

                x_batch = F.relu(x_batch)
                if self._dropout > 0:
                    x_batch = F.dropout(x_batch, self._dropout, training=self.training)
                DiGx.append(x_batch)

        x = torch.cat(DiGx, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin2(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.batch_norm2(x)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_3MixBN_SymCat(torch.nn.Module):
        '''
        revised for edge_index confusion
        first layer is cat(Sym, DiGib), second layer is DiGib
        '''
        def __init__(self, m, input_dim,  out_dim, args):
            super(DiGCN_IB_3MixBN_SymCat, self).__init__()
            self.layer = args.layer
            nhid = args.feat_dim
            self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
            self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
            self._dropout = args.dropout
            self.batch_norm1 = nn.BatchNorm1d(nhid)
            self.batch_norm2 = nn.BatchNorm1d(nhid)
            self.batch_norm3 = nn.BatchNorm1d(out_dim)

            self.gconv = DGCNConv()
            self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
            self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

            self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
            self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
            # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
            if self.layer >3:
                self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

            self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
            self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

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
            x = self.batch_norm1(x)     # with this is a bit better
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

            x = self.batch_norm3(x)       # keep this is better performance
            # x = F.relu(x)
            if self._dropout > 0:
                x = F.dropout(x, self._dropout, training=self.training)

            return x


class DiGCN_IB_3MixBN_SymCat_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_3MixBN_SymCat_nhid, self).__init__()
        self.layer = args.layer
        nhid = args.feat_dim
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
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

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


class DiGCN_IB_3MixBN_SymCat_batch_nhid(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''

    def __init__(self, m, input_dim,  out_dim, args):
        super(DiGCN_IB_3MixBN_SymCat_batch_nhid, self).__init__()
        self.batch_size = args.batch_size
        self.layer = args.layer
        layer = args.layer
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        # self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, nhid, bias=False)
        self.lin3 = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---cat(Sym DiG)
        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv1(x)  # with this block or without, almost the same result
        x = x.permute((0, 2, 1)).squeeze()
        x = self.batch_norm1(x)  # with this is a bit better
        x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        # second layer --DiGib
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                sym_outputs = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    batch_x = x[start_idx:end_idx]

                    mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                    edge_index_batch = edge_index[:, mask] - start_idx

                    DiGedge_indexi_batch = ()
                    DiGedge_weighti_batch = ()
                    for i in range(len(edge_index_tuple)):
                        mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                                  (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                        edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                        edge_weighti_batch = edge_weight_tuple[i][mask_i]

                        DiGedge_indexi_batch += (edge_indexi_batch,)
                        DiGedge_weighti_batch += (edge_weighti_batch,)

                    mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                               (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                    edge_in_batch = edge_in[:, mask_in]
                    edge_in_batch = edge_in_batch - start_idx
                    in_w_batch = in_w[mask_in]
                    mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                                (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                    edge_out_batch = edge_out[:, mask_out]
                    edge_out_batch = edge_out_batch - start_idx
                    out_w_batch = out_w[mask_out]

                    # Forward pass for the current batch
                    symx = iter_layer(batch_x)
                    symx1 = self.gconv(symx, edge_index_batch)
                    symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                    symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                    symx_batch = symx1 + symx2 + symx3
                    sym_outputs.append(symx_batch)

                x = torch.cat(sym_outputs, dim=0)

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin3(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            x_batch = symx1 + symx2 + symx3
            x_batch = self.batch_norm3(x_batch)  # keep this is better performance

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            sym_outputs.append(x_batch)

        x = torch.cat(sym_outputs, dim=0)

        # x = F.relu(x)
        if self._dropout > 0:
            x = F.dropout(x, self._dropout, training=self.training)

        return x
class DiGCN_IB_3MixBN_SymCat_batch(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is DiGib
    '''
    def __init__(self, m, input_dim,  out_dim, args, batch_size=1000):
        super(DiGCN_IB_3MixBN_SymCat_batch, self).__init__()
        self.batch_size = args.batch_size
        self.layer = args.layer
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        # self.linx = torch.nn.Linear(nhid, nhid,nhid, nhid, bias=False)
        if self.layer > 3:
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.reg_params = list(self.ib1.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_index_tuple, edge_weight_tuple):
        # first layer---Sym + DiG

        batch_size = self.batch_size  # Define your batch size
        num_samples = x.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        DiGoutputs = []
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            # x_batch = self.batch_norm1(x_batch)       # without is better
            DiGx_batch = F.dropout(DiGx_batch, p=self._dropout, training=self.training)
            DiGoutputs.append(DiGx_batch)

        DiGx = torch.cat(DiGoutputs, dim=0)
        symx = torch.cat(sym_outputs, dim=0)

        x = torch.cat((DiGx, symx), dim=-1)
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
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            # x_batch = self.batch_norm1(x_batch)       # without is better
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)

        # x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        # more than 3 layer
        if self.layer > 3:
            for iter_layer in self.linx:
                sym_outputs = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    batch_x = x[start_idx:end_idx]

                    mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
                    edge_index_batch = edge_index[:, mask] - start_idx

                    DiGedge_indexi_batch = ()
                    DiGedge_weighti_batch = ()
                    for i in range(len(edge_index_tuple)):
                        mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                                  (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                        edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                        edge_weighti_batch = edge_weight_tuple[i][mask_i]
        
                        DiGedge_indexi_batch += (edge_indexi_batch,)
                        DiGedge_weighti_batch += (edge_weighti_batch,)

                    mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                               (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
                    edge_in_batch = edge_in[:, mask_in]
                    edge_in_batch = edge_in_batch - start_idx
                    in_w_batch = in_w[mask_in]
                    mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                                (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
                    edge_out_batch = edge_out[:, mask_out]
                    edge_out_batch = edge_out_batch - start_idx
                    out_w_batch = out_w[mask_out]

                    # Forward pass for the current batch
                    symx = iter_layer(batch_x)
                    symx1 = self.gconv(symx, edge_index_batch)
                    symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
                    symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
                    symx_batch = symx1 + symx2 + symx3
                    sym_outputs.append(symx_batch)

                x = torch.cat(sym_outputs, dim=0)

                # x = self.batch_norm2(x)  # without this is better performance
                x = F.relu(x)
                if self._dropout > 0:
                    x = F.dropout(x, self._dropout, training=self.training)

        # third layer
        sym_outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) &
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))
            edge_index_batch = edge_index[:, mask] - start_idx

            mask_in = ((edge_in[0] >= start_idx) & (edge_in[0] < end_idx) &
                       (edge_in[1] >= start_idx) & (edge_in[1] < end_idx))
            edge_in_batch = edge_in[:, mask_in]
            edge_in_batch = edge_in_batch - start_idx
            in_w_batch = in_w[mask_in]
            mask_out = ((edge_out[0] >= start_idx) & (edge_out[0] < end_idx) &
                        (edge_out[1] >= start_idx) & (edge_out[1] < end_idx))
            edge_out_batch = edge_out[:, mask_out]
            edge_out_batch = edge_out_batch - start_idx
            out_w_batch = out_w[mask_out]

            # Forward pass for the current batch
            symx = self.lin1(batch_x)
            symx1 = self.gconv(symx, edge_index_batch)
            symx2 = self.gconv(symx, edge_in_batch, in_w_batch)
            symx3 = self.gconv(symx, edge_out_batch, out_w_batch)
            symx_batch = symx1 + symx2 + symx3
            sym_outputs.append(symx_batch)

        x = torch.cat(sym_outputs, dim=0)

        x = self.batch_norm3(x)  # keep this is better performance
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
        super(DiGCN_IB_3MixBN_SymCat_Sym_nhid, self).__init__()
        self.layer = args.layer
        nhid = args.feat_dim
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
            self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

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
class DiGCN_IB_3MixBN_SymCat_Sym(torch.nn.Module):
    '''
    revised for edge_index confusion
    first layer is cat(Sym, DiGib), second layer is addSym
    '''
    def __init__(self,m,  input_dim,  out_dim, args):
        super(DiGCN_IB_3MixBN_SymCat_Sym, self).__init__()
        self.layer = args.layer
        nhid = args.feat_dim
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid,args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self._dropout = args.dropout
        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(out_dim)

        self.gconv = DGCNConv()
        self.Conv1 = nn.Conv1d(2 * nhid, nhid, kernel_size=1)
        self.Conv2 = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.lin2_ = torch.nn.Linear(nhid, nhid, bias=False)
        if self.layer > 3:
            self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 3)])
            self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 3)])

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

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
        #
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

        return x


class DiGCN_IB_XBN_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super(DiGCN_IB_XBN_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.layer = args.layer
        layer = args.layer
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid,nhid) for _ in range(layer-2)])

        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm1(x)

        for iter_layer in self.ibx:
            x = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
            x = self.batch_norm3(x)

        x = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
        x = self.batch_norm2(x)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1))
        x = x.squeeze(0)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class Di_IB_XBN_nhid_ConV(torch.nn.Module):
    def __init__(self, m, input_dim,   out_dim, args):
        super(Di_IB_XBN_nhid_ConV, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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
        super(Di_IB_XBN_nhid_ConV_JK, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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
        super(Di_IB_X_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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

class Si_IB_X_nhid(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super(Si_IB_X_nhid, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        self.layer = args.layer

        if self.layer == 1:
            self.ib1 = InceptionBlock_Si(m, input_dim, out_dim, args)
        else:
            self.ib1 = InceptionBlock_Si(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Si(m, nhid, out_dim, args)
        if self.layer > 2:
            self.ibx = nn.ModuleList([InceptionBlock_Si(m, nhid, nhid, args) for _ in range(self.layer - 2)])

        # self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        # self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index, edge_weight):
        x = features
        x = self.ib1(x, edge_index, edge_weight)
        x = F.dropout(x, p=self._dropout, training=self.training)
        if self.layer == 1:
            return x

        if self.layer > 2:
            for iter_layer in self.ibx:
                x = F.dropout(x, p=self._dropout, training=self.training)
                x = iter_layer(x, edge_index, edge_weight)

        x = self.ib2(x, edge_index, edge_weight)

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_nhid_para(torch.nn.Module):
    def __init__(self, m, num_features, out_dim, args):
        super(DiGCN_IB_XBN_nhid_para, self).__init__()
        nhid = args.feat_dim
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
        super(DiGCN_IB_X_nhid_para_Jk, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
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

class DiGCN_IB_X_nhid_para(torch.nn.Module):
    def __init__(self, m, input_dim, out_dim, args):
        super(DiGCN_IB_X_nhid_para, self).__init__()
        self._dropout = args.dropout
        nhid = args.feat_dim
        if args.layer==1:
            self.ib1 = InceptionBlock_Di_list(m, input_dim, out_dim, args)
        else:
            self.ib1 = InceptionBlock_Di_list(m, input_dim, nhid, args)

        self.ib2 = InceptionBlock_Di_list(m, nhid, out_dim, args)
        self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib1
        self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)])  # coef for ib2
        self._dropout = args.dropout

        self.layer = args.layer
        layer = args.layer
        self.ibx=nn.ModuleList([InceptionBlock_Di_list(m, nhid, nhid, args) for _ in range(layer-2)])
        self.coefx = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(20)]) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()
        self.coefs = list(self.coef1)+list(self.coef2)+list(self.coefx.parameters())
        # self.coefs = [self.coef1, self.coef2, self.coefx]     # wrong

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef1[i] * x_list[i]
        # x = self.batch_norm1(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        # x = F.relu(x)
        if self.layer == 1:
            return x

        if self.layer > 2:
            for iter_layer, iter_coef in zip(self.ibx, self.coefx):
                x_list = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
                x = x_list[0]
                for i in range(1, len(x_list)):
                    x += iter_coef[i] * x_list[i]
                x = F.dropout(x, p=self._dropout, training=self.training)

        x_list = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
        x = x_list[0]
        for i in range(1, len(x_list)):
            x += self.coef2[i] * x_list[i]

        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

# class DiGCN_IB_XBN_nhid_para(torch.nn.Module):
#     def __init__(self, num_features, hidden,  out_dim, dropout=0.5, layer=2):
#         super(DiGCN_IB_XBN_nhid_para, self).__init__()
        self.ib1 = InceptionBlock_Qinlist(num_features, hidden)
#         self.ib2 = InceptionBlock_Qinlist(hidden, hidden)
#         self.coef1 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(5)])  # coef for ib1
#         self.coef2 = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(5)])  # coef for ib2
#         self._dropout = dropout
#         self.Conv = nn.Conv1d(hidden,  out_dim, kernel_size=1)
#
#         self.batch_norm1 = nn.BatchNorm1d(hidden)
#         self.batch_norm2 = nn.BatchNorm1d(hidden)
#         self.batch_norm3 = nn.BatchNorm1d(hidden)
#
#         self.layer = layer
#         self.ibx=nn.ModuleList([InceptionBlock_Qinlist(hidden,hidden) for _ in range(layer-2)])
#         self.coefx = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(5)]) for _ in range(layer - 2)])
#
#         self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
#         self.non_reg_params = self.ib2.parameters()
#         self.coefs = list(self.coef1)+list(self.coef2)+list(self.coefx.parameters())
#         # self.coefs = [self.coef1, self.coef2, self.coefx]     # wrong
#
#     def forward(self, features, edge_index_tuple, edge_weight_tuple):
#         x = features
#         x_list = self.ib1(x, edge_index_tuple, edge_weight_tuple)
#         x = x_list[0]
#         for i in range(1, len(x_list)):
#             x += self.coef1[i] * x_list[i]
#         # x = self.batch_norm1(x)
#
#         for iter_layer, iter_coef in zip(self.ibx, self.coefx):
#             x_list = iter_layer(x,  edge_index_tuple, edge_weight_tuple)
#             x = x_list[0]
#             for i in range(1, len(x_list)):
#                 x += iter_coef[i] * x_list[i]
#             # x = self.batch_norm3(x)
#
#         x_list = self.ib2(x,  edge_index_tuple, edge_weight_tuple)
#         x = x_list[0]
#         for i in range(1, len(x_list)):
#             x += self.coef2[i] * x_list[i]
#
#         x = self.batch_norm2(x)
#         x = x.unsqueeze(0)
#         x = x.permute((0, 2, 1))
#         x = self.Conv(x)
#         x = x.permute((0, 2, 1))
#         x = x.squeeze(0)
#
#         x = F.dropout(x, p=self._dropout, training=self.training)
#         return x


class DiGCN_IB_XBN_batch_nhid(torch.nn.Module):
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_XBN_batch_nhid, self).__init__()
        nhid = args.feat_dim
        layer = args.layer
        self._dropout = args.dropout
        self.batch_size = args.batch_size
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid, nhid, args)
        self.Conv = nn.Conv1d(nhid,  out_dim, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.layer = args.layer
        self.ibx = nn.ModuleList([InceptionBlock_Di(m, nhid, nhid, args) for _ in range(layer - 2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            DiGx_batch = self.ib1(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

            x_batch = self.batch_norm1(DiGx_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        for iter_layer in self.ibx:
            num_samples = features.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                # Forward pass for the current batch
                DiGx_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)

                x_batch = self.batch_norm3(DiGx_batch)
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            x = torch.cat(outputs, dim=0)

        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            x_batch = self.batch_norm2(x_batch)

            x_batch = x_batch.unsqueeze(0)  # ?
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = self.Conv(x_batch)
            x_batch = x_batch.permute((0, 2, 1))
            x_batch = x_batch.squeeze(0)

            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

class DiGCN_IB_XBN_batch(torch.nn.Module):
    def __init__(self,m,  input_dim, out_dim, args):
        super(DiGCN_IB_XBN_batch, self).__init__()
        self._dropout = args.dropout
        self.batch_size = args.batch_size
        nhid = args.feat_dim
        layer = args.layer
        self.ib1 = InceptionBlock_Di(m, input_dim, nhid, args)
        self.ib2 = InceptionBlock_Di(m, nhid,  out_dim)
        # self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d( out_dim)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.layer = args.layer
        self.ibx=nn.ModuleList([InceptionBlock_Di(m, nhid,nhid) for _ in range(layer-2)])

        self.reg_params = list(self.ib1.parameters()) + list(self.ibx.parameters())
        self.non_reg_params = self.ib2.parameters()

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features

        batch_size = self.batch_size  # Define your batch size
        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)


            # Forward pass for the current batch
            x_batch = self.ib1(batch_x,DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm1(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        for iter_layer in self.ibx:
            num_samples = features.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            outputs = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_x = x[start_idx:end_idx]

                DiGedge_indexi_batch = ()
                DiGedge_weighti_batch = ()
                for i in range(len(edge_index_tuple)):
                    mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                              (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                    edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                    edge_weighti_batch = edge_weight_tuple[i][mask_i]

                    DiGedge_indexi_batch += (edge_indexi_batch,)
                    DiGedge_weighti_batch += (edge_weighti_batch,)

                # Forward pass for the current batch
                x_batch = iter_layer(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
                
                x_batch = self.batch_norm3(x_batch)
                x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
                outputs.append(x_batch)
            x = torch.cat(outputs, dim=0)

        num_samples = features.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_x = x[start_idx:end_idx]

            DiGedge_indexi_batch = ()
            DiGedge_weighti_batch = ()
            for i in range(len(edge_index_tuple)):
                mask_i = ((edge_index_tuple[i][0] >= start_idx) & (edge_index_tuple[i][0] < end_idx) &
                          (edge_index_tuple[i][1] >= start_idx) & (edge_index_tuple[i][1] < end_idx))
                edge_indexi_batch = edge_index_tuple[i][:, mask_i] - start_idx
                edge_weighti_batch = edge_weight_tuple[i][mask_i]

                DiGedge_indexi_batch += (edge_indexi_batch,)
                DiGedge_weighti_batch += (edge_weighti_batch,)

            # Forward pass for the current batch
            x_batch = self.ib2(batch_x, DiGedge_indexi_batch, DiGedge_weighti_batch)
            
            x_batch = self.batch_norm2(x_batch)
            x_batch = F.dropout(x_batch, p=self._dropout, training=self.training)
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

def create_DiG_IB_nhid(m, nfeat, nclass, args):

    if args.layer == 1:
        model = DiGCN_IB_1BN_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_nhid(m, nfeat, nclass, args)
    return model


def create_Di_IB_nhid0(m, nfeat, nclass, args):

    if args.layer == 1:
        model = Di_IB_1BN_nhid0(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = Di_IB_2BN_nhid0(m, nfeat, nclass, args)
    else:
        model = Di_IB_XBN_nhid0(m, nfeat, nclass, args)
    return model

def create_Di_IB_nhid(m, nfeat, nclass, args):
    if args.layer == 1:
        model = Di_IB_1_nhid(m, nfeat,  nclass, args)
    elif args.layer == 2:
        model = Di_IB_2_nhid(m, nfeat,  nclass, args)
    else:
        model = Di_IB_X_nhid(m, nfeat,  nclass, args)
    return model

def create_Si_IB_nhid(m, nfeat, nclass, args):
    nlayer = args.layer
    if args.layer == 1:
        model = Di_IB_1BN_nhid(m, nfeat,  nclass, args)
    elif args.layer == 2:
        model = Si_IB_2BN_nhid(m, nfeat,  nclass, args)
    else:
        model = Di_IB_XBN_nhid(m, nfeat,  nclass, args)
    return model

def create_DiG_IB_nhid_para(m, nfeat, nclass, args):

    if args.layer == 1:
        model = DiGCN_IB_1BN_nhid_para(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_nhid_para(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_nhid_para(m, nfeat, nclass, args)
    return model


def create_DiG_IB_batch(m, nfeat, nclass, args, batchSize):
    if args.layer == 1:
        model = DiGCN_IB_1BN_batch(m, nfeat, nclass, args, batchSize)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_batch(m, nfeat, nclass, args, batchSize)
    else:
        model = DiGCN_IB_XBN_batch(m, nfeat, nclass, args, batchSize)
    return model

def create_DiG_IB_batch_nhid(m, nfeat, nclass, args, batchSize):
    if args.layer == 1:
        model = DiGCN_IB_1BN_batch_nhid(m, nfeat, nclass, args, batchSize)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_batch_nhid(m, nfeat, nclass, args, batchSize)
    else:
        model = DiGCN_IB_XBN_batch_nhid(m, nfeat, nclass, args, batchSize)
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

def create_DiG_IB_Sym(m, nfeat, nclass, args):
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
        model = DiGCN_IB_1BN_Sym(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_Sym(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_Sym(m, nfeat, nclass, args)
    return model

def create_DiG_IB_Sym_batch_nhid(m, nfeat, nclass, args, batchSize):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    if args.layer == 1:
        model = DiGCN_IB_1BN_Sym_batch_nhid(m, nfeat, nclass, args, batchSize)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_Sym_batch_nhid(m, nfeat, nclass, args, batchSize)
    else:
        model = DiGCN_IB_XBN_Sym_batch_nhid(m, nfeat, nclass, args, batchSize)
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

def create_DiG_IB_SymCat_batch_nhid(m, nfeat, nclass, args):
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
        model = DiGCN_IB_1BN_SymCat_batch_nhid(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_SymCat_batch_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_SymCat_batch_nhid(m, nfeat, nclass, args)
    return model

def create_DiG_IB_SymCat_batchConvOut(m, nfeat, nclass, args):
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
        model = DiGCN_IB_1BN_SymCat_batchConvOut(m, nfeat, nclass, args)
    elif args.layer == 2:
        model = DiGCN_IB_2BN_SymCat_batchConvOut(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_XBN_SymCat_batchConvOut(m, nfeat, nclass, args)
    return model

def create_DiG_MixIB_SymCat(m, nfeat, nclass, args):
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
        model = DiGCN_IB_2MixBN_SymCat(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat(m, nfeat, nclass, args)
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

def create_DiG_MixIB_SymCat_batch(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batch_size:

    Returns:

    '''
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_batch(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_batch(m, nfeat, nclass, args)
    return model

def create_DiG_MixIB_SymCat_batch_nhid(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batch_size:

    Returns:

    '''
    nlayer = args.layer
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_batch_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_batch_nhid(m, nfeat, nclass, args)
    return model

def create_DiG_MixIB_SymCat_Sym(m, nfeat, nclass, args):
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
    nlayer = args.layer
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym(m, nfeat, nclass, args)
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

def create_DiG_MixIB_SymCat_Sym_batch(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    nlayer = args.layer
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_batch(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_batch(m, nfeat, nclass, args)
    return model

def create_DiG_MixIB_SymCat_Sym_batch_nhid(m, nfeat, nclass, args):
    '''
    revised for edge_index confusion
    Args:
        nfeat:
        nhid:
        nclass:
        dropout:
        nlayer:
        batchSize:

    Returns:

    '''
    nlayer = args.layer
    if args.layer == 1:
         raise NotImplementedError('mixed can not be from one layer!')
    elif args.layer == 2:
        model = DiGCN_IB_2MixBN_SymCat_Sym_batch_nhid(m, nfeat, nclass, args)
    else:
        model = DiGCN_IB_3MixBN_SymCat_Sym_batch_nhid(m, nfeat, nclass, args)
    return model