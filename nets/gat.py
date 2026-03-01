import typing
from typing import Any

from nets.geometric_baselines import get_norm_adj

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

from torch_geometric.nn import GATConv

import torch.nn as nn

import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class UnifiedGATRATConv(MessagePassing):
    r"""
    Random-Attention GAT:
    - Learned attention REMOVED
    - Edge weights sampled randomly in [0.0001, 10000]
    - Softmax normalization preserved
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        residual: bool = False,
        args: Optional[Any] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.posweight = args.posweight
        if args.net in ['GAT', 'RAT', 'UAT']:
            self.attention_mode = args.net.lower()
        else:
            self.attention_mode = 'gat'
        self.inci_norm = args.inci_norm
        self.num_nodes= args.num_nodes

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        # Node feature projection (KEEP — not attention)
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(
                in_channels, heads * out_channels,
                bias=False, weight_initializer='glorot'
            )
        else:
            self.lin_src = Linear(
                in_channels[0], heads * out_channels,
                bias=False, weight_initializer='glorot'
            )
            self.lin_dst = Linear(
                in_channels[1], heads * out_channels,
                bias=False, weight_initializer='glorot'
            )

        if self.attention_mode == "gat":
            self.att_src = Parameter(torch.empty(1, heads, out_channels))
            self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)


        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        if self.attention_mode == "gat":
            glorot(self.att_src)
            glorot(self.att_dst)
            glorot(self.att_edge)
            zeros(self.bias)


    @overload
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    # -------------------------------------------------------------

    def forward(  # noqa
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
    ):

        H, C = self.heads, self.out_channels
        res: Optional[Tensor] = None

        # ---- Node feature projection (unchanged) ----
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"   # keep as GAT

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None    # keep as GAT
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"    # keep as GAT

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None   # keep as GAT

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if self.attention_mode == 'gat':
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)

        # ---- Self-loops (unchanged) ----
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        if isinstance(edge_index, Tensor):
            index = edge_index[1]
            dim_size = int(index.max()) + 1 if size is None else int(size[1])
            E = index.numel()
            ptr = None
        else:  # SparseTensor
            row, col, _ = edge_index.coo()
            dim_size = edge_index.size(1)
            index = row         # col is wrong!
            ptr = edge_index.storage.rowptr()
            E = index.numel()

        if self.attention_mode == 'gat':
            alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,size=size)
        else:
            if self.attention_mode == 'rat':   # TODO check heads
                alpha = torch.empty((E, self.heads),  device=index.device).uniform_(1e-4, 1e4)
            elif self.attention_mode == 'uat':
                alpha = torch.ones((E, self.heads),device=index.device)
            else:
                raise NotImplementedError(f"Unknown attention_mode: {self.attention_mode}")

        # SAME as GAT
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.inci_norm == 'softmax':
            alpha = softmax(alpha, index, ptr, dim_size)
        else:
            if hasattr(edge_index, 'coo'):
                row, col, _ = edge_index.coo()
                edge_index1 = torch.stack([row, col], dim=0)
            elif hasattr(edge_index, 'row') and hasattr(edge_index, 'col'):
                row = edge_index.row
                col = edge_index.col
                edge_index1 = torch.stack([row, col], dim=0)
            else:
                pass
            try:
                row, col = edge_index1
            except:
                row, col = edge_index
            # del edge_index1
            alphas = []
            for i in range(alpha.shape[1]):
                alpha_i = alpha[:, i]
                if self.posweight in ['e', '2', 'abs']:
                    alpha_i = self.PositiveAttention(alpha_i)
                adj = SparseTensor(row=row, col=col, value=alpha_i, sparse_sizes=(self.num_nodes, self.num_nodes))
                alpha_i_out = self._alpha_from_adj(adj, norm=self.inci_norm)
                alphas.append(alpha_i_out)
            alpha = torch.stack(alphas, dim=1)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # ---- Message passing (unchanged) ----
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)   # TODO edge_index
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                return out, (edge_index, alpha)
            return out, edge_index.set_value(alpha, layout='coo')
        return out

    def PositiveAttention(self, alpha_i):
        if self.posweight == 'abs':
            alpha_i = torch.abs(alpha_i)

        elif self.posweight == '2':
            alpha_i = torch.pow(2.0, alpha_i)  # 2^alpha_i

        elif self.posweight == 'e':
            alpha_i = torch.exp(alpha_i)  # e^alpha_i

        else:
            raise NotImplementedError(f"Unknown posweight type: {posweight}")

        return alpha_i

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, index, ptr, dim_size)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def _alpha_from_adj(self, adj_t: SparseTensor, norm='dir') -> Tensor:
        norm_adj = get_norm_adj(adj_t, norm=norm)  # uses gcn_norm(adj, add_self_loops=0)

        edge_weight = norm_adj.storage.value()
        if edge_weight is None:
            edge_weight = torch.ones(norm_adj.nnz(), device=norm_adj.device())

        return edge_weight

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, {self.out_channels}, heads={self.heads})"
        )


class StandGATXBN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,args):
        super().__init__()
        self.nonlinear = args.nonlinear
        self.Conv = nn.Conv1d(nhid, nclass, kernel_size=1)
        self._cached_adj_t = None

        ConvClass = UnifiedGATRATConv
        self.layer = args.layer

        head = args.heads
        num_head = 1
        head_dim = nhid//num_head

        if args.net=='GAT' and args.originGAT and args.inci_norm=='softmax':
            self.conv1 = GATConv(nfeat, head_dim, heads=args.heads, concat=False)
            self.conv2 = GATConv(nhid, head_dim, heads=head, concat=False)
            self.convx = nn.ModuleList([GATConv(nhid, head_dim, heads=head, concat=False) for _ in range(args.layer - 2)])
        else:
            self.conv1 = ConvClass(nfeat, head_dim, heads=args.heads, args=args, concat=False)
            self.conv2 = ConvClass(nhid, head_dim, heads=head, args=args, concat=False)
            self.convx = nn.ModuleList([ConvClass(nhid, head_dim, heads=head, args= args, concat=False) for _ in range(args.layer-2)])
        self.dropout_p = dropout
        self.is_add_self_loops = True

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nhid)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()


    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).t()   # checked needing t

        edge_index = self._cached_adj_t
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        if self.layer == 1:
            return self.tran_lin(x)

        if self.nonlinear:
            x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                x = F.dropout(x, p= self.dropout_p, training=self.training)
                x = iter_layer(x, edge_index)
                x = self.batch_norm3(x)
                if self.nonlinear:
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


