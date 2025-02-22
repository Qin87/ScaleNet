from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
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
import torch
import torch.nn as nn
import torch_sparse


# from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP
from torch.nn import Parameter, ModuleList
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum
from torch_sparse import mul



from torch_geometric.utils import add_self_loops

from nets.jumping_weight import JumpingKnowledge

def gcn_norm_option(inci_norm, edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=0,flow: str = "source_to_target", dtype=None):
    '''
copy from torch-geometric, but they are wrong. and add different norm options
    '''
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops == 1:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        if inci_norm == 'sym':
            idx = 0 if flow == 'source_to_target' else 1
            deg = torch_sparse.sum(adj_t, dim=idx)  # Qin dim from 1 to 0
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
            return adj_t
        elif inci_norm == 'row':
            idx = 0 if flow == 'source_to_target' else 1
            row_sum = sparsesum(adj_t, dim=idx)
            inv_deg = 1 / row_sum.view(-1, 1)
            inv_deg.masked_fill_(inv_deg == float("inf"), 0.0)

            return mul(adj_t, inv_deg)
        elif inci_norm == 'dir':
            idx = 0 if flow == 'source_to_target' else 1
            device = adj_t.device()
            in_deg = sparsesum(adj_t, dim=idx)
            in_deg_inv_sqrt = in_deg.pow(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

            out_deg = sparsesum(adj_t, dim=1-idx)
            out_deg_inv_sqrt = out_deg.pow(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

            out_deg_inv_sqrt = out_deg_inv_sqrt.to(device)
            in_deg_inv_sqrt = in_deg_inv_sqrt.to(adj_t.device())

            adj0 = mul(adj_t, out_deg_inv_sqrt.view(-1, 1))
            adj1_t = mul(adj0, in_deg_inv_sqrt.view(1, -1))
            return adj1_t


    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        if inci_norm == 'sym':
            row, col = edge_index[0], edge_index[1]
            idx = col if flow == 'source_to_target' else row

            deg = scatter(value, idx, 0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        elif inci_norm == 'row':
            row, col = edge_index[0], edge_index[1]
            idx = col if flow == 'source_to_target' else row

            row_sum = scatter(value, idx, 0, dim_size=num_nodes, reduce='sum')
            inv_deg = 1 / row_sum.view(-1, 1)
            inv_deg.masked_fill_(inv_deg == float("inf"), 0.0)
            value = inv_deg[row] * value

        elif inci_norm == 'dir':
            row, col = edge_index[0], edge_index[1]
            idx = col if flow == 'source_to_target' else row
            device = adj_t.device()
            in_deg = scatter(value, idx, 0, dim_size=num_nodes, reduce='sum')
            in_deg_inv_sqrt = in_deg.pow(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

            idx_ = row if flow == 'source_to_target' else col
            out_deg = scatter(value, idx_, 0, dim_size=num_nodes, reduce='sum')
            out_deg_inv_sqrt = out_deg.pow(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

            out_deg_inv_sqrt = out_deg_inv_sqrt.to(device)
            in_deg_inv_sqrt = in_deg_inv_sqrt.to(adj_t.device())

            value = in_deg_inv_sqrt[row] * value * out_deg_inv_sqrt[col]

        return set_sparse_value(adj_t.coalesce(), value), None

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops == 1:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    if inci_norm == 'sym':
        row, col = edge_index[0], edge_index[1]
        idx = col if flow == 'source_to_target' else row
        deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif inci_norm == 'row':
        row, col = edge_index[0], edge_index[1]
        idx = col if flow == 'source_to_target' else row

        row_sum = scatter(edge_weight, idx, 0, dim_size=num_nodes, reduce='sum')
        inv_deg = 1 / row_sum.view(-1, 1)
        inv_deg.masked_fill_(inv_deg == float("inf"), 0.0)
        edge_weight = inv_deg[row] * edge_weight
    elif inci_norm == 'dir':
        row, col = edge_index[0], edge_index[1]
        idx = col if flow == 'source_to_target' else row
        device = adj_t.device()
        in_deg = scatter(edge_weight, idx, 0, dim_size=num_nodes, reduce='sum')
        in_deg_inv_sqrt = in_deg.pow(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        idx_ = row if flow == 'source_to_target' else col
        out_deg = scatter(edge_weight, idx_, 0, dim_size=num_nodes, reduce='sum')
        out_deg_inv_sqrt = out_deg.pow(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        out_deg_inv_sqrt = out_deg_inv_sqrt.to(device)
        in_deg_inv_sqrt = in_deg_inv_sqrt.to(adj_t.device())

        edge_weight = in_deg_inv_sqrt[row] * edge_weight * out_deg_inv_sqrt[col]

    return edge_index, edge_weight


class GCNConv_inciNormOption(MessagePassing):
    '''
    copy from GCNConv, but introduce options for different incidence normalization
    '''

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            args,
            improved: bool = False,
            cached: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = args.First_self_loop
        self.inci_norm = args.inci_norm

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.inci_norm:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm_option( self.inci_norm,  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm_option( self.inci_norm,   # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class ScaleNet_2025(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.feat_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([DirGCNConv_Feb18(nfeat, output_dim, args)])
        else:
            self.convs = ModuleList([DirGCNConv_Feb18(nfeat, nhid, args)])
            for _ in range(layer - 2):
                self.convs.append(DirGCNConv_Feb18(nhid, nhid, args))
            self.convs.append(DirGCNConv_Feb18(nhid, output_dim, args))

        num_scale = layer
        self.mlp = None
        if jumping_knowledge:
            input_dim = hidden_dim * num_scale if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear

        self.adj, self.adj_t = None, None
        self.adj_A_A, self.adj_A_At, self.adj_At_A, self.adj_At_At = None, None, None, None

    def forward(self, x, edge_index):
        if self.adj is None:
            num_nodes = x.shape[0]
            self.adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
            self.adj_t = SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(num_nodes, num_nodes))
        if self.adj_A_A is None:
            self.adj_A_A = self.adj @ self.adj
            self.adj_A_At = self.adj @ self.adj_t
            self.adj_At_A = self.adj_t @ self.adj
            self.adj_At_At = self.adj_t @ self.adj_t

        xs = []

        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index)
            x = conv(x, self.adj, self.adj_t, self.adj_A_A,  self.adj_A_At, self.adj_At_A, self.adj_At_At)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                if self.nonlinear:
                    x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return x

class DirGCNConv_Feb18(torch.nn.Module):
    '''
    just use GCN and SAGE
    '''
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = GCNConv_inciNormOption(input_dim, output_dim, args)
            self.lin_dst_to_src = GCNConv_inciNormOption(input_dim, output_dim, args)

            self.linx = nn.ModuleList([GCNConv_inciNormOption(input_dim, output_dim, args) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.conv2_1 = Linear(output_dim*2, output_dim)

        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim,  root_weight=True)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=True)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=True) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.conv2_1 = Linear(output_dim * 2, output_dim)

            self.lin_sage = nn.ModuleList([Linear(input_dim, output_dim) for i in range(3)])    # self
        elif args.conv_type == 'dir-gat':
            # heads = args.heads
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim*heads, heads=heads)for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim*heads)
            self.conv2_1 = Linear(output_dim*heads*2, output_dim*heads)
        else:
            raise NotImplementedError

        self.First_self_loop = args.First_self_loop
        self.rm_gen_sloop = args.rm_gen_sloop
        self.differ_AA = args.differ_AA
        self.differ_AAt = args.differ_AAt
        if self.differ_AA or self.differ_AAt:
            args.betaDir, args.gamaDir = -1, -1

        self.alpha = nn.Parameter(torch.ones(1) * args.alphaDir, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1) * args.betaDir, requires_grad=False)
        self.gama = nn.Parameter(torch.ones(1) * args.gamaDir, requires_grad=False)

        self.norm_list = []

        self.BN_model = args.BN_model
        self.inci_norm = args.inci_norm

        self.conv_type = args.conv_type

        self.adj_norm, self.adj_t_norm = None, None

        # self
        self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out = None, None, None, None
        self.adj_intersection, self.adj_intersection_in_in, self.adj_intersection_in_out = None, None, None
        self.adj_union, self.adj_union_in_in, self.adj_union_in_out = None, None, None
        self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out = None, None, None, None
        self.Intersect_alpha, self.Union_alpha, self.Intersect_beta, self.Union_beta, self.Intersect_gama, self.Union_gama = None, None, None, None, None, None

        num_scale = 3
        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge:
            input_dim_jk = output_dim * num_scale if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.linjk = Linear(input_dim_jk, output_dim)

    def forward(self, x,  adj, adj_t, adj_A_A, adj_A_At, adj_At_A, adj_At_At):
        num_nodes = x.shape[0]
        if self.conv_type == 'dir-gcn':
            out1 = self.alpha * self.lin_src_to_dst(x, adj) + (1-self.alpha) * self.lin_dst_to_src(x, adj_t)

            if not (self.beta == -1 and self.gama == -1):
                out2 = self.beta * self.linx[0](x, adj_A_A) + (1 - self.beta) * self.linx[1](x, adj_A_At)
                out3 = self.gama * self.linx[2](x, adj_At_A) + (1 - self.gama) * self.linx[3](x, adj_At_At)
            else:
                out2 = torch.zeros_like(out1)
                out3 = torch.zeros_like(out1)


        elif self.conv_type in ['dir-gat', 'dir-sage']:
            edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
            if not(self.beta == -1 and self.gama == -1) and self.edge_in_in is None:
                self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out =get_higher_edge_index(edge_index, num_nodes, rm_gen_sLoop=rm_gen_sLoop)
                self.Intersect_alpha, self.Union_alpha = edge_index_u_i(edge_index, edge_index_t)
                self.Intersect_beta, self.Union_beta = edge_index_u_i(self.edge_in_out, self.edge_out_in)
                self.Intersect_gama, self.Union_gama = edge_index_u_i(self.edge_in_in, self.edge_out_out)

                if self.differ_AA:
                    diff_0 = remove_shared_edges(self.edge_in_in, edge_index, edge_index_t)
                    diff_1 = remove_shared_edges(self.edge_out_out, edge_index, edge_index_t)
                elif self.differ_AAt:
                    diff_0 = remove_shared_edges(self.edge_in_out, edge_index, edge_index_t)
                    diff_1 = remove_shared_edges(self.edge_out_in, edge_index, edge_index_t)
                if self.differ_AA or self.differ_AAt:
                    edge_index = diff_0
                    edge_index_t = diff_1

            out1 = aggregate_index(x, self.alpha, self.lin_src_to_dst, edge_index, self.lin_dst_to_src, edge_index_t, self.Intersect_alpha, self.Union_alpha)
            if not (self.beta == -1 and self.gama == -1):
                if self.beta != -1:
                    out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in, self.Intersect_beta, self.Union_beta)
                else:
                    out2 = torch.zeros_like(out1)
                if self.gama != -1:
                    out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out, self.Intersect_gama, self.Union_gama)
                else:
                    out3 = torch.zeros_like(out1)

            else:
                out2 = torch.zeros_like(out1)
                out3 = torch.zeros_like(out1)

        else:
            raise NotImplementedError

        xs = [out1, out2, out3]

        if self.jumping_knowledge_inner:
            x = self.jump(xs)
            x = self.linjk(x)
        else:
            x = sum(out for out in xs)

        if self.BN_model:
            x = self.batch_norm2(x)
        return x