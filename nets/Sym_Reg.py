from operator import mul, matmul
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, fill_diag


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
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
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
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class DGCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        self.in_channels = None
        self.out_channels = None
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,size=None)

        self.in_channels = x.size(1)
        self.out_channels = out.size(1)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SymRegLayer2BN_add(torch.nn.Module):
    """
    Don't try again to simplify it by deleting Conv, because the catenation
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = self.lin2.parameters()

    # def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_Qin_in_tensor, edge_Qin_out_tensor):
    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        # edge_weight = normalize_edges_all1(edge_index)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1+x2+x3
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + x2 + x3
        return x

class SymRegLayer2BN_para_add(torch.nn.Module):
    """
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.coef1 = BiasParameter()
        self.coef2 = BiasParameter()

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = self.lin2.parameters()
        self.coefs = [self.coef1, self.coef2]

    # def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_Qin_in_tensor, edge_Qin_out_tensor):
    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + self.coef1.bias[0] * x2 + self.coef1.bias[1] * x3
        # x = self.batch_norm1(x)
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + self.coef2.bias[0] * x2 + self.coef2.bias[1] * x3
        x = self.batch_norm2(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        return x

class SymRegLayerXBN_add(torch.nn.Module):
    """
    Don't try again to simplify it by deleting Conv, because the catenation
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=3):
        super().__init__()
        self.layer = layer
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = self.lin2.parameters()

    # def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_Qin_in_tensor, edge_Qin_out_tensor):
    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1+x2+x3
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        for iter_layer in self.linx:
            x = iter_layer(x)
            x1 = self.gconv(x, edge_index)
            x2 = self.gconv(x, edge_in, in_w)
            x3 = self.gconv(x, edge_out, out_w)

            x = x1 + x2 + x3
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + x2 + x3
        return x

class BiasParameter(nn.Parameter):
    def __init__(self):
        super().__init__()
        self.bias = (torch.randn(2))
        # super().__init__(torch.randn(2))
    def forward(self):
        return self.bias
class SymRegLayerXBN_para_add(torch.nn.Module):
    """
    Don't try again to simplify it by deleting Conv, because the catenation
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=3):
        super().__init__()
        self.layer = layer
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(out_dim, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, nhid, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)
        self.linx = nn.ModuleList([torch.nn.Linear(nhid, nhid, bias=False) for _ in range(layer - 2)])

        self.coef1 = BiasParameter()
        self.coef2 = BiasParameter()
        self.coefx = [BiasParameter() for _ in range(layer - 2)]

        nn.init.ones_(self.coef1.bias)
        nn.init.ones_(self.coef2.bias)
        for bias_param in self.coefx:
            nn.init.ones_(bias_param.bias)

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.batch_normx = nn.BatchNorm1d(nhid)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = self.lin2.parameters()
        self.coefs = [self.coef1, self.coef2]+ self.coefx

    # def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w, edge_Qin_in_tensor, edge_Qin_out_tensor):
    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + self.coef1.bias[0] * x2 + self.coef1.bias[1] * x3
        x = self.batch_norm1(x)     # keep is better
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        # for i, iter_layer in self.linx:
        for i, iter_layer in zip(self.coefx, self.linx):
            x = iter_layer(x)
            x1 = self.gconv(x, edge_index)
            x2 = self.gconv(x, edge_in, in_w)
            x3 = self.gconv(x, edge_out, out_w)

            x = x1 + i.bias[0] * x2 + i.bias[1] * x3
            # x = self.batch_normx(x)       # without is better
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + self.coef2.bias[0] * x2 + self.coef2.bias[1] * x3
        x = self.batch_norm2(x)     # keep is better
        # x = F.relu(x)     # worse
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)      # with this is better

        return x


class SymRegLayer1BN_add(torch.nn.Module):
    """
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, out_dim, bias=False)
        # self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, nhid))
        # self.bias2 = nn.Parameter(torch.Tensor(1, out_dim))

        nn.init.zeros_(self.bias1)
        # nn.init.zeros_(self.bias2)

        self.batch_norm1 = nn.BatchNorm1d(out_dim)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = []

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1+x2+x3

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        return x

class SymRegLayer1BN_para_add(torch.nn.Module):
    """
    """
    def __init__(self, input_dim, nhid, out_dim,dropout=False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(nhid, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, out_dim, bias=False)
        self.lin2 = torch.nn.Linear(nhid, out_dim, bias=False)

        self.coefi = nn.Parameter(torch.randn(1))
        self.coef2 = nn.Parameter(torch.randn(1))
        nn.init.constant_(self.coefi, 1)
        nn.init.constant_(self.coef2, 1)

        self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.reg_params = list(self.lin1.parameters()) + list(self.gconv.parameters())
        self.non_reg_params = self.lin2.parameters()
        self.coefs = [self.coefi, self.coef2]

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x = x1 + self.coefi * x2 + self.coef2 * x3
        x = self.batch_norm1(x)
        # x = F.relu(x)     # worse so desert

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        return x


def create_SymReg_add(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = SymRegLayer1BN_add(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = SymRegLayer2BN_add(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = SymRegLayerXBN_add(nfeat, nhid, nclass, dropout, nlayer)
    return model

def create_SymReg_para_add(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = SymRegLayer1BN_para_add(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = SymRegLayer2BN_para_add(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = SymRegLayerXBN_para_add(nfeat, nhid, nclass, dropout, nlayer)
    return model