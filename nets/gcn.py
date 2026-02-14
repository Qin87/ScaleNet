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


from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import reset, glorot, zeros



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
        deg = adj_t.sum(dim=1)
        deg = deg.clamp(min=1e-12)
        deg_inv_sqrt = deg.pow_(-0.5)
        if torch.isnan(deg_inv_sqrt).any():
            raise RuntimeError("NaN detected in deg_inv_sqrt — stopping training to prevent corrupt gradients.")
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
        deg = deg.clamp(min=1e-12)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        if torch.isnan(deg_inv_sqrt).any():
            raise RuntimeError("NaN detected in deg_inv_sqrt(maybe due to negative degree) — stopping training to prevent corrupt gradients.")
        return edge_index, deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[col]


class StandGCNXBN(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        self._cached_adj_t = None
        self.BN_model = args.BN_model
        nhid = args.hid_dim
        dropout = args.dropout
        nlayer = args.layer
        is_add_self_loops = args.First_self_loop
        norm = args.gcn_norm
        self.is_add_self_loops = is_add_self_loops  # Qin True is the original
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
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(adj, sparse_sizes=(num_nodes, num_nodes)).t()

        adj = self._cached_adj_t
        x = self.conv1(x, adj)

        if self.layer == 1:
            # x = F.dropout(x,p= self.dropout_p, training=self.training)
            # if self.BN_model:
            #     x = self.batch_norm2(x)
            return x
        x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                # x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, adj, edge_weight)
                # if self.BN_model:
                #     x= self.batch_norm3(x)
                x = F.relu(x)

        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, adj, edge_weight)
        # if self.BN_model:
        #     x = self.batch_norm2(x)
        # # x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x


class GraphSAGEXBatNorm(nn.Module):
    def __init__(self,  nfeat, nclass, args):
        super().__init__()
        self._cached_adj_t = None
        self.dropout_p = args.dropout
        nhid = args.hid_dim
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
        num_nodes = int(x.shape[0])
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(adj, sparse_sizes=(num_nodes, num_nodes)).t()

        adj = self._cached_adj_t
        x = self.conv1(x, adj)
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
                x = iter_layer(x, adj,edge_weight)
                if self.BN:
                    x = self.batch_norm3(x)
                x = F.relu(x)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, adj,edge_weight)
        if self.BN:
            x = self.batch_norm2(x)

        return x


class ParaGCNXBN(nn.Module):
    def __init__(self,num_node, num_edges, nfeat, nhid, nclass, dropout, nlayer=3, norm=True):
        super().__init__()

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
