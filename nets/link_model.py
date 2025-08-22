import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm


class LINKX_Github(nn.Module):
    """ our LINKX method with skip connections
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super().__init__()
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2 * hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

        self.reset_parameters()  # -

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, x, edge_index):
        row, col = edge_index
        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(self.num_nodes, self.num_nodes)
                         ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x


class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super().__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row - row.min()  # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits


class LINK_Concat(nn.Module):
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):
        super().__init__()
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.cache = cache
        self.x = None

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x, edge_index):
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
            N = x.shape[0]
            row, col = edge_index
            col = col + self.in_channels
            feat_nz = x.nonzero(as_tuple=True)
            feat_row, feat_col = feat_nz
            full_row = torch.cat((feat_row, row))
            full_col = torch.cat((feat_col, col))
            value = x[feat_nz]
            full_value = torch.cat((value,
                                    torch.ones(row.shape[0], device=value.device)))
            x = SparseTensor(row=full_row, col=full_col,
                             sparse_sizes=(N, N + self.in_channels)
                             ).to_torch_sparse_coo_tensor()
            if self.cache:
                self.x = x
        else:
            x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class LINK_Add(nn.Module):
    """ add A and X """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super().__init__()
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)

        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.cache = cache
        self.x = None
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)


    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x, edge_index):
        N = x.shape[0]
        row, col = edge_index
        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(N, N)
                         ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(x, input_tensor=True)
        x = F.relu(xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super().__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x