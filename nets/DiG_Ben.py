import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

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


class DiG_Ben1(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim,  dropout, layer=1):
        super(DiG_Ben1, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

        # # # type2
        # self.reg_params = list(self.conv1.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_Ben2(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout, layer=2):
        super(DiG_Ben2, self).__init__()
        self.dropout = dropout

        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, hid_dim)
        self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

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
        x = F.relu(x)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

class DiG_BenX(torch.nn.Module):

    def __init__(self, input_dim,  hid_dim, out_dim, dropout, layer=3):
        super(DiG_BenX, self).__init__()
        self.dropout = dropout
        self.conv1 = DIGCNConv(input_dim, hid_dim)
        self.conv2 = DIGCNConv(hid_dim, hid_dim)
        self.convx = nn.ModuleList([DIGCNConv(hid_dim, hid_dim) for _ in range(layer-2)])
        self.Conv = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # type1
        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

        # # type2
        # self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters()) + list(self.conv2.parameters())
        # self.non_reg_params = self.Conv.parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        for iter_layer in self.convx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.conv2(x, edge_index))      # I should desert this line
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return x

def create_DiG(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = DiG_Ben1(nfeat, nhid, nclass, dropout, nlayer)
    elif nlayer == 2:
        model = DiG_Ben2(nfeat, nhid, nclass, dropout, nlayer)
    else:
        model = DiG_BenX(nfeat, nhid, nclass, dropout, nlayer)
    return model