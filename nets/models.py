import torch
from torch import nn
from torch_geometric.nn import GCNConv, SGConv, APPNP, JumpingKnowledge
from torch.nn import BatchNorm1d, Embedding, ModuleList, ReLU
from torch_geometric.nn import GINEConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from nets.pgnn_conv import pGNNConv
from nets.gpr_conv import GPR_prop


class pGNNNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)
        self.BN1= nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.BN1(self.lin1(x)))      # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class pGNNNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, num_hid, mu, p, K, cached=cached)
        self.conv2 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)
        self.BN1 = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        # x = F.relu(self.lin1(x))
        x = F.relu(self.BN1(self.lin1(x)))      # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class pGNNNetX(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,layer=3,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.layerx = nn.ModuleList([pGNNConv(num_hid, num_hid, mu, p, K, cached=cached) for _ in range(layer-2)])
        self.BN1 = nn.BatchNorm1d(num_hid)

        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.BN1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for iter_layer in self.layerx:
            x = F.relu(self.BN1(iter_layer(x, edge_index, edge_weight)))        # Qin add BN Apr30
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class MLPNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)
        self.BN1 = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.BN1(self.layer1(x)))        # Qin add BN on Apr29
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

class MLPNetX(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout, layer=3):
        super().__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)
        self.layerx = nn.ModuleList([torch.nn.Linear(num_hid, num_hid) for _ in range(layer-2)])
        self.BN1 = nn.BatchNorm1d(num_hid)
        self.BNx = nn.BatchNorm1d(num_hid)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.BN1(self.layer1(x)))    # Qin add BN Apr29
        # x = self.BN1(self.layer1(x))  # Qin add BN Apr29
        x = F.dropout(x, p=self.dropout, training=self.training)
        for iter_layer in self.layerx:
            x = F.relu(self.BNx(iter_layer(x)))    # Qin add BN Apr29
            # x = self.BNx(iter_layer(x))    # Qin add BN Apr29
            # x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x= torch.relu(x)
        return F.log_softmax(x, dim=1)

class MLPNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,num_hid,
                 out_channels,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, out_channels)
        # self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.layer1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.layer2(x)
        return F.log_softmax(x, dim=1)

def create_MLP(nfeat, nhid, nclass, dropout, nlayer):
    if nlayer == 1:
        model = MLPNet1(nfeat, nhid, nclass, dropout)
    elif nlayer == 2:
        model = MLPNet2(nfeat, nhid, nclass, dropout)
    else:
        model = MLPNetX(nfeat, nhid, nclass, dropout, nlayer)
    return model


def create_pgnn(nfeat, nhid, nclass,mu=0.1,p=2,K=2, dropout=0.5, layer=3):
    if layer == 1:
        model = pGNNNet1(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5)
    elif layer == 2:
        model = pGNNNet2(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5)
    else:
        model = pGNNNetX(nfeat, nhid, nclass,mu=0.1, p=2,K=2, dropout=0.5, layer=3)
    return model

def create_SGC(nfeat, nhid, nclass, dropout, nlayer, K):
    if nlayer == 1:
        model = SGCNet1(nfeat, nhid, nclass, dropout, K)
    elif nlayer == 2:
        model = SGCNet2(nfeat, nhid, nclass, dropout,K)
    else:
        model = SGCNetX(nfeat, nhid, nclass, dropout, nlayer, K)
    return model


class SGCNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,nhid,
                 out_channels,dropout,
                 K=2,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class SGCNet2(torch.nn.Module):
    def __init__(self,
                 in_channels,nhid,
                 out_channels,dropout,
                 K=2,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, nhid, K=K, cached=cached)
        self.conv2 = SGConv(nhid, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class SGCNetX(torch.nn.Module):
    def __init__(self,
                 in_channels, nhid,
                 out_channels,dropout=0.5, layer=3,
                 K=2,
                 cached=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SGConv(in_channels, nhid, K=K, cached=cached)
        self.conv2 = SGConv(nhid, out_channels, K=K, cached=cached)
        self.layerx = nn.ModuleList([SGConv(nhid, nhid, K=K, cached=cached) for _ in range(layer-2)])

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        for iter_layer in self.layerx:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index, edge_weight))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class JKNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0,
                 dropout=0.5,layer=4):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.lin1 = torch.nn.Linear(num_hid, out_channels)
        self.one_step = APPNP(K=K, alpha=alpha)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=num_hid,
                                   num_layers=layer)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index, edge_weight)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class GPRGNNNet1(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.BN1 = nn.BatchNorm1d(num_hid)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.BN1(self.lin1(x)))          # Qin add BN Apr30
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int, d_conv: int, order_by_degree: False):
        super().__init__()
        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)

            if self.model_type == 'mamba':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)

            if self.model_type == 'transformer':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5, att_type='transformer')

            # conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        return self.mlp(x)


