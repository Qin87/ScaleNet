import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP

from nets.Sym_Reg import DGCNConv


class APPNP1LayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP2LayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim,num_classes, dropout, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNPXLayerWithGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.convx = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        # self.convx = GCNConv(hidden_dim, hidden_dim)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        for iter_layer in self.convx:
            # x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class APPNP1BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP2BN(nn.Module):
    def __init__(self, in_features, hidden_dim,num_classes, dropout, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNPXBN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.convx = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        # self.convx = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.propagate = APPNP(K=K, alpha=alpha)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))

        for iter_layer in self.convx:
            # x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index)))

        x = self.propagate(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class APPNP1Simp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super().__init__()
        self.line1 = nn.Linear(in_features, num_classes)
        self.conv1 = APPNP(K=10, alpha=alpha)

        self.batch_norm1 = nn.BatchNorm1d(num_classes)

        self.reg_params = []
        self.non_reg_params = self.conv1.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.batch_norm1(self.conv1(x, edge_index))
        return x


class APPNP2Simp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, alpha, K):
        super().__init__()
        self.line1 = nn.Linear(in_features, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, num_classes)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.line2(x)
        # x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.batch_norm2(self.conv2(x, edge_index))
        return x

class APPNP_Model(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, alpha = 0.1, dropout = False, layer=3):
        super().__init__()
        self.dropout = dropout
        self.line1 = nn.Linear(input_dim, filter_num)
        self.line2 = nn.Linear(filter_num, filter_num)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)
        self.layer = layer
        if layer == 3:
            self.line3 = nn.Linear(filter_num, filter_num)
            self.conv3 = APPNP(K=10, alpha=alpha)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = self.conv1(x, edge_index)
        if self.layer == 1:
            return self.process_output(x)

        x = F.relu(x)
        x = self.line2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.line3(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        return self.process_output(x)

    def process_output(self, x):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()
        return F.log_softmax(x, dim=1)


class SymModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super().__init__()
        self.layer = layer
        self.dropout = dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(filter_num * 3, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim, filter_num, bias=False)
        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        nn.init.zeros_(self.bias1)
        if layer > 1:
            self.linx = nn.ModuleList([torch.nn.Linear(filter_num * 3, filter_num, bias=False) for _ in range(layer - 1)])
            self.biasx = nn.ParameterList([nn.Parameter(torch.Tensor(1, filter_num)) for _ in range(layer - 1)])
            for bias in self.biasx:
                nn.init.zeros_(bias)
    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis=-1)
        if self.layer == 1:
            return self.process_output(x)

        x = F.relu(x)

        for i in range(self.layer - 1):
            x = self.linx[i](x)
            x1 = self.gconv(x, edge_index)
            x2 = self.gconv(x, edge_in, in_w)
            x3 = self.gconv(x, edge_out, out_w)

            x1 += self.biasx[i]
            x2 += self.biasx[i]
            x3 += self.biasx[i]

            x = torch.cat((x1, x2, x3), axis=-1)
            x = F.relu(x)


        return self.process_output(x)


    def process_output(self, x):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        # return F.log_softmax(x, dim=1)    # change july 24
        return x

class ChebModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, K, dropout = False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(input_dim, filter_num, K)
        self.conv2 = ChebConv(filter_num, filter_num, K)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = ChebConv(filter_num, filter_num, K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.layer == 1:
            return self.process_output(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        return self.process_output(x)

    def process_output(self, x):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        return F.log_softmax(x, dim=1)

class GCNModel_Cheb(torch.nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, dropout=False, layer=2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, filter_num)
        self.mlp1 = torch.nn.Linear(filter_num, filter_num)
        self.conv2 = GCNConv(filter_num, filter_num)
        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.conv3 = GCNConv(filter_num, filter_num)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.mlp1(x)       # TODO  Qin for test
        if self.layer == 1:
            return self.process_output(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.layer == 3:
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        return self.process_output(x)

    def process_output(self, x):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        return F.log_softmax(x, dim=1)

class APPNPXSimp_BN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout, layer, alpha, K):
        super().__init__()

        self.line1 = nn.Linear(in_features, hidden_dim)
        self.line2 = nn.Linear(hidden_dim, num_classes)

        self.conv1 = APPNP(K=10, alpha=alpha)
        self.conv2 = APPNP(K=10, alpha=alpha)

        self.linex = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layer - 2)])
        self.convx = nn.ModuleList([APPNP(K=10, alpha=alpha) for _ in range(layer - 2)])

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_classes)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

        self.reg_params = list(self.conv1.parameters()) + list(self.convx.parameters())
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = self.line1(x)
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))

        for iter_layer, line_layer in zip(self.convx, self.linex):
            x = line_layer(x)
            x = F.relu(self.batch_norm3(iter_layer(x, edge_index)))

        x = self.line2(x)
        x = self.batch_norm2(self.conv2(x, edge_index))
        return x

def create_APPNPGGPT_0(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1LayerWithGCN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2LayerWithGCN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXLayerWithGCN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model

def create_APPNPGGPT(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1BN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2BN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXBN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model

def create_APPNPSimp(nfeat, nhid, nclass, dropout, nlayer, alpha=0.1, K=10):
    if nlayer == 1:
        model = APPNP1Simp_BN(nfeat, nhid, nclass,  dropout, alpha=0.1, K=10)
    elif nlayer == 2:
        model = APPNP2Simp_BN(nfeat, nhid, nclass, dropout,  alpha, K)
    else:
        model = APPNPXSimp_BN(nfeat, nhid, nclass, dropout, nlayer, alpha, K)
    return model