import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge



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





