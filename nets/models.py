import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge

from nets.DiG_NoConv import DiG_SimpleXBN_nhid_Pan
from src.pgnn_conv import pGNNConv
from src.gpr_conv import GPR_prop


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


class GATNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)


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


class APPNPNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0.1,
                 dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
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


class GPRGNNNet1_Qin(torch.nn.Module):
    '''
    Qin want to move prop before conv, worse
    '''
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
        x = F.relu(self.lin1(x))


        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class GPRGNNNet2(torch.nn.Module):
    '''
    Qin want to move prop before conv,not use
    '''
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

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
            self.prop2 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)
            self.prop2 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))



        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index, edge_weight)

        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop2(x, edge_index, edge_weight)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        return F.log_softmax(x, dim=1)


from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

# from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

class GPSConv(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            att_type: str = 'transformer',
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            d_state: int = 16,
            d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        if self.att_type == 'mamba':
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        ### Global attention transformer-style model.
        if self.att_type == 'transformer':
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = h[mask]

        if self.att_type == 'mamba':

            if self.order_by_degree:
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                _, x = sort_edge_index(order_tensor, edge_attr=x)

            if self.shuffle_ind == 0:
                h, mask = to_dense_batch(x, batch)
                h = self.self_attn(h)[mask]
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    h_ind_perm = permute_within_batch(x, batch)
                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                    h_i = self.self_attn(h_i)[mask][h_ind_perm]
                    mamba_arr.append(h_i)
                h = sum(mamba_arr) / self.shuffle_ind
        ###

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')


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


