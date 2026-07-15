import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import GCNConv

from nets.jumping_weight import JumpingKnowledge

from utils.utils import get_norm_adj


def get_conv2(input_dim, output_dim, args):
    if args.conv_type2 == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=args.self_loops)
    elif args.conv_type2 == "faber":
        return FaberConv(input_dim, output_dim, args)
    elif args.conv_type2 == "scale":
        return ScaleConv(input_dim, output_dim, args)
    else:
        raise ValueError(f"Convolution type {args.conv_type2} not supported")

class GNN2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_type = args.conv_type2
        self.lrelu_slope = args.lrelu_slope

        output_dim = args.hid_dim
        if args.layer == 1:
            self.convs = ModuleList([get_conv2(args.num_features, output_dim, args)])
        else:
            self.convs = ModuleList([get_conv2(args.num_features, args.hid_dim, args)])
            for _ in range(args.layer - 2):
                self.convs.append(get_conv2(args.hid_dim, args.hid_dim, args))
            self.convs.append(get_conv2(args.hid_dim, output_dim, args))

        if args.jk:
            input_dim = args.hid_dim * args.layer if args.jk == "cat" else args.hid_dim
            self.lin = Linear(input_dim, args.n_cls)
            self.jump = JumpingKnowledge(mode=args.jk, channels=args.hid_dim, num_layers=args.layer)
        else:
            self.lin = Linear(args.hid_dim, args.n_cls)

        self.num_layers = args.layer
        self.dropout = args.dropout
        self.jk = args.jk
        self.normalize = args.normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i != len(self.convs) - 1 or self.jk or len(self.convs) == 1:
                x = F.relu(x)
                # x = F.leaky_relu(x,negative_slope= self.lrelu_slope)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jk:
            x = self.jump(xs)
            x = self.lin(x)
        else:
            x = self.lin(x)

        return x


class ScaleConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lins_dst_to_src = torch.nn.ModuleList([Linear(input_dim, output_dim) for _ in range(2 * args.k_plus)])
        self.lins_src_to_dst = torch.nn.ModuleList([Linear(input_dim, output_dim) for _ in range(2 * args.k_plus)])

        self.alpha = args.alphaDir
        self.beta = args.betaDir
        self.gamma = args.gamaDir
        self.adj_norm, self.adj_t_norm = None, None
        self.inci_norm = args.inci_norm
        self.BN_model = args.BN_model
        self.batch_norm2 = nn.BatchNorm1d(output_dim)

        self.k_plus = args.k_plus
        self.exponent = args.exponent
        self.weight_penalty = args.weight_penalty
        self.zero_order = args.zero_order
        self.structure = args.structure
        self.cat_A_X = args.cat_A_X

        if self.structure != 0:
            self.mlp_struct = Linear(args.num_nodes, output_dim)
        if self.cat_A_X != 0:
            self.mlp_cat = Linear(2 * output_dim, output_dim)
        if self.zero_order:
            self.lin_zero = Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm=self.inci_norm, exponent=self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm=self.inci_norm, exponent=self.exponent)

        y = self.adj_norm @ x
        y_t = self.adj_t_norm @ x
        sum_src_to_dst = self.lins_src_to_dst[0](y)
        sum_dst_to_src = self.lins_dst_to_src[0](y_t)
        totalA = 0
        if self.alpha != -1:
            totalA = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src
            if self.BN_model:
                totalA = self.batch_norm2(totalA)

        totalB = 0
        totalC = 0
        if self.k_plus > 1:
            def get_weight(i):
                if self.weight_penalty == 'exp':
                    return 1 / (2 ** i)
                elif self.weight_penalty == 'lin':
                    return 1 / i
                elif self.weight_penalty == 'None' or self.weight_penalty is None:
                    return 1
                else:
                    raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")

            yy = y
            ytyt = y_t
            yty = y
            yyt = y_t
            for i in range(1, self.k_plus):
                yy = self.adj_norm @ yy
                yty = self.adj_t_norm @ yty

                yyt = self.adj_norm @ yyt
                ytyt = self.adj_t_norm @ ytyt

                w = get_weight(i)

                if self.beta != -1:
                    b_term = (
                            self.beta * self.lins_src_to_dst[2 * i - 1](yyt)
                            + (1 - self.beta) * self.lins_src_to_dst[2 * i - 1](yty)
                    )
                    totalB += b_term * w
                    # if self.BN_model:
                    #     totalB = self.batch_norm2(totalB)

                if self.gamma != -1:
                    c_term = (
                            self.gamma * self.lins_src_to_dst[2 * i](yy)
                            + (1 - self.gamma) * self.lins_dst_to_src[2 * i](ytyt)
                    )
                    totalC += c_term * w
                    # if self.BN_model:
                    #     totalC = self.batch_norm2(totalC)

        mpnn_total = totalA + totalB + totalC

        if self.structure != 0:
            struct_value = self.adj_norm @ self.mlp_struct.weight.T
            total = self.structure * struct_value + (1 - self.structure) * mpnn_total
        else:
            total = mpnn_total

        if self.cat_A_X:
            struct_value = self.adj_norm @ self.mlp_struct.weight.T
            concat_feat = torch.cat([struct_value, mpnn_total], dim=1)
            concat_output = self.mlp_cat(concat_feat)
            total += concat_output

        if self.zero_order:
            total = total + self.lin_zero(x)

        return total


class FaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_plus = args.k_plus
        self.exponent = args.exponent
        self.weight_penalty = args.weight_penalty
        self.zero_order = args.zero_order

        if self.zero_order:
            self.lin_src_to_dst_zero = Linear(input_dim, output_dim)
            self.lin_dst_to_src_zero = Linear(input_dim, output_dim)

        self.lins_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(args.k_plus)
        ])

        self.lins_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(args.k_plus)
        ])

        self.alpha = args.alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent=self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent=self.exponent)

        y = self.adj_norm @ x
        y_t = self.adj_t_norm @ x
        sum_src_to_dst = self.lins_src_to_dst[0](y)
        sum_dst_to_src = self.lins_dst_to_src[0](y_t)
        if self.zero_order:
            sum_src_to_dst = sum_src_to_dst + self.lin_src_to_dst_zero(x)
            sum_dst_to_src = sum_dst_to_src + self.lin_dst_to_src_zero(x)

        if self.k_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1, self.k_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y) / (2 ** i)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t) / (2 ** i)

            elif self.weight_penalty == 'lin':
                for i in range(1, self.k_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y) / i
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t) / i
            elif self.weight_penalty == None:
                for i in range(1, self.k_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)
            else:
                raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")

        total = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src

        return total
