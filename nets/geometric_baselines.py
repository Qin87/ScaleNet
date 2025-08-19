import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
# from dgl.backend.mxnet import no_grad
from torch import triu
from torch.nn import Linear, ModuleList, init
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum
from torch_sparse import mul


# from dgl.python.dgl import add_self_loop
from nets.gcn import gcn_norm
from torch_geometric.utils import add_self_loops

from nets.jumping_weight import JumpingKnowledge


def get_conv(conv_type, input_dim, output_dim, alpha):      # from Rossi(LoG)
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
        # return DirGCNConv(input_dim, output_dim)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class DirGCNConv(torch.nn.Module):
    # def __init__(self, input_dim, output_dim, alpha):
    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")     # this is key: improve from 57 to 72

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")  #

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )

def count_upper_triangle_edges(self):
    row, col, _ = self.coo()
    upper_triangle_mask = row < col
    upper_triangle_count = upper_triangle_mask.sum().item()
    return upper_triangle_count

class DirGCNConv_2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.lin = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(4)])

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])

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
        self.mlp = None
        # if args.mlpIn:
        #     nhid = 64
        #     self.mlp = torch.nn.Linear(input_dim, output_dim)
        #     num_scale += 1
        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge:
            input_dim_jk = output_dim * num_scale if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.linjk = Linear(input_dim_jk, output_dim)


    def forward(self, x, edge_index):
        # x0= x
        device = edge_index.device
        if self.First_self_loop == 1:
            edge_index, _ = add_self_loops(edge_index, fill_value=1)
        elif self.First_self_loop == -1:
            edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        num_nodes = x.shape[0]

        if self.rm_gen_sloop:
            rm_gen_sLoop = True
        else:
            rm_gen_sLoop = False

        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                self.adj_norm = get_norm_adj(adj, norm=self.inci_norm)     # this is key: improve from 57 to 72

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                self.adj_t_norm = get_norm_adj(adj_t, norm=self.inci_norm)  #
                # print('edge number(A, At):', sparse_all(self.adj_norm), sparse_all(self.adj_t_norm))

            # if self.adj_norm_in_out is None and not (self.beta == -1 and self.beta == -1):
            if self.adj_norm_in_out is None:
                # if :
                #     break
                self.adj_norm_in_out = get_norm_adj(adj @ adj_t,norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_in = get_norm_adj(adj_t @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_in_in = get_norm_adj(adj @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_out = get_norm_adj(adj_t @ adj_t, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)

                self.norm_list = [self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out]
                # print('edge_num of AAt, AtA, AA, AtAt: ',
                #       sparse_all(self.adj_norm_in_out, k=1),
                #       sparse_all(self.adj_norm_out_in, k=1),
                #       sparse_all(self.adj_norm_in_in, k=1),
                #       sparse_all(self.adj_norm_out_out, k=1))

                if self.differ_AA:
                    Union_A_AA, Intersect_A_AA, diff_0 = share_edge(self.adj_norm_in_in, self.adj_norm, self.adj_t_norm)
                    Union_A_AtAt, Intersect_A_AtAt, diff_t = share_edge(self.adj_norm_out_out, self.adj_norm, self.adj_t_norm)
                elif self.differ_AAt:
                    Union_A_AAt,  Intersect_A_AAt, diff_0= share_edge(self.adj_norm_in_out, self.adj_norm, self.adj_t_norm)
                    Union_A_AtA, Intersect_A_AtA, diff_t = share_edge(self.adj_norm_out_in, self.adj_norm, self.adj_t_norm)
                if self.differ_AA or self.differ_AAt:
                    indices = torch.stack([torch.tensor(pair) for pair in diff_0], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor1 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_norm = get_norm_adj(sparse_tensor1, norm=self.inci_norm).to(self.adj_t_norm.device())

                    indices = torch.stack([torch.tensor(pair) for pair in diff_t], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor2 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_t_norm = get_norm_adj(sparse_tensor2, norm=self.inci_norm).to(self.adj_t_norm.device())
                if 3 in (self.alpha, self.beta, self.gama) and self.adj_intersection is None:
                    self.adj_intersection = intersection_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_intersection_in_out = intersection_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_intersection_in_in = intersection_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

                if 2 in (self.alpha, self.beta, self.gama) and self.adj_union is None:
                    self.adj_union = union_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_union_in_out = union_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_union_in_in = union_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

            out1 = aggregate(x, self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.adj_intersection, self.adj_union,  inci_norm=self.inci_norm)
            if not (self.beta == -1 and self.gama == -1):
                out2 = aggregate(x, self.beta, self.linx[0], self.norm_list[0], self.linx[1], self.norm_list[1], self.adj_intersection_in_out, self.adj_union_in_out, inci_norm=self.inci_norm)
                out3 = aggregate(x, self.gama, self.linx[2], self.norm_list[2], self.linx[3], self.norm_list[3], self.adj_intersection_in_in, self.adj_union_in_in, inci_norm=self.inci_norm)
            else:
                # out2 = out3 = torch.zeros_like(out1)
                out2 = torch.zeros_like(out1)
                out3 = torch.zeros_like(out1)
            # out2 += 1*self.lin[1](x)
            # a = 1*self.lin[1](x)
            # b = 1*self.lin[2](x)
            # c = 1*self.lin[3](x)
            # out2 += 1*self.lin[1](x) + self.lin[2](x) + self.lin[3](x)
            # out3 += 2*self.lin[1](x)+ 2*self.lin[2](x)

            # out2 += 1 * self.lin[1](x)
            # out3 += 1 * self.lin[2](x)

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

        if self.mlp:
            # x = torch.cat((self.mlp(x0), x), dim=-1)
            # x = self.conv2_1(x)

            x += self.mlp(x)

        if self.BN_model:
            x = self.batch_norm2(x)


        return x

def getHP(adj, device):
    num_nodes = adj.sparse_sizes()[0]

    # Create an identity matrix in COO format
    identity_indices = torch.arange(num_nodes, device=device)
    identity_indices = torch.stack([identity_indices, identity_indices])
    identity_values = torch.ones(num_nodes, device=adj.device())

    # Convert the adjacency matrix to COO format
    adj_row, adj_col, adj_values = adj.coo()  # Unpack three values
    adj_indices = torch.stack([adj_row, adj_col])
    # adj_indices, adj_values = adj.coo()

    # Combine the identity and adjacency matrices
    combined_indices = torch.cat([identity_indices, adj_indices], dim=1)
    combined_values = torch.cat([identity_values, -adj_values])

    # Create a new SparseTensor
    I_adj = SparseTensor(
        row=combined_indices[0],
        col=combined_indices[1],
        value=combined_values,
        sparse_sizes=(num_nodes, num_nodes)
    )

    # Optionally, you might want to coalesce the tensor to combine duplicate entries
    I_adj = I_adj.coalesce()
    return I_adj

class HighFreConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim,  root_weight=False)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=False)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=False) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-gat':
            # heads = args.heads
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim*heads, heads=heads)for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim*heads)
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
        # self.mlp = None
        # if args.mlp:
        #     self.mlp = torch.nn.Linear(input_dim, output_dim)

        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge:
            input_dim_jk = output_dim * num_scale if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.lin = Linear(input_dim_jk, output_dim)


    def forward(self, x, edge_index):
        device = edge_index.device
        edge_index, _ = remove_self_loops(edge_index)

        row, col = edge_index
        num_nodes = x.shape[0]

        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                self.adj_norm = get_norm_adj(adj, norm=self.inci_norm)     # this is key: improve from 57 to 72

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                self.adj_t_norm = get_norm_adj(adj_t, norm=self.inci_norm)  #

                self.adj_norm = getHP(self.adj_norm, device)
                self.adj_t_norm = getHP(self.adj_t_norm, device)

            # if self.adj_norm_in_out is None:
            #     self.adj_norm_in_out = get_norm_adj(adj @ adj_t,norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
            #     self.adj_norm_out_in = get_norm_adj(adj_t @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
            #     self.adj_norm_in_in = get_norm_adj(adj @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
            #     self.adj_norm_out_out = get_norm_adj(adj_t @ adj_t, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
            #
            #     self.norm_list = [self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out]
            #
            #     if self.differ_AA:
            #         Union_A_AA, Intersect_A_AA, diff_0 = share_edge(self.adj_norm_in_in, self.adj_norm, self.adj_t_norm)
            #         Union_A_AtAt, Intersect_A_AtAt, diff_t = share_edge(self.adj_norm_out_out, self.adj_norm, self.adj_t_norm)
            #     elif self.differ_AAt:
            #         Union_A_AAt,  Intersect_A_AAt, diff_0= share_edge(self.adj_norm_in_out, self.adj_norm, self.adj_t_norm)
            #         Union_A_AtA, Intersect_A_AtA, diff_t = share_edge(self.adj_norm_out_in, self.adj_norm, self.adj_t_norm)
            #     if self.differ_AA or self.differ_AAt:
            #         indices = torch.stack([torch.tensor(pair) for pair in diff_0], dim=0).t()
            #         row = indices[0]
            #         col = indices[1]
            #         sparse_tensor1 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            #         self.adj_norm = get_norm_adj(sparse_tensor1, norm=self.inci_norm).to(self.adj_t_norm.device())
            #
            #         indices = torch.stack([torch.tensor(pair) for pair in diff_t], dim=0).t()
            #         row = indices[0]
            #         col = indices[1]
            #         sparse_tensor2 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            #         self.adj_t_norm = get_norm_adj(sparse_tensor2, norm=self.inci_norm).to(self.adj_t_norm.device())
            #     if 3 in (self.alpha, self.beta, self.gama) and self.adj_intersection is None:
            #         self.adj_intersection = intersection_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
            #         self.adj_intersection_in_out = intersection_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
            #         self.adj_intersection_in_in = intersection_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)
            #
            #     if 2 in (self.alpha, self.beta, self.gama) and self.adj_union is None:
            #         self.adj_union = union_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
            #         self.adj_union_in_out = union_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
            #         self.adj_union_in_in = union_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

            out1 = aggregate(x, self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.adj_intersection, self.adj_union,  inci_norm=self.inci_norm)
            if not (self.beta == -1 and self.gama == -1):
                out2 = aggregate(x, self.beta, self.linx[0], self.norm_list[0], self.linx[1], self.norm_list[1], self.adj_intersection_in_out, self.adj_union_in_out, inci_norm=self.inci_norm)
                out3 = aggregate(x, self.gama, self.linx[2], self.norm_list[2], self.linx[3], self.norm_list[3], self.adj_intersection_in_in, self.adj_union_in_in, inci_norm=self.inci_norm)
            else:
                out2 = out3 = torch.zeros_like(out1)
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
                out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in, self.Intersect_beta, self.Union_beta)
                out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out, self.Intersect_gama, self.Union_gama)
            else:
                out2 = out3 = torch.zeros_like(out1)

        else:
            raise NotImplementedError

        xs = [out1, out2, out3]
        # if self.mlp:
        #     xs.append(self.mlp(x))

        if self.jumping_knowledge_inner:
            x = self.jump(xs)
            x = self.lin(x)
        else:
            x = sum(out for out in xs)

        # if self.mlp:
        #     x = torch.cat((self.mlp(x), x), dim=-1)
            # torch.vstack(self.mlp(x), x)

        if self.BN_model:
            x = self.batch_norm2(x)


        return x

class RanConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim,  root_weight=False)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=False)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=False) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-gat':
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim*heads, heads=heads)for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim*heads)
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
            self.lin = Linear(input_dim_jk, output_dim)


    def forward(self, x, edge_index):
        device = edge_index.device
        if self.First_self_loop == 1:

            edge_index, _ = add_self_loops(edge_index, fill_value=1)
        elif self.First_self_loop == 'remove':
            edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        num_nodes = x.shape[0]

        if self.rm_gen_sloop == 'remove':
            rm_gen_sLoop = True
        else:
            rm_gen_sLoop = False

        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                edge_weight = torch.rand(len(row)) * (10000 - 0.0001) + 0.0001
                edge_weight = edge_weight.to(device)
                self.adj_norm = directed_norm_weight(adj, edge_weight)     # this is key: improve from 57 to 72

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                edge_weight_t = edge_weight[adj_t.storage.csr2csc()]
                # edge_weight_t = torch.rand(len(row)) * (10000 - 0.0001) + 0.0001
                self.adj_t_norm = directed_norm_weight(adj_t, edge_weight_t)

            if self.adj_norm_in_out is None:
                self.adj_norm_in_out = get_norm_adj(adj @ adj_t,norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_in = get_norm_adj(adj_t @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_in_in = get_norm_adj(adj @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_out = get_norm_adj(adj_t @ adj_t, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)

                self.norm_list = [self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out]

                if self.differ_AA:
                    Union_A_AA, Intersect_A_AA, diff_0 = share_edge(self.adj_norm_in_in, self.adj_norm, self.adj_t_norm)
                    Union_A_AtAt, Intersect_A_AtAt, diff_t = share_edge(self.adj_norm_out_out, self.adj_norm, self.adj_t_norm)
                elif self.differ_AAt:
                    Union_A_AAt,  Intersect_A_AAt, diff_0= share_edge(self.adj_norm_in_out, self.adj_norm, self.adj_t_norm)
                    Union_A_AtA, Intersect_A_AtA, diff_t = share_edge(self.adj_norm_out_in, self.adj_norm, self.adj_t_norm)
                if self.differ_AA or self.differ_AAt:
                    indices = torch.stack([torch.tensor(pair) for pair in diff_0], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor1 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_norm = get_norm_adj(sparse_tensor1, norm=self.inci_norm).to(self.adj_t_norm.device())

                    indices = torch.stack([torch.tensor(pair) for pair in diff_t], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor2 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_t_norm = get_norm_adj(sparse_tensor2, norm=self.inci_norm).to(self.adj_t_norm.device())
                if 3 in (self.alpha, self.beta, self.gama) and self.adj_intersection is None:
                    self.adj_intersection = intersection_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_intersection_in_out = intersection_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_intersection_in_in = intersection_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

                if 2 in (self.alpha, self.beta, self.gama) and self.adj_union is None:
                    self.adj_union = union_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_union_in_out = union_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_union_in_in = union_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

            out1 = aggregate(x, self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.adj_intersection, self.adj_union,  inci_norm=self.inci_norm)
            if not (self.beta == -1 and self.gama == -1):
                out2 = aggregate(x, self.beta, self.linx[0], self.norm_list[0], self.linx[1], self.norm_list[1], self.adj_intersection_in_out, self.adj_union_in_out, inci_norm=self.inci_norm)
                out3 = aggregate(x, self.gama, self.linx[2], self.norm_list[2], self.linx[3], self.norm_list[3], self.adj_intersection_in_in, self.adj_union_in_in, inci_norm=self.inci_norm)
            else:
                out2 = out3 = torch.zeros_like(out1)
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
                out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in, self.Intersect_beta, self.Union_beta)
                out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out, self.Intersect_gama, self.Union_gama)
            else:
                out2 = out3 = torch.zeros_like(out1)

        else:
            raise NotImplementedError

        xs = [out1, out2, out3]

        if self.jumping_knowledge_inner:
            x = self.jump(xs)
            x = self.lin(x)
        else:
            x = sum(out for out in xs)

        if self.BN_model:
            x = self.batch_norm2(x)


        return x

class DirConv_tSNE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args, visualize=False):
        super().__init__()
        self.visual = visualize

        self.input_dim = input_dim
        self.output_dim = output_dim

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim,  root_weight=False)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=False)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=False) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-gat':
            # heads = args.heads
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim*heads, heads=heads)for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim*heads)
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
        # self.mlp = None
        # if args.mlp:
        #     self.mlp = torch.nn.Linear(input_dim, output_dim)
        #     num_scale += 1
        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge:
            input_dim_jk = output_dim * num_scale if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.lin = Linear(input_dim_jk, output_dim)


    def forward(self, x, edge_index, y, epoch):
        device = edge_index.device
        if self.First_self_loop == 1:

            edge_index, _ = add_self_loops(edge_index, fill_value=1)
        elif self.First_self_loop == 'remove':
            edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index
        num_nodes = x.shape[0]

        if self.rm_gen_sloop == 'remove':
            rm_gen_sLoop = True
        else:
            rm_gen_sLoop = False

        if self.conv_type == 'dir-gcn':
            if self.adj_norm is None:
                adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                self.adj_norm = get_norm_adj(adj, norm=self.inci_norm)     # this is key: improve from 57 to 72

                adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                self.adj_t_norm = get_norm_adj(adj_t, norm=self.inci_norm)  #
                # print('edge number(A, At):', sparse_all(self.adj_norm), sparse_all(self.adj_t_norm))

            if self.adj_norm_in_out is None:
                self.adj_norm_in_out = get_norm_adj(adj @ adj_t,norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_in = get_norm_adj(adj_t @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_in_in = get_norm_adj(adj @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                self.adj_norm_out_out = get_norm_adj(adj_t @ adj_t, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)

                self.norm_list = [self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out]
                # print('edge_num of AAt, AtA, AA, AtAt: ',
                #       sparse_all(self.adj_norm_in_out, k=1),
                #       sparse_all(self.adj_norm_out_in, k=1),
                #       sparse_all(self.adj_norm_in_in, k=1),
                #       sparse_all(self.adj_norm_out_out, k=1))

                if self.differ_AA:
                    Union_A_AA, Intersect_A_AA, diff_0 = share_edge(self.adj_norm_in_in, self.adj_norm, self.adj_t_norm)
                    Union_A_AtAt, Intersect_A_AtAt, diff_t = share_edge(self.adj_norm_out_out, self.adj_norm, self.adj_t_norm)
                elif self.differ_AAt:
                    Union_A_AAt,  Intersect_A_AAt, diff_0= share_edge(self.adj_norm_in_out, self.adj_norm, self.adj_t_norm)
                    Union_A_AtA, Intersect_A_AtA, diff_t = share_edge(self.adj_norm_out_in, self.adj_norm, self.adj_t_norm)
                if self.differ_AA or self.differ_AAt:
                    indices = torch.stack([torch.tensor(pair) for pair in diff_0], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor1 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_norm = get_norm_adj(sparse_tensor1, norm=self.inci_norm).to(self.adj_t_norm.device())

                    indices = torch.stack([torch.tensor(pair) for pair in diff_t], dim=0).t()
                    row = indices[0]
                    col = indices[1]
                    sparse_tensor2 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_t_norm = get_norm_adj(sparse_tensor2, norm=self.inci_norm).to(self.adj_t_norm.device())
                if 3 in (self.alpha, self.beta, self.gama) and self.adj_intersection is None:
                    self.adj_intersection = intersection_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_intersection_in_out = intersection_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_intersection_in_in = intersection_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

                if 2 in (self.alpha, self.beta, self.gama) and self.adj_union is None:
                    self.adj_union = union_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                    self.adj_union_in_out = union_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                    self.adj_union_in_in = union_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

            out1 = aggregate(x, self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.adj_intersection, self.adj_union,  inci_norm=self.inci_norm)
            if not (self.beta == -1 and self.gama == -1):
                out2 = aggregate(x, self.beta, self.linx[0], self.norm_list[0], self.linx[1], self.norm_list[1], self.adj_intersection_in_out, self.adj_union_in_out, inci_norm=self.inci_norm)
                out3 = aggregate(x, self.gama, self.linx[2], self.norm_list[2], self.linx[3], self.norm_list[3], self.adj_intersection_in_in, self.adj_union_in_in, inci_norm=self.inci_norm)
            else:
                out2 = out3 = torch.zeros_like(out1)
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
                out2 = aggregate_index(x, self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in, self.Intersect_beta, self.Union_beta)
                out3 = aggregate_index(x, self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out, self.Intersect_gama, self.Union_gama)
            else:
                out2 = out3 = torch.zeros_like(out1)

        else:
            raise NotImplementedError

        xs = [out1, out2, out3]
        # if self.mlp:
        #     xs.append(self.mlp(x))

        if self.jumping_knowledge_inner:
            x = self.jump(xs)
            x = self.lin(x)
        else:
            x = sum(out for out in xs)
        if (self.visual and epoch%100 == 0 and self.training)  or (epoch<10 ):
            with torch.no_grad():
                # visualize_batch_norm_effect_QQ(x, y, epoch)
                # visualize_batch_norm_effect_PCA(x, y, edge_index, epoch)
                visualize_batch_norm_effect_tSNE(x, y, edge_index, epoch)


        if self.BN_model:
            x = self.batch_norm2(x)

        return x

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from scipy import stats

def visualize_batch_norm_effect_QQ(X, y, epoch, feature_indices=None, num_features=4):
    # Ensure X is on CPU and convert to numpy
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X

    # Get the number of samples and features in the data
    num_samples, num_total_features = X_np.shape
    print(f"X shape: {X_np.shape}")

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(num_total_features)
    batch_norm.eval()  # Set to evaluation mode
    X_bn_np = batch_norm(torch.tensor(X_np).float()).detach().numpy()

    # Handle y
    if y is not None:
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        print(f"y shape: {y_np.shape}")
        if y_np.ndim == 1 and len(y_np) == num_samples:
            unique_labels = np.unique(y_np)
            colors = y_np
        elif y_np.shape == (1, num_samples):  # If y is a 1xN array
            y_np = y_np.flatten()
            unique_labels = np.unique(y_np)
            colors = y_np
        else:
            print("Warning: y shape doesn't match X. Using default coloring.")
            y_np = np.zeros(num_samples)
            unique_labels = [0]
            colors = y_np
    else:
        y_np = np.zeros(num_samples)
        unique_labels = [0]
        colors = y_np

    # If feature_indices is not provided, randomly select num_features
    if feature_indices is None:
        feature_indices = np.random.choice(num_total_features, min(num_features, num_total_features), replace=False)
    elif isinstance(feature_indices, int):
        feature_indices = [feature_indices]  # Convert single integer to list

    # Filter out any feature indices that are out of bounds
    feature_indices = [idx for idx in feature_indices if idx < num_total_features]

    if not feature_indices:
        print("No valid feature indices provided. Please check your input.")
        return

    num_features = min(len(feature_indices), 4)  # Limit to 4 features maximum

    # Create subplots
    fig, axes = plt.subplots(num_features, 2, figsize=(20, 5*num_features))
    # fig.subplots_adjust(hspace=10, wspace=2)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    if num_features == 1:
        axes = axes.reshape(1, -1)

    for i, feature_idx in enumerate(feature_indices[:num_features]):
        # Get the data for the current feature
        orig_data = X_np[:, feature_idx]
        bn_data = X_bn_np[:, feature_idx]

        # Original data plot
        ax_orig = axes[i, 0]
        ax_orig.set_title(f'Original - Feature {feature_idx}, Epoch {epoch}')
        osm, osr = stats.probplot(orig_data, dist="norm", fit=False)
        scatter_orig = ax_orig.scatter(osm, osr, c=colors, cmap='viridis', alpha=0.5, marker='o')
        ax_orig.plot([np.min(osm), np.max(osm)], [np.min(osr), np.max(osr)], 'r--')
        # ax_orig.set_xlabel('Theoretical Quantiles')
        # ax_orig.set_ylabel('Sample Quantiles')
        plt.colorbar(scatter_orig, ax=ax_orig, label='Label')

        # Batch normalized data plot
        ax_bn = axes[i, 1]
        ax_bn.set_title(f'Batch Normalized - Feature {feature_idx}, Epoch {epoch}')
        bsm, bsr = stats.probplot(bn_data, dist="norm", fit=False)
        scatter_bn = ax_bn.scatter(bsm, bsr, c=colors, cmap='viridis', alpha=0.5, marker='x')
        ax_bn.plot([np.min(bsm), np.max(bsm)], [np.min(bsr), np.max(bsr)], 'r--')
        # ax_bn.set_xlabel('Theoretical Quantiles')
        # ax_bn.set_ylabel('Sample Quantiles')
        plt.colorbar(scatter_bn, ax=ax_bn, label='Label')

    plt.tight_layout()
    plt.show()


def visualize_batch_norm_effect_PCA_highContrast(X, y, edge_index, epoch, n_components=2, random_state=42):
    # Ensure X is on CPU
    X = X.cpu()

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(X.shape[1])
    batch_norm.eval()  # Set to evaluation mode
    X_bn = batch_norm(X)

    # Convert to numpy for PCA
    X_np = X.numpy()
    X_bn_np = X_bn.numpy()

    # If y is a tensor, convert it to numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Apply PCA to original data
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_np)

    # Function to create and save a single plot
    def create_and_save_plot(data, title, filename):
        plt.figure(figsize=(12, 10))

        # Set a dark background
        plt.style.use('dark_background')

        # Plot edges with a bright, high-contrast color
        for (src, dst) in edge_index.t().tolist():
            plt.plot(data[[src, dst], 0], data[[src, dst], 1], color='#00FF00', alpha=0.3, linewidth=0.5, zorder=1)

        # Plot nodes using a high-contrast colormap
        scatter = plt.scatter(data[:, 0], data[:, 1], c=y, cmap='plasma', s=50, edgecolor='white', linewidth=0.5, zorder=2)

        plt.title(title, fontsize=16, fontweight='bold', color='white')
        plt.xlabel('PC 1', fontsize=12, fontweight='bold', color='white')
        plt.ylabel('PC 2', fontsize=12, fontweight='bold', color='white')

        # Add a colorbar with a descriptive label
        cbar = plt.colorbar(scatter)
        cbar.set_label('Class', fontsize=12, fontweight='bold', color='white')

        # Add explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.text(0.05, 0.95, f'Explained variance ratio:\nPC1: {explained_variance_ratio[0]:.2f}\nPC2: {explained_variance_ratio[1]:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, color='white',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', boxstyle='round,pad=0.5'))

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"High-contrast visualization saved as {filename}")

    # Create and save plot for original data
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_and_save_plot(X_pca, f'PCA of Original Data (Epoch {epoch})',
                         f'pca_{current_time}_epoch_{epoch}_high_contrast.png')


def visualize_batch_norm_effect_PCA(X, y, edge_index, epoch, n_components=2, random_state=42):
    # Ensure X is on CPU
    X = X.cpu()

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(X.shape[1])
    batch_norm.eval()  # Set to evaluation mode
    X_bn = batch_norm(X)

    # Convert to numpy for PCA
    X_np = X.numpy()
    X_bn_np = X_bn.numpy()

    # If y is a tensor, convert it to numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Apply PCA to original data
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_np)

    # Function to create and save a single plot
    def create_and_save_plot(data, title, filename):
        plt.figure(figsize=(10, 8))

        # Plot edges
        for (src, dst) in edge_index.t().tolist():
            plt.plot(data[[src, dst], 0], data[[src, dst], 1], color='gray', alpha=0.5, zorder=1)

        # Plot nodes
        scatter = plt.scatter(data[:, 0], data[:, 1], c=y, cmap='rainbow', zorder=2)
        plt.title(title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.colorbar(scatter)

        # Add explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.text(0.05, 0.95, f'Explained variance ratio: {explained_variance_ratio[0]:.2f}, {explained_variance_ratio[1]:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Visualization saved as {filename}")

    # Create and save plot for original data
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_and_save_plot(X_pca, f'PCA of Original Data in epoch {epoch}',
                         f'pca_{current_time}_epoch_{epoch}-original_data.png')

def visualize_batch_norm_effect_PCA_BN(X, y, epoch, n_components=2, random_state=42):
    # Ensure X is on CPU
    X = X.cpu()

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(X.shape[1])
    batch_norm.eval()  # Set to evaluation mode
    X_bn = batch_norm(X)

    # Convert to numpy for PCA
    X_np = X.numpy()
    X_bn_np = X_bn.numpy()

    # If y is a tensor, convert it to numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Apply PCA to original data
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_np)

    # Apply the same PCA transformation to batch normalized data
    # X_bn_pca = pca.transform(X_bn_np)
    #
    # wasserstein_dist = [
    #     wasserstein_distance(X_pca[:, i], X_bn_pca[:, i]) for i in range(n_components)
    # ]
    # mean_dist = np.mean(pairwise_distances(X_pca, X_bn_pca))
    # correlation = np.corrcoef(X_pca.flatten(), X_bn_pca.flatten())[0, 1]

    # Function to create and save a single plot
    def create_and_save_plot(data, title, filename):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis')
        plt.title(title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.colorbar(scatter)

        # Add explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.text(0.05, 0.95, f'Explained variance ratio: {explained_variance_ratio[0]:.2f}, {explained_variance_ratio[1]:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Visualization saved as {filename}")

    # Create and save plot for original data
    create_and_save_plot(X_pca, f'PCA of Original Data in epoch {epoch}',
                         f'pca_epoch_{epoch}-original_data.png')

    # Create and save plot for batch normalized data
    # create_and_save_plot(X_bn_pca, f'PCA of Batch Normalized Data in epoch {epoch}',
    #                      f'pca_epoch_{epoch}-batch_norm_data.png')
from datetime import datetime

def visualize_batch_norm_effect_tSNE(X, y, edge_index, epoch, perplexity=30, random_state=42):
    # Ensure X is on CPU
    X = X.cpu()

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(X.shape[1])
    batch_norm.eval()  # Set to evaluation mode
    X_bn = batch_norm(X)

    # Convert to numpy for t-SNE
    X_np = X.numpy()
    X_bn_np = X_bn.numpy()

    # If y is a tensor, convert it to numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Apply t-SNE to original data
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_np)

    # Apply t-SNE to batch normalized data
    X_bn_tsne = tsne.fit_transform(X_bn_np)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot original data
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='rainbow',  zorder=2)
    ax1.set_title('t-SNE of Original Data in epoch '+str(epoch))
    ax1.set_xlabel('t-SNE feature 1')
    ax1.set_ylabel('t-SNE feature 2')
    plt.colorbar(scatter1, ax=ax1)

    # Plot edges for original data
    for (src, dst) in edge_index.t().tolist():
        ax1.plot([X_tsne[src, 0], X_tsne[dst, 0]],
                 [X_tsne[src, 1], X_tsne[dst, 1]],
                 'k-', alpha=0.1, linewidth=0.5, zorder=1)

    # Plot batch normalized data
    scatter2 = ax2.scatter(X_bn_tsne[:, 0], X_bn_tsne[:, 1], c=y, cmap='rainbow',  zorder=2)
    ax2.set_title('t-SNE of Batch Normalized Data in epoch '+str(epoch))
    ax2.set_xlabel('t-SNE feature 1')
    ax2.set_ylabel('t-SNE feature 2')
    plt.colorbar(scatter2, ax=ax2)

    # Plot edges for batch normalized data
    for (src, dst) in edge_index.t().tolist():
        ax2.plot([X_bn_tsne[src, 0], X_bn_tsne[dst, 0]],
                 [X_bn_tsne[src, 1], X_bn_tsne[dst, 1]],
                 'k-', alpha=0.1, linewidth=0.5,  zorder=1)

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f't-SNE_comparison_with_edges_{current_time}_epoch_{epoch}.png')
    # plt.show()
    # plt.show(block=False)
    # plt.pause(2)
    plt.close()

def visualize_batch_norm_effect_tSNE_noEdge(X, y, epoch, perplexity=30, random_state=42):
    # Ensure X is on CPU
    X = X.cpu()

    # Create and apply BatchNorm layer
    batch_norm = nn.BatchNorm1d(X.shape[1])
    batch_norm.eval()  # Set to evaluation mode
    X_bn = batch_norm(X)

    # Convert to numpy for t-SNE
    X_np = X.numpy()
    X_bn_np = X_bn.numpy()

    # If y is a tensor, convert it to numpy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Apply t-SNE to original data
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_np)

    # Apply t-SNE to batch normalized data
    X_bn_tsne = tsne.fit_transform(X_bn_np)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot original data
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
    ax1.set_title('t-SNE of Original Data in epoch'+str(epoch) )
    ax1.set_xlabel('t-SNE feature 1')
    ax1.set_ylabel('t-SNE feature 2')
    plt.colorbar(scatter1, ax=ax1)

    # Plot batch normalized data
    scatter2 = ax2.scatter(X_bn_tsne[:, 0], X_bn_tsne[:, 1], c=y, cmap='tab10')
    ax2.set_title('t-SNE of Batch Normalized Data in epoch'+str(epoch) )
    ax2.set_xlabel('t-SNE feature 1')
    ax2.set_ylabel('t-SNE feature 2')
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f't-SNE_comparison_{current_time}_epoch_{epoch}.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

class DirGCNConv_sloop(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args, jk_sl):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if args.conv_type == 'dir-gcn':
            self.lin_src_to_dst = Linear(input_dim, output_dim)
            self.lin_dst_to_src = Linear(input_dim, output_dim)

            self.linx = nn.ModuleList([Linear(input_dim, output_dim) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-sage':
            self.lin_src_to_dst = SAGEConv(input_dim, output_dim,  root_weight=False)
            self.lin_dst_to_src = SAGEConv(input_dim, output_dim, root_weight=False)

            self.linx = nn.ModuleList([SAGEConv(input_dim, output_dim, root_weight=False) for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif args.conv_type == 'dir-gat':
            heads = 1
            self.lin_src_to_dst = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)
            self.lin_dst_to_src = GATConv(input_dim, output_dim*heads, heads=heads, add_self_loops=False)

            self.linx = nn.ModuleList([GATConv(input_dim, output_dim*heads, heads=heads)for i in range(4)])

            self.batch_norm2 = nn.BatchNorm1d(output_dim*heads)
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

        num_scale = 2
        # self.mlp = None
        # if args.mlp:
        #     self.mlp = torch.nn.Linear(input_dim, output_dim)
        #     num_scale += 1
        jumping_knowledge = args.jk_inner
        self.jumping_knowledge_inner = jumping_knowledge
        if jumping_knowledge:
            input_dim_jk = output_dim * num_scale if jumping_knowledge == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=input_dim, num_layers=3)
            self.lin = Linear(input_dim_jk, output_dim)

        self.jumping_knowledge_sloop = jk_sl
        if self.jumping_knowledge_sloop:
            num_scale_sl = 2
            input_dim_jk_sl = output_dim * num_scale_sl if self.jumping_knowledge_sloop == "cat" else output_dim
            self.jump = JumpingKnowledge(mode=self.jumping_knowledge_sloop, channels=input_dim, num_layers=3)
            self.lin = Linear(input_dim_jk_sl, output_dim)

    def forward(self, x, edge_index, flag):
        if self.rm_gen_sloop == 'remove':
            rm_gen_sLoop = True
        else:
            rm_gen_sLoop = False

        device = edge_index.device

        edge_index_add, _ = add_self_loops(edge_index, fill_value=1)
        # edge_index_rm, _ = remove_self_loops(edge_index)
        edge_index_list = [edge_index_add,  edge_index]

        x_sloop = []
        if isinstance(x, torch.Tensor):
            x0 = [x.clone(), x.clone(), x.clone()]
        elif len(x) == 2:
            x0 = x
        for i, edge_index_temp in enumerate(edge_index_list):
            row, col = edge_index_temp
            num_nodes = x0[i].shape[0]

            if self.conv_type == 'dir-gcn':
                if self.adj_norm is None or flag:
                # if flag:

                    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_norm = get_norm_adj(adj, norm=self.inci_norm)     # this is key: improve from 57 to 72

                    adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                    self.adj_t_norm = get_norm_adj(adj_t, norm=self.inci_norm)  #
                    # print('edge number(A, At):', sparse_all(self.adj_norm), sparse_all(self.adj_t_norm))

                if self.adj_norm_in_out is None or flag:
                # if flag:

                    self.adj_norm_in_out = get_norm_adj(adj @ adj_t,norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                    self.adj_norm_out_in = get_norm_adj(adj_t @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                    self.adj_norm_in_in = get_norm_adj(adj @ adj, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)
                    self.adj_norm_out_out = get_norm_adj(adj_t @ adj_t, norm=self.inci_norm, rm_gen_sLoop=rm_gen_sLoop)


                    self.norm_list = [self.adj_norm_in_out, self.adj_norm_out_in, self.adj_norm_in_in, self.adj_norm_out_out]
                    # print('edge_num of AAt, AtA, AA, AtAt: ',
                    #       sparse_all(self.adj_norm_in_out, k=1),
                    #       sparse_all(self.adj_norm_out_in, k=1),
                    #       sparse_all(self.adj_norm_in_in, k=1),
                    #       sparse_all(self.adj_norm_out_out, k=1))

                    if self.differ_AA:
                        Union_A_AA, Intersect_A_AA, diff_0 = share_edge(self.adj_norm_in_in, self.adj_norm, self.adj_t_norm)
                        Union_A_AtAt, Intersect_A_AtAt, diff_t = share_edge(self.adj_norm_out_out, self.adj_norm, self.adj_t_norm)
                    elif self.differ_AAt:
                        Union_A_AAt,  Intersect_A_AAt, diff_0= share_edge(self.adj_norm_in_out, self.adj_norm, self.adj_t_norm)
                        Union_A_AtA, Intersect_A_AtA, diff_t = share_edge(self.adj_norm_out_in, self.adj_norm, self.adj_t_norm)
                    if self.differ_AA or self.differ_AAt:
                        indices = torch.stack([torch.tensor(pair) for pair in diff_0], dim=0).t()
                        row = indices[0]
                        col = indices[1]
                        sparse_tensor1 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                        self.adj_norm = get_norm_adj(sparse_tensor1, norm=self.inci_norm).to(self.adj_t_norm.device())

                        indices = torch.stack([torch.tensor(pair) for pair in diff_t], dim=0).t()
                        row = indices[0]
                        col = indices[1]
                        sparse_tensor2 = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
                        self.adj_t_norm = get_norm_adj(sparse_tensor2, norm=self.inci_norm).to(self.adj_t_norm.device())
                    if 3 in (self.alpha, self.beta, self.gama) and self.adj_intersection is None:
                        self.adj_intersection = intersection_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                        self.adj_intersection_in_out = intersection_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                        self.adj_intersection_in_in = intersection_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)

                    if 2 in (self.alpha, self.beta, self.gama) and self.adj_union is None:
                        self.adj_union = union_adj_norm(self.adj_norm, self.adj_t_norm, self.inci_norm, device)
                        self.adj_union_in_out = union_adj_norm(self.norm_list[0], self.norm_list[1], self.inci_norm, device)
                        self.adj_union_in_in = union_adj_norm(self.norm_list[2], self.norm_list[3], self.inci_norm, device)


                out1 = aggregate(x0[i], self.alpha, self.lin_src_to_dst, self.adj_norm, self.lin_dst_to_src, self.adj_t_norm, self.adj_intersection, self.adj_union,  inci_norm=self.inci_norm)
                if not (self.beta == -1 and self.gama == -1):
                    out2 = aggregate(x0[i], self.beta, self.linx[0], self.norm_list[0], self.linx[1], self.norm_list[1], self.adj_intersection_in_out, self.adj_union_in_out, inci_norm=self.inci_norm)
                    out3 = aggregate(x0[i], self.gama, self.linx[2], self.norm_list[2], self.linx[3], self.norm_list[3], self.adj_intersection_in_in, self.adj_union_in_in, inci_norm=self.inci_norm)
                else:
                    out2 = out3 = torch.zeros_like(out1)
            elif self.conv_type in ['dir-gat', 'dir-sage']:
                # edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index_t = torch.stack([edge_index_temp[1], edge_index_temp[0]], dim=0)
                if not(self.beta == -1 and self.gama == -1) and self.edge_in_in is None:
                    self.edge_in_out, self.edge_out_in, self.edge_in_in, self.edge_out_out =get_higher_edge_index(edge_index_temp, num_nodes, rm_gen_sLoop=rm_gen_sLoop)
                    self.Intersect_alpha, self.Union_alpha = edge_index_u_i(edge_index_temp, edge_index_t)
                    self.Intersect_beta, self.Union_beta = edge_index_u_i(self.edge_in_out, self.edge_out_in)
                    self.Intersect_gama, self.Union_gama = edge_index_u_i(self.edge_in_in, self.edge_out_out)

                    if self.differ_AA:
                        diff_0 = remove_shared_edges(self.edge_in_in, edge_index_temp, edge_index_t)
                        diff_1 = remove_shared_edges(self.edge_out_out, edge_index_temp, edge_index_t)
                    elif self.differ_AAt:
                        diff_0 = remove_shared_edges(self.edge_in_out, edge_index_temp, edge_index_t)
                        diff_1 = remove_shared_edges(self.edge_out_in, edge_index_temp, edge_index_t)
                    if self.differ_AA or self.differ_AAt:
                        edge_index_temp = diff_0
                        edge_index_t = diff_1

                out1 = aggregate_index(x0[i], self.alpha, self.lin_src_to_dst, edge_index_temp, self.lin_dst_to_src, edge_index_t, self.Intersect_alpha, self.Union_alpha)
                if not (self.beta == -1 and self.gama == -1):
                    out2 = aggregate_index(x0[i], self.beta, self.linx[0], self.edge_in_out, self.linx[1], self.edge_out_in, self.Intersect_beta, self.Union_beta)
                    out3 = aggregate_index(x0[i], self.gama, self.linx[2], self.edge_in_in, self.linx[3], self.edge_out_out, self.Intersect_gama, self.Union_gama)
                else:
                    out2 = out3 = torch.zeros_like(out1)

            else:
                raise NotImplementedError

            xs = [out1, out2, out3]
            # if self.mlp:
            #     xs.append(self.mlp(x))

            if self.jumping_knowledge_inner:
                x = self.jump(xs)
                x = self.lin(x)
            else:
                x = sum(out for out in xs)

            if self.BN_model:
                x = self.batch_norm2(x)

            x_sloop.append(x)

        # are_close = torch.allclose(x_sloop[0], x_sloop[1], rtol=1e-5, atol=1e-8)
        # print(f"Are the tensors (0 and 1) close enough to be considered equal? {are_close}")
        #
        # are_close = torch.allclose(x_sloop[0], x_sloop[2], rtol=1e-5, atol=1e-8)
        # print(f"Are the tensors (0 and 2) close enough to be considered equal? {are_close}")
        if self.jumping_knowledge_sloop:
            x = self.jump(x_sloop)
            x = self.lin(x)
        else:
            # x = sum(out for out in x_sloop)
            x = x_sloop


        return x


def to_edge_set(edge_index):
    # Convert edge_index to a set of tuples
    return set(tuple(edge) for edge in edge_index.t().tolist())


def remove_shared_edges(self_edge_index, edge_index, edge_index_t):
    # Convert edge indices to sets of tuples
    self_edge_set = to_edge_set(self_edge_index)
    edge_set = to_edge_set(edge_index)
    edge_t_set = to_edge_set(edge_index_t)

    # Find shared edges
    shared_edges = self_edge_set.intersection(edge_set).union(self_edge_set.intersection(edge_t_set))

    # Remove shared edges from self_edge_set
    filtered_edges = self_edge_set.difference(shared_edges)

    # Convert the filtered edges back to tensor format
    filtered_edge_list = list(filtered_edges)
    filtered_edge_tensor = torch.tensor(filtered_edge_list).t()

    return filtered_edge_tensor

def edge_index_u_i(edge_index, edge_index_t):
    # Convert edge_index and edge_index_t to sets of tuples
    edge_set = set(tuple(edge) for edge in edge_index.t().tolist())
    edge_set_t = set(tuple(edge) for edge in edge_index_t.t().tolist())

    # Compute the union of both edge sets
    union_edge_set = edge_set.union(edge_set_t)

    # Convert the set of tuples back to tensor format
    union_edge_list = list(union_edge_set)
    union_edge_tensor = torch.tensor(union_edge_list).t()

    intersection_edge_set = edge_set.intersection(edge_set_t)

    # Convert the set of tuples back to tensor format
    intersection_edge_list = list(intersection_edge_set)
    intersection_edge_tensor = torch.tensor(intersection_edge_list).t()

    return intersection_edge_tensor, union_edge_tensor


def edge_index_to_adj(edge_index, num_nodes):
    import torch.sparse as sp
    # Create the adjacency matrix from edge_index
    row = edge_index[0]
    col = edge_index[1]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    return adj

def get_index(adj_aat):
    # adj_aat = adj_aat.coalesce()  # Convert to sparse tensor with COO format
    row, col = adj_aat.storage._row, adj_aat.storage._col
    # row, col = adj_aat.indices()
    edge_index_aat = torch.stack([row, col], dim=0)

    return edge_index_aat
def get_higher_edge_index(edge_index, num_nodes, rm_gen_sLoop=0):
    adj = edge_index_to_adj(edge_index, num_nodes)
    adj_in_out = adj @ adj.t()
    adj_out_in =  adj.t() @ adj

    adj_aa = adj @ adj
    adj_out_out = adj.t() @ adj.t()

    if rm_gen_sLoop:
        adj_in_out[torch.arange(num_nodes), torch.arange(num_nodes)] = 0
        adj_out_in[torch.arange(num_nodes), torch.arange(num_nodes)] = 0


    return get_index(adj_in_out), get_index(adj_out_in), get_index(adj_aa), get_index(adj_out_out)



def aggregate(x, alpha, lin0, adj0, lin1, adj1,  intersection, union, inci_norm='inci_norm'):
    if alpha == 2:
        out = lin0(union @ x)
    elif alpha == 3:
        out = lin0(intersection @ x)
    else:
        out = (1+alpha)*(alpha * lin0(adj0 @ x) + (1 - alpha) * lin1(adj1 @ x))

    return out

def aggregate_index(x, alpha, lin0, index0, lin1, index1,  intersection, union):
    if alpha == 2:
        out = lin0(x, union)
    elif alpha == 3:
        out = lin0(x, intersection)
    else:
        out = (1+alpha)*((1 - alpha) * lin0(x, index0) +  alpha * lin1(x, index1))

    # if conv_type == 'dir-sage':
    #     out = out + lin(x)
    return out

def union_adj_norm(adj0, adj1, inci_norm, device):
    # device = adj0.device

    row1 = adj0.storage.row()
    row2 = adj1.storage.row()
    # new_row = torch.cat(row1, row2)
    new_row = torch.cat((row1, row2), dim=0)
    col1 = torch.tensor(adj0.storage.col())
    col2 = torch.tensor(adj1.storage.col())
    new_col = torch.cat((col1, col2), dim=0)

    union_edges = torch.stack([new_row, new_col], dim=1)
    unique_edges = torch.unique(union_edges, dim=0)

    row = unique_edges[:, 0].to(device)
    col = unique_edges[:, 1].to(device)
    num_nodes = adj0.size(0)
    unique_edges = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    new_adj_norm = get_norm_adj(unique_edges, norm=inci_norm).to(device)

    return new_adj_norm


def intersection_adj_norm(adj0, adj1, inci_norm, device):
    row1 = adj0.storage.row()
    row2 = adj1.storage.row()
    col1 = adj0.storage.col()
    col2 = adj1.storage.col()

    # Stack the row and col tensors to get edge lists
    edges1 = torch.stack([row1, col1], dim=1)
    edges2 = torch.stack([row2, col2], dim=1)

    # Sort the edges to enable intersection using PyTorch operations
    edges1 = edges1[edges1[:, 0].argsort()]
    edges2 = edges2[edges2[:, 0].argsort()]

    # Use torch.unique and torch's intersection logic to find common edges
    edges1_set = torch.unique(edges1, dim=0)
    edges2_set = torch.unique(edges2, dim=0)

    # Find common edges by using broadcasting and comparison
    with torch.no_grad():
        try:
            # edges1_set = edges1_set.to_sparse()
            # edges2_set = edges2_set.to_sparse()
            intersection_mask = (edges1_set[:, None] == edges2_set).all(dim=2).any(dim=1)
        except:
            edges1_set = edges1_set.cpu()
            edges2_set = edges2_set.cpu()
            intersection_mask = (edges1_set[:, None] == edges2_set).all(dim=2).any(dim=1)

    intersection = edges1_set[intersection_mask]

    # Extract row and col from the intersection tensor
    row = intersection[:, 0].to(device)
    col = intersection[:, 1].to(device)
    num_nodes = adj0.size(0)
    unique_edges = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    new_adj_norm = get_norm_adj(unique_edges, norm=inci_norm).to(device)

    return new_adj_norm

def filter_upper_triangle(edges):
    """Filter edges to include only those in the upper triangle."""
    return {edge for edge in edges if edge[0] < edge[1]}
    # return torch.tensor(edge for edge in edges if edge[0] < edge[1])
def tensor_to_tuple(tensor):
    return tuple(map(int, tensor.cpu().numpy()))
def share_edge(m1, m2, m3=None):
    # import torch
    row1 = m1.storage.row()
    row2 = m2.storage.row()
    # new_row = torch.cat(row1, row2)
    new_row = torch.cat((row1, row2), dim=0)
    col1 = torch.tensor(m1.storage.col())
    col2 = torch.tensor(m2.storage.col())
    new_col = torch.cat((col1, col2), dim=0)

    union_edges = torch.stack([new_row, new_col], dim=1)
    unique_edges = torch.unique(union_edges, dim=0)

        # Convert tensors to sets of tuples for intersection
    edges1 = torch.stack([row1, col1], dim=1)
    edges2 = torch.stack([row2, col2], dim=1)
    set1 = set(map(tuple, edges1.tolist()))
    set2 = set(map(tuple, edges2.tolist()))

    # Find the intersection
    intersection = set1.intersection(set2)

    # Convert the result back to a tensor
    intersection_tensor = torch.tensor(list(intersection))
    # unique_edges = torch.tensor(list(unique_edges))
    # intersection_tensor = filter_upper_triangle(intersection_tensor)
    # intersection_tensor = torch.tensor(list(intersection_tensor), dtype=torch.int64)
    # unique_edges = filter_upper_triangle(unique_edges)

    difference0 = set1.difference(set2)
    if m3 is not None:
        row3 = m3.storage.row()
        col3 = torch.tensor(m3.storage.col())
        edges3 = torch.stack([row3, col3], dim=1)
        set3 = set(map(tuple, edges3.tolist()))
        difference = difference0.difference(set3)

    print('union, intersction, diff-A, diff-A-At:', len(unique_edges), len(intersection_tensor), len(difference0), len(difference))
    return unique_edges, intersection_tensor, torch.tensor(list(difference))



def sparse_triu(sparse_matrix, k=0):
    # count the non-zero edges in upper triangle, that what GCN takes in.
    row = sparse_matrix.storage.row()
    col = sparse_matrix.storage.col()
    values = sparse_matrix.storage.value()

    # Create mask for upper triangular elements
    mask = col - row >= k

    # Apply mask
    new_values = values[mask]

    return (new_values != 0).sum().item()

def sparse_all(sparse_matrix, k=0):
    # count the non-zero edges in upper triangle, that what GCN takes in.
    # row = sparse_matrix.storage.row()
    # col = sparse_matrix.storage.col()
    values = sparse_matrix.storage.value()

    # Create mask for upper triangular elements
    # mask = col - row >= k

    # Apply mask
    # new_values = values[mask]

    return (values != 0).sum().item()



class DirGCNConv_Qin(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight):
        device = edge_index.device
        num_nodes = edge_index.max().item() + 1

        adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to(device)

        # Perform sparse matrix multiplication
        out = torch.sparse.mm(adj_matrix, x)

        return self.lin_src_to_dst(out)

def get_norm_adj(adj, norm, rm_gen_sLoop=0):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=0)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj, rm_gen_sLoop=rm_gen_sLoop)
    elif norm is None or norm==0:
        return adj
    else:
        raise ValueError(f"{norm} normalization is not supported")

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))



def add_self_loop_qin(adj):
    """Add self-loops to the adjacency matrix."""
    device= adj.device()
    row, col, value = adj.coo()

    # Get the size of the adjacency matrix (number of nodes)
    num_nodes = adj.sparse_sizes()[0]

    # Create self-loop indices (diagonal elements)
    self_loop_indices = torch.arange(num_nodes).to(device)

    # Create the new row, col, and value arrays
    new_row = torch.cat([row, self_loop_indices], dim=0)
    new_col = torch.cat([col, self_loop_indices], dim=0)

    if value is not None:
        # Assuming self-loop weight of 1.0, adjust this if needed
        self_loop_value = torch.ones(num_nodes, dtype=value.dtype, device=value.device)
        new_value = torch.cat([value, self_loop_value], dim=0)
    else:
        new_value = None

    # Create the new adjacency matrix with self-loops added
    adj = SparseTensor(row=new_row, col=new_col, value=new_value, sparse_sizes=adj.sparse_sizes())
    return adj


def directed_norm(adj, rm_gen_sLoop=False):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    device = adj.device()
    in_deg = sparsesum(adj, dim=0)
    # in_deg = torch_sparse.sum(adj, dim=0).to(torch.float)
    in_deg_inv_sqrt = in_deg.pow(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    # out_deg = torch_sparse.sum(adj, dim=1).to(torch.float)
    out_deg_inv_sqrt = out_deg.pow(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    out_deg_inv_sqrt = out_deg_inv_sqrt.to(device)
    in_deg_inv_sqrt = in_deg_inv_sqrt.to(adj.device())

    adj0 = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj1 = mul(adj0, in_deg_inv_sqrt.view(1, -1))

    # adj0 = torch_sparse.mul(adj, out_deg_inv_sqrt.view(-1, 1))
    # adj1 = torch_sparse.mul(adj0, in_deg_inv_sqrt.view(1, -1))

    return adj1

def directed_norm_weight(adj, edge_weight=None, rm_gen_sLoop=False):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    if edge_weight is not None:
        row, col, _ = adj.coo()
        new_values = edge_weight
        adj = torch_sparse.SparseTensor(row=row, col=col, value=new_values, sparse_sizes=adj.sparse_sizes())
        return adj        # this to test without inci-norm

    device = adj.device()
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    out_deg_inv_sqrt = out_deg_inv_sqrt.to(device)
    in_deg_inv_sqrt = in_deg_inv_sqrt.to(adj.device())

    adj0 = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj1 = mul(adj0, in_deg_inv_sqrt.view(1, -1))

    return adj1

def get_model(num_features,  n_cls, args):
    return GNN(
        num_features=num_features,
        hidden_dim=args.hid_dim,
        num_layers=args.layer,
        num_classes=n_cls,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alphaDir,
        learn_alpha=args.learn_alpha,
    )

class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )

class GNN(torch.nn.Module):     # from Rossi(LoG paper)
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1/2,
        learn_alpha=False,
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.hid_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([DirGCNConv_2(nfeat, output_dim, args)])
        else:
            self.convs = ModuleList([DirGCNConv_2(nfeat, nhid, args)])
            for _ in range(layer - 2):
                self.convs.append(DirGCNConv_2(nhid, nhid, args))
            self.convs.append(DirGCNConv_2(nhid, output_dim, args))

        num_scale = layer
        if jumping_knowledge:
            input_dim = hidden_dim * num_scale if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
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

class High_Frequent(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.hid_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([HighFreConv(nfeat, output_dim, args)])
        else:
            self.convs = ModuleList([HighFreConv(nfeat, nhid, args)])
            for _ in range(layer - 2):
                self.convs.append(HighFreConv(nhid, nhid, args))
            self.convs.append(HighFreConv(nhid, output_dim, args))

        num_scale = layer
        if jumping_knowledge:
            input_dim = hidden_dim * num_scale if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear


    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
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

class RandomNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.hid_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([RanConv(nfeat, output_dim, args)])
        else:
            self.convs = ModuleList([RanConv(nfeat, nhid, args)])
            for _ in range(layer - 2):
                self.convs.append(RanConv(nhid, nhid, args))
            self.convs.append(RanConv(nhid, output_dim, args))

        num_scale = layer
        if jumping_knowledge:
            input_dim = hidden_dim * num_scale if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear


    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
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

class ScaleNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.hid_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        if layer == 1:
            self.convs = ModuleList([DirConv_tSNE(nfeat, output_dim, args, visualize=True )])
        else:
            self.convs = ModuleList([DirConv_tSNE(nfeat, nhid, args, visualize=False)])
            for _ in range(layer - 2):
                self.convs.append(DirConv_tSNE(nhid, nhid, args, visualize=False))
            self.convs.append(DirConv_tSNE(nhid, output_dim, args, visualize=True))

        num_scale = layer
        if jumping_knowledge:
            input_dim = hidden_dim * num_scale if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear


    def forward(self, x, edge_index, y, epoch):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, y, epoch)
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

class Sloop_JKNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        jumping_knowledge = args.jk
        layer = args.layer
        nhid = args.hid_dim
        hidden_dim = nhid
        normalize = args.normalize
        dropout = args.dropout
        nonlinear = args.nonlinear

        output_dim = nhid if jumping_knowledge else nclass
        jkSl_end = 'max'
        jkSl_inner = 0
        if layer == 1:
            self.convs = ModuleList([DirGCNConv_sloop(nfeat, output_dim, args, jk_sl=jkSl_end)])
        else:
            self.convs = ModuleList([DirGCNConv_sloop(nfeat, nhid, args, jk_sl=jkSl_inner)])
            for _ in range(layer - 2):
                self.convs.append(DirGCNConv_sloop(nhid, nhid, args, jk_sl=jkSl_inner))
            self.convs.append(DirGCNConv_sloop(nhid, output_dim, args, jk_sl=jkSl_end))

        if jumping_knowledge:
            n= layer*3
            input_dim = hidden_dim * n if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, nclass)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=layer)

        self.num_layers = layer
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.nonlinear = nonlinear

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, flag=1-i)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                if self.nonlinear:
                    if isinstance(x, list):
                        x = [F.relu(i) for i in x]
                    else:
                        x = F.relu(x)

                if isinstance(x, list):
                    x = [F.dropout(i, p=self.dropout, training=self.training) for i in x]
                else:
                    x = F.dropout(x, p=self.dropout, training=self.training)

                if self.normalize:
                    if isinstance(x, list):
                        x = [F.normalize(i, p=2, dim=1) for i in x]
                    else:
                        x = F.normalize(x, p=2, dim=1)
            if isinstance(x, list):
                xs.extend(x)
            else:
                xs.append(x)

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return x


from torch_geometric.nn import MessagePassing, APPNP
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, args):
        super().__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
