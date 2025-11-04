import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim         # Ben
import os
import yaml
import torch
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric
from torch_sparse import SparseTensor
import torch

def test_directed(edge_index):
    set_edges = set()
    bi_direct = 0
    self_loop = 0
    for i in range(edge_index.shape[1]):
        if edge_index[1][i].item() == edge_index[0][i].item():
            self_loop += 1
        edge_inv = frozenset([edge_index[1][i].item(), edge_index[0][i].item()])

        edge = frozenset([edge_index[0][i].item(), edge_index[1][i].item()])
        if edge_inv in set_edges:
            bi_direct += 1
        set_edges.add(edge)
    print("selfloop: {}, Num_bidirect_edges: {}, total_num_edges: {}".format(self_loop, bi_direct, edge_index.shape[1]))
    if bi_direct * 2 == edge_index.shape[1] - self_loop:
        return False
    return True

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)

class F1Scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, factor, patience):
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_F1_score = float('-inf')  # Set to negative infinity initially
        super().__init__(optimizer)

    def step(self, F1_score=None, epoch=None):
        if F1_score is not None:
            if F1_score > self.best_F1_score:
                self.best_F1_score = F1_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.counter = 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.factor
                    print(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']}")


def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "./best_hyperparameters.yml"
    # print(os.getcwd())
    # # os.chdir("..")      # Qin
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")
    # print(args)
    return args

def get_norm_adj(adj, norm, exponent=-0.25, W=None):
    if W is not None:
        row, col, value = adj.coo()
        if value is None:
            weighted_value = W
        else:
            weighted_value = value * W
        from torch_sparse import SparseTensor
        adj = SparseTensor(row=row, col=col, value=weighted_value,
                           sparse_sizes=adj.sparse_sizes())
        return adj
    elif norm == "sym":       # Din^(-0.5)ADin^(-0.5)
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "dir_ones":
        return directed_norm_ones(adj)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "col":
        return col_norm(adj)
    elif norm == "dir":
        return directed_norm(adj, exponent)
    elif norm is None or norm=="0":
        return adj
    elif norm == "opposite":
        return directed_opposite_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def directed_norm_ones(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.add_self_loops
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 1.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 1.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    return adj


def directed_opposite_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 1.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 1.0)

    adj = mul(adj, out_deg_inv_sqrt.view(1, -1))
    adj = mul(adj, in_deg_inv_sqrt.view(-1, 1))

    return adj

def col_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=0)
    scaled_inverted_col_sum = row_sum.pow_(-0.20)
    scaled_inverted_col_sum.masked_fill_(scaled_inverted_col_sum == float("inf"), 0.0)
    matrix = mul(adj, row_sum.view(1, -1))

    return matrix


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj, exponent):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(exponent)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(exponent)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    return adj

def print_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print(f"[{tag}] CUDA not available, skipping memory check")