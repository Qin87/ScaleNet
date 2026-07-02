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
    best_params_file_path = "best_hyperparameters.yml"
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

def get_norm_adj(adj, norm, exponent=-0.25 ):
    if norm == "sym":       # Din^(-0.5)ADin^(-0.5)
        return gcn_norm(adj, add_self_loops=0)
    # elif norm == "dir_ones":
    #     return directed_norm_ones(adj)
    elif norm == "row":
        return row_norm(adj)
    # elif norm == "col":
    #     return col_norm(adj)
    elif norm == "dir":
        return directed_norm(adj, exponent)
    elif norm is None or norm==0  or norm=="0":
        return adj
    # elif norm == "opposite":
    #     return directed_opposite_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj, exponent=-0.5):
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