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
        super(CrossEntropy, self).__init__()

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
