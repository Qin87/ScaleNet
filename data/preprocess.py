# externel
import torch
import csv, os
import numpy as np
import pickle as pk
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
from torch_geometric.datasets import WebKB, WikipediaNetwork



def load_syn(root, name=None):
    print(root + '.pk', 'rb')
    print(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))
    data = pk.load(open(root + '.pk', 'rb'))
    if os.path.isdir(root) is False:
        try:
            os.makedirs(root)
        except FileExistsError:
            print('Folder exists!')
    return [data]



