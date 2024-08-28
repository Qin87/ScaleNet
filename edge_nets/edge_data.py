import itertools
import sys
import time

import torch
import numpy as np
import pickle as pk
import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.data import Data
from torch import Tensor
from torch_sparse import SparseTensor, coalesce
# from stellargraph.data import EdgeSplitter    # can't install Ben
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling, dropout_adj
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import scipy
import os
from joblib import Parallel, delayed
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add

from nets.geometric_baselines import get_norm_adj


def fast_sparse_boolean_multi_hop(A, k):
    row, col = A._indices()     # the non-zero elements in the sparse tensor
    n = A.size(0)

    # Create dictionary for fast lookup
    neighbors = {i: set() for i in range(n)}        # Initializes a dictionary neighbors :  key i represents a node, and the corresponding value is an empty set
    for r, c in zip(row.tolist(), col.tolist()):
        neighbors[r].add(c)     # store the neighbors (nodes directly connected by edges) for each node

    def k_hop_neighbors(node, hops):
        if hops == 0:
            return {node}       # Returns a set containing only the node itself
        if hops == 1:
            return set(neighbors[node])     # Returns the set of immediate neighbors of the node
        prev_hop = k_hop_neighbors(node, hops - 1)
        current_hop = set()
        for prev_node in prev_hop:
            current_hop.update(neighbors[prev_node])
        return current_hop

    all_hops = []
    for hop in range(1, k + 1):
        new_edges = set()
        for node in range(n):
            hop_neighbors = k_hop_neighbors(node, hop)
            new_edges.update((node, h) for h in hop_neighbors if h != node)

        new_indices = torch.tensor(list(new_edges), dtype=torch.long).t()
        values = torch.ones(new_indices.size(1), dtype=torch.bool)
        all_hops.append(torch.sparse_coo_tensor(new_indices, values, (n, n)).coalesce())

    return tuple(all_hops)

def sub_adj(edge_index, prob, seed):
    sub_train, sub_test = train_test_split(edge_index.T, test_size=prob, random_state=seed)
    sub_train, sub_val = train_test_split(sub_train, test_size=0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T


def edges_positive(edge_index):
    # return true edges and reverse edges
    return edge_index, edge_index[[1, 0]]


def edges_negative(edge_index):
    from torch_geometric.utils import to_undirected

    size = edge_index.max().item() + 1
    adj = np.zeros((size, size), dtype=np.int8)
    adj[edge_index[0], edge_index[1]] = 1
    x, y = np.where((adj - adj.T) < 0)

    reverse = torch.from_numpy(np.c_[x[:, np.newaxis], y[:, np.newaxis]])
    undirected_index = to_undirected(edge_index)
    negative = negative_sampling(undirected_index, num_neg_samples=edge_index[0].shape[0], force_undirected=False)

    _from_, _to_ = negative[0].unsqueeze(0), negative[1].unsqueeze(0)
    neg_index = torch.cat((_from_, _to_), axis=0)
    # neg_index = torch.cat((reverse.T, neg_index), axis = 1)
    # print(edge_index.shape, reverse.shape, neg_index.shape)
    return reverse.T, neg_index


def split_negative(edge_index, prob, seed, neg_sampling=True):
    reverse, neg_index = edges_negative(edge_index)
    if neg_sampling:
        neg_index = torch.cat((reverse, neg_index), axis=1)
    else:
        neg_index = reverse

    sub_train, sub_test = train_test_split(neg_index.T, test_size=prob, random_state=seed)
    sub_train, sub_val = train_test_split(sub_train, test_size=0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T


def label_pairs_gen(pos, neg):
    pairs = torch.cat((pos, neg), axis=-1)
    label = np.r_[np.ones(len(pos[0])), np.zeros(len(neg[0]))]
    return pairs, label


def generate_dataset_2class(edge_index, splits=10, test_prob=0.6):
    # this function doesn't consider the connectivity during removing edges for validation/testing
    from torch_geometric.utils import to_undirected
    datasets = {}

    for i in range(splits):
        train, val, test = sub_adj(edge_index, prob=test_prob, seed=i * 10)
        train_neg, val_neg, test_neg = split_negative(edge_index, seed=i * 10, prob=test_prob)
        ############################################
        # training data
        ############################################
        # positive edges, reverse edges, negative edges
        datasets[i] = {}

        datasets[i]['graph'] = train
        datasets[i]['undirected'] = to_undirected(train).numpy().T

        rng = np.random.default_rng(i)

        datasets[i]['train'] = {}
        pairs, label = label_pairs_gen(train, train_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['train']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['train']['label'] = label[perm]

        ############################################
        # validation data
        ############################################
        # positive edges, reverse edges, negative edges

        datasets[i]['validate'] = {}
        pairs, label = label_pairs_gen(val, val_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['validate']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['validate']['label'] = label[perm]
        ############################################
        # test data
        ############################################
        # positive edges, reverse edges, negative edges

        datasets[i]['test'] = {}
        pairs, label = label_pairs_gen(test, test_neg)
        perm = rng.permutation(len(pairs[0]))
        datasets[i]['test']['pairs'] = pairs[:, perm].numpy().T
        datasets[i]['test']['label'] = label[perm]
    return datasets


# in-out degree calculation
# def in_out_degree(edge_index, size):
#     A = coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])), shape=(size, size), dtype=np.float32).tocsr()
#     out_degree = np.sum(A, axis=0).T
#     in_degree = np.sum(A, axis=1)
#     degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
#     return degree

def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix((np.ones(len(edge_index)), (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis = 0).T
    in_degree = np.sum(np.abs(A), axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def undirected_label2directed_label(adj, edge_pairs, task):
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = edge_pairs.copy()
    counter = 0
    for i, e in enumerate(edge_pairs):  # directed edges
        if adj[e[0], e[1]] + adj[e[1], e[0]] > 0:  # exists an edge
            if adj[e[0], e[1]] > 0:
                if adj[e[1], e[0]] == 0:  # rule out undirected edges
                    if counter % 2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                else:
                    new_edge_pairs[i] = [e[0], e[1]]
                    labels[i] = -1
            else:  # the other direction, and not an undirected edge
                if counter % 2 == 0:
                    labels[i] = 0
                    new_edge_pairs[i] = [e[1], e[0]]
                    counter += 1
                else:
                    labels[i] = 1
                    new_edge_pairs[i] = [e[0], e[1]]
                    counter += 1
        else:  # negative edges
            labels[i] = 2
            new_edge_pairs[i] = [e[0], e[1]]

    if task != 2:
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        rng = np.random.default_rng(1000)
        neg_half = rng.choice(neg, size=len(neg) - np.sum(labels == 0), replace=False)
        labels[neg_half] = -1
    return new_edge_pairs[labels >= 0], labels[labels >= 0]


def generate_dataset_3class(edge_index, size, save_path, splits=10, probs=[0.15, 0.05], task=2, label_dim=2):
    # print(os.getcwd())
    # print(sys.argv[0])
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_directory)
    save_file = save_path + 'task' + str(task) + 'dim' + str(label_dim) + 'prob' + str(int(probs[0] * 100)) + '_' + str(int(probs[1] * 100)) + '.pk'
    if os.path.exists(save_file):
        print('File exists!')
        d_results = pk.load(open(save_file, 'rb'))
        return d_results

    row, col = edge_index[0], edge_index[1]
    # adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    # A_dense = np.array(adj.todense())
    # edge_num = np.sum(A_dense)
    # print( "undirected rate:", 1.0*np.sum(A_dense * A_dense.T)/edge_num )

    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    # G = nx.from_scipy_sparse_matrix(A)  # create an undirected graph based on the adjacency
    G = nx.from_scipy_sparse_array(A)  # create an undirected graph based on the adjacency

    def iteration(ind):
        datasets = {}
        edge_splitter_test = EdgeSplitter(G)
        G_test, ids_test, _ = edge_splitter_test.train_test_split(p=float(probs[0]), method="global", keep_connected=True, seed=ind)
        ids_test, labels_test = undirected_label2directed_label(A, ids_test, task)

        edge_splitter_val = EdgeSplitter(G_test)
        G_val, ids_val, _ = edge_splitter_val.train_test_split(p=float(probs[1]), method="global", keep_connected=True, seed=ind)
        ids_val, labels_val = undirected_label2directed_label(A, ids_val, task)

        edge_splitter_train = EdgeSplitter(G_val)
        _, ids_train, _ = edge_splitter_train.train_test_split(p=0.99, method="global", keep_connected=False, seed=ind)
        ids_train, labels_train = undirected_label2directed_label(A, ids_train, task)

        # observation after removing edges for training/validation/testing
        edges = [e for e in G_val.edges]
        # convert back to directed graph
        oberved_edges = np.zeros((len(edges), 2), dtype=np.int32)
        undirected_edges = np.zeros((2 * len(G.edges), 2), dtype=np.int32)

        for i, e in enumerate(edges):
            if A[e[0], e[1]] > 0:
                oberved_edges[i, 0] = int(e[0])
                oberved_edges[i, 1] = int(e[1])
            if A[e[1], e[0]] > 0:
                oberved_edges[i, 0] = int(e[1])
                oberved_edges[i, 1] = int(e[0])

        for i, e in enumerate(G.edges):
            if A[e[0], e[1]] > 0 or A[e[1], e[0]] > 0:
                undirected_edges[i, :] = [int(e[1]), e[0]]
                undirected_edges[i + len(edges), :] = [int(e[0]), e[1]]
        if label_dim == 2:
            ids_train = ids_train[labels_train < 2]
            labels_train = labels_train[labels_train < 2]
            ids_test = ids_test[labels_test < 2]
            labels_test = labels_test[labels_test < 2]
            ids_val = ids_val[labels_val < 2]
            labels_val = labels_val[labels_val < 2]
        ############################################
        # training data
        ############################################
        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(oberved_edges.T).long()
        datasets[ind]['undirected'] = undirected_edges

        datasets[ind]['train'] = {}
        datasets[ind]['train']['pairs'] = ids_train
        datasets[ind]['train']['label'] = labels_train
        ############################################
        # validation data
        ############################################
        datasets[ind]['validate'] = {}
        datasets[ind]['validate']['pairs'] = ids_val
        datasets[ind]['validate']['label'] = labels_val
        ############################################
        # test data
        ############################################
        datasets[ind]['test'] = {}
        datasets[ind]['test']['pairs'] = ids_test
        datasets[ind]['test']['label'] = labels_test
        return datasets

    # use larger n_jobs if the number of cpus is enough
    try:
        p_data = Parallel(n_jobs=4)(delayed(iteration)(ind) for ind in range(20))
    except:
        p_data = Parallel(n_jobs=1)(delayed(iteration)(ind) for ind in range(20))

    d_results = {}
    for ind in p_data:
        split = list(ind.keys())[0]
        d_results[split] = ind[split]

    if os.path.isdir(save_path) == False:
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print('Folder exists!')

    if os.path.exists(save_file) == False:
        try:
            pk.dump(d_results, open(save_file, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
        except FileExistsError:
            print('File exists!')

    return d_results


#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
def get_pr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):     # from DiGCN, Qin never change
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    else:
        edge_weight = torch.FloatTensor(edge_weight).to(edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    # pagerank p
    p_pr = (1.0 - alpha) * p_dense + alpha / num_nodes * torch.ones((num_nodes, num_nodes), dtype=dtype, device=p.device)

    eig_value, left_vector = scipy.linalg.eig(p_pr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    # assert val[0] == 1.0

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi / pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_pr
    L = (torch.mm(torch.mm(pi_sqrt, p_pr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_pr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # # let little possbility connection to 0, make L sparse
    # L[ L < (1/num_nodes)] = 0
    # L[ L < 5e-4] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    # from DiGCN, Qin never change
    if edge_weight == None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes + 1, num_nodes + 1]))
    p_v[0:num_nodes, 0:num_nodes] = (1 - alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi / pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(selfloop, edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    if selfloop == 'add':
        edge_index, _ = add_self_loops(edge_index.long(), fill_value=1, num_nodes=num_nodes)  # with selfloop, QiG get better
    elif selfloop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



def Qin_get_second_directed_adj0(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    p_dense = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L = L_in
    L[L_out == 0] = 0        # intersection

    # L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    edge_index = L_indices
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def union_edge_index(edge_index):
    # Concatenate the original edge_index with its reverse
    union = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Remove duplicates
    union = torch.unique(union, dim=1)

    return union

def Qin_get_directed_adj(args, edge_index, num_nodes, dtype, edge_weight=None):
    selfloop = args.First_self_loop
    norm = args.inci_norm
    device = edge_index.device
    if selfloop == 'add':
        edge_index, _ = add_self_loops(edge_index.long(), fill_value=1, num_nodes=num_nodes)       # with selfloop, QiG get better
    elif selfloop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    edge_index = torch.unique(edge_index, dim=1).to(device)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1).to(device)

    # type 1: conside different inci-norm
    if norm == 'dir':
        row, col = edge_index
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        # all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        edge_weight = adj_norm.storage.value()
    elif norm == 'sym':
        # type 2: only GCN_norm
        edge_weight = normalize_row_edges(edge_index, num_nodes).to(device)


    return edge_index,  edge_weight

def WCJ_get_directed_adj(args, edge_index, num_nodes, dtype, edge_weight=None):
    # norm = args.inci_norm
    norm = 'sym'
    self_loop = args.First_self_loop
    W_degree = args.W_degree
    # random value to edge weights
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    if self_loop == 'add':
        edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value=1, num_nodes=num_nodes)  # with selfloop, QiG get better
    elif self_loop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg0 = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)  # row degree
    deg1 = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes).to(device)  # col degree
    deg2 = deg0 + deg1

    # plt.hist(deg0.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Original Distribution of  degree0:NPZ')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if W_degree == 0:  # in-degree
        edge_weight = deg0[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg0")
    elif W_degree == 1:     # out-degree
        edge_weight = deg1[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg1")
    elif W_degree == 2:     # total-degree
        edge_weight = deg2[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg2")
    elif W_degree == 3:     # random number in [1,100]
        edge_weight = torch.randint(1, 101, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 300:     # random number in [1,10000]
        edge_weight = torch.randint(1, 10001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 30000:     # random number in [1,1000000]
        edge_weight = torch.randint(1, 1000001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 400:  # random number in [0.001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.999 + 0.001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 40000:  # random number in [0.00001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.99999 + 0.00001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 4:  # random number in [0.1,1]       # random number in [0.1,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.9 + 0.1
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 5:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == 50:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        edge_weight = torch.abs(torch.sin(edge_weight))
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == -3:   # three peaks
        edge_weight = trimodal_distribution(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -2:   # two peaks
        edge_weight = trimodal_distribution2(edge_index.size(1), edge_index.device, dtype)
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
    elif W_degree == -4:   # two peaks
        edge_weight = trimodal_distribution4(edge_index.size(1), edge_index.device, dtype)
    else:
        NotImplementedError('Not Implemented edge-weight type')

    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Original Distribution of  WiG-2 edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    if norm == 'sym':
        # row normalization
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    else:
        # type 1: conside different inci-norm
        row, col = edge_index
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        # all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        edge_weight = adj_norm.storage.value()

        # type 2: only GCN_norm
        # edge_weight = normalize_row_edges(edge_index, num_nodes).to(device)

    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Normalized Edge weight range: [{min_val}, {max_val}]")

    # # plt.xlim(0, 2)
    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Normalized Distribution of  WiG-2 edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    return edge_index,  edge_weight

# def Qin_get_appr_directed_adj0(alpha, edge_index, num_nodes, dtype, edge_weight=None):
#     """
#     based on get_appr_directed_adj, all weights to 1, this is equal to GCN(norm inside GCNConV is False, and better than GCN with norm)
#     QinDiG worked for telegram
#         alpha:
#         edge_index:
#         num_nodes:
#         dtype:
#         edge_weight:
#
#     Returns:
#
#     """
#
#     device = edge_index.device
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                      device=edge_index.device)
#     fill_value = 1
#     edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)
#     edge_index = edge_index.to(device)
#
#     return edge_index,  edge_weight


def get_appr_directed_adj2(selfloop, alpha, edge_index, num_nodes, dtype, edge_weight=None):
    device = edge_index.device

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    if selfloop == 'add':
        edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value=1, num_nodes=num_nodes)  # with selfloop, QiG get better
    elif selfloop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)
    deg_inv = deg.pow(-1).to(device)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # plt.hist(deg.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of degree:Cora')       # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense().to(device)
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1])).to(device)     # dummy node
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense      # original P
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v.cpu()   # for p_ppr.numpy()  # this is new P with one dummy node

    p_ppr_sparse = csr_matrix(p_ppr.numpy())
    eig_value, left_vector = scipy.linalg.eig(p_ppr_sparse.toarray(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real).to(device)     # Qin ask: why only real?       # converting a NumPy array containing eigenvalues to a PyTorch tensor
    left_vector = torch.from_numpy(left_vector.real).to(device)
    val, ind = eig_value.sort(descending=True)
    #  sort the tensor eig_value in descending order and also get the corresponding indices of the sorted elements
    #
    pi = left_vector[:,ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]    # X+1 back to X  # remove the dummy node
    p_ppr = p_dense.to(device)
    pi = pi/pi.sum()  # norm pi
    #
    # # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0
    pi_inv_sqrt = pi.pow(-0.5)      # (183,) to (183,)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag().to(device)     # (183,) to (183, 183)
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag().to(device)

    # L_appr   # actually, L_appr= I-L, so this L is the equivalent their version of symmetric_A of digraph
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0       # a bit time consuming
    L[torch.isnan(L)] = 0    # make nan to 0
    # L[torch.isnan(L)] = 1  # make nan to 1   # TODO delete it after testing(Qin use 1, original is 0)---worse

    # L = (p_ppr + p_ppr.t()) / 2.0       # TODO delete it after testing

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()     # the indices of all nonzero elements in the input tensor L, arranged as a tensor where each column represents the indices of a nonzero element
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices      # their transformed edges of this symmetric_A of digraph
    edge_weight = L_values

    # edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    # edge_weight= torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (0.1 - 0.00001) + 0.0001   # TODO delete this just for test
    # perm = torch.randperm(edge_weight.size(0), device=edge_weight.device)
    # edge_weight = edge_weight[perm]

    # edge_weight = torch.abs(torch.sin(edge_weight))     # TODO delete
    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Original Edge weight range: [{min_val}, {max_val}]")



    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Normalized Edge weight range: [{min_val}, {max_val}]")
    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of  DiG edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # delete TODO
    # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    # edge_weight = torch.cat([edge_weight, torch.ones((num_nodes,), device=edge_weight.device)])
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    return edge_index, edge_weight

import torch
import torch.distributions as dist
def trimodal_distribution(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 100.0, 1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001, 0.1, 100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(3, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples

def trimodal_distribution4(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 0.01, 900, 5000.0], device=device, dtype=dtype)
    stds = torch.tensor([10, 100, 1000, 5000], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(4, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples

def trimodal_distribution2(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001,  1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001,  100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(2, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples


def fast_sparse_boolean_multi_hop_union(A, k):
    row, col = A._indices()
    n = A.size(0)

    # Create dictionaries for fast lookup of both incoming and outgoing neighbors
    out_neighbors = {i: set() for i in range(n)}
    in_neighbors = {i: set() for i in range(n)}
    for r, c in zip(row.tolist(), col.tolist()):
        out_neighbors[r].add(c)
        in_neighbors[c].add(r)

    def k_hop_neighbors(node, hops, neighbors_dict):
        if hops == 0:
            return {node}
        if hops == 1:
            return set(neighbors_dict[node])
        prev_hop = k_hop_neighbors(node, hops - 1, neighbors_dict)
        current_hop = set()
        for prev_node in prev_hop:
            current_hop.update(neighbors_dict[prev_node])
        return current_hop

    all_hops = []
    for hop in range(1, k + 1):
        new_edges = set()
        for node in range(n):
            # Compute outgoing k-hop neighbors
            out_hop_neighbors = k_hop_neighbors(node, hop, out_neighbors)
            # Compute incoming k-hop neighbors
            in_hop_neighbors = k_hop_neighbors(node, hop, in_neighbors)
            # Union of outgoing and incoming k-hop neighbors
            hop_neighbors = out_hop_neighbors.union(in_hop_neighbors)
            new_edges.update((node, h) for h in hop_neighbors if h != node)

        new_indices = torch.tensor(list(new_edges), dtype=torch.long).t()
        values = torch.ones(new_indices.size(1), dtype=torch.bool)
        all_hops.append(torch.sparse_coo_tensor(new_indices, values, (n, n)))

    return tuple(all_hops)

# def union_sparse_tensors(A_in, A_out):
#     device = A_in.device
#     indices_in = A_in.indices()
#     indices_out = A_out.indices()
#     combined_indices = torch.cat([indices_in, indices_out], dim=1)
#     unique_indices = torch.unique(combined_indices, dim=1).to(device)
#     values = torch.ones(unique_indices.size(1), dtype=torch.float32).to(device)
#
#     return torch.sparse_coo_tensor(unique_indices, values, size=A_in.size(), dtype=torch.float32)


def intersect_sparse_tensors(A_in, A_out):
    device = A_in.device

    indices_in = A_in.indices()
    indices_out = A_out.indices()

    A_in = A_in.to_dense()
    A_out = A_out.to_dense()

    A_in_hat = A_in.to(device)
    A_out_hat = A_out.to(device)

    A_in_hat[A_out == 0] = 0  # intersection
    A_out_hat[A_in == 0] = 0

    # L^{(2)}
    # intersection = (A_in_hat + A_out_hat) / 2.0
    intersection = A_in_hat

    indices = intersection.nonzero().t()
    values = torch.ones(indices.size(1), dtype=torch.float32).to(device)

    num_intersecting_edges = torch.count_nonzero(intersection)
    # print('!!!!',  time.time())

    return torch.sparse_coo_tensor(indices, values, size=intersection.size(), dtype=torch.float32)


def sparse_mm_chunked(A, B, chunk_size):
    """
    Perform sparse matrix multiplication in chunks to manage memory usage.
    """
    A = A.coalesce()
    B = B.coalesce()

    A_indices = A.indices()
    A_values = A.values()
    B_indices = B.indices()
    B_values = B.values()

    result_indices = []
    result_values = []

    num_chunks = (A.size(1) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, A.size(1))

        A_chunk = torch.sparse_coo_tensor(
            A_indices[:, (A_indices[1] >= start) & (A_indices[1] < end)],
            A_values[(A_indices[1] >= start) & (A_indices[1] < end)],
            (A.size(0), end - start)
        )

        B_chunk = torch.sparse_coo_tensor(
            B_indices[:, (B_indices[0] >= start) & (B_indices[0] < end)],
            B_values[(B_indices[0] >= start) & (B_indices[0] < end)],
            (end - start, B.size(1))
        )

        AB_chunk = torch.sparse.mm(A_chunk.to_dense(), B_chunk.to_dense()).to_sparse()

        result_indices.append(AB_chunk.indices())
        result_values.append(AB_chunk.values())

    result_indices = torch.cat(result_indices, dim=1)
    result_values = torch.cat(result_values)
    result = torch.sparse_coo_tensor(result_indices, result_values, (A.size(0), B.size(1))).coalesce()

    return result

def sparse_mm_safe(A, B):
    try:
        return torch.sparse.mm(A, B)
    except RuntimeError as e:
        if "CUDA error: insufficient resources" in str(e):
            print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
            return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
        else:
            raise e

def generate_possible_B_products(A, m):
    # List of matrices to be used in combinations (A and A transpose)
    elements = [A, A.t()]

    # Generate all possible combinations of A and A transpose of length k
    all_combinations = list(itertools.product(elements, repeat=m))

    # Compute the product for each combination
    results1 = []
    results2 = []
    for combination in all_combinations:
        B = combination[0]
        for mat in combination[1:]:
            B = sparse_mm_safe(B, mat)
        B1 = sparse_mm_safe(A, B)        # make sure the first is A
        B2 = sparse_mm_safe(A.t(), B)  # make sure the first is A.t()
        results1.append(sparse_mm_safe(B1, B1.t()))
        results2.append(sparse_mm_safe(B2, B2.t()))

    return [results1, results2]


def sparese_remove_self_loops(sparse_matrix):
    # Ensure the sparse matrix is in coalesced format (no duplicate entries)
    sparse_matrix = sparse_matrix.coalesce()

    # Create a mask to filter out diagonal elements (self-loops)
    mask = sparse_matrix.indices()[0] != sparse_matrix.indices()[1]

    # Apply the mask and create a new sparse tensor without self-loops
    return torch.sparse.FloatTensor(
        sparse_matrix.indices()[:, mask],
        sparse_matrix.values()[mask],
        sparse_matrix.size()
    )


# Removing self-loops from A_in and A_out

def sparse_boolean_multi_hopExhaust(args, A, k, mode='union'):
    selfloop = args.rm_gen_sloop

    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    A_out = sparse_mm_safe(A.t(), A)
    # num_nonzero_in = A_in._nnz()
    # num_nonzero_out = A_out._nnz()
    # print('number of edges:', num_nonzero_in, num_nonzero_out)

    if mode == 'union':
        A_result = A_in + A_out
        A_result = A_result.coalesce()
        A_result._values().clamp_(0, 1)  # Ensuring binary values
    else:   # intersection
        A_result = intersect_sparse_tensors(A_in, A_out)

    if selfloop == 'remove':
        A_result = sparese_remove_self_loops(A_result)
    all_hops = [A_result]

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        [in_list, out_list] = generate_possible_B_products(A, hop)
        for A_in, A_out in zip(in_list, out_list):
            if mode == 'union':
                A_result = A_in + A_out
                A_result = A_result.coalesce()
                A_result._values().clamp_(0, 1)  # Ensuring binary values
            else:
                A_result = intersect_sparse_tensors(A_in, A_out)

            # num_nonzero_result = A_result._nnz()
            # print('num of edges:', num_nonzero_result)
            if selfloop == 'remove':
                A_result = sparese_remove_self_loops(A_result)
            all_hops.append(A_result)

    return tuple(all_hops)

def sparse_boolean_multi_hop_DirGNN(has_1_order, rm_gen_self_loop, A, k):
    order_tuple_list = []

    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    order_tuple_0 = [A, A.t()]

    if k<1:
        return tuple(order_tuple_0)

    if has_1_order:
        order_tuple_list.append(order_tuple_0)

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A)
    # A_out = sparse_mm_safe(A.t(), A.t())
    A_out = A_in.t()

    B_in = sparse_mm_safe(A, A.t())
    # B_out = sparse_mm_safe(A, A.t())
    B_out = B_in.t()
    num_nonzero_in = A_in._nnz()
    num_nonzero_out = B_in._nnz()
    print('number of edges:', num_nonzero_in, num_nonzero_out)

    if rm_gen_self_loop == 'remove':
        order_tuple_1 = [sparse_remove_self_loops(A_in), sparse_remove_self_loops(B_in), sparse_remove_self_loops(A_out), sparse_remove_self_loops(B_out)]
    else:
        order_tuple_1 = [A_in, B_in, A_out, B_out]
    order_tuple_list.append(order_tuple_1)

    for hop in range(1, k):
        order_tuple_temp = []
        for edge_matrix in order_tuple_list[-1]:        # TODO : might improve efficiency for symmetry
            N_in = sparse_mm_safe(edge_matrix, A)
            N_out = sparse_mm_safe(edge_matrix, A.t())
            if rm_gen_self_loop == 'remove':
                order_tuple_temp.extend([sparse_remove_self_loops(N_in), sparse_remove_self_loops(N_out)])
            else:
                order_tuple_temp.extend([N_in, N_out])
        order_tuple_list.append(order_tuple_temp)

    return tuple(tensor for sub_list in order_tuple_list for tensor in sub_list)

def sparse_remove_self_loops(matrix):
    # Function to remove self-loops by setting diagonal elements to zero.
    # Extract the indices and values of the sparse matrix
    indices = matrix._indices()
    values = matrix._values()

    # Filter out diagonal elements (self-loops)
    mask = indices[0] != indices[1]
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Create a new sparse matrix without self-loops
    new_matrix = torch.sparse_coo_tensor(new_indices, new_values, matrix.size(), dtype=matrix.dtype)
    new_matrix = new_matrix.coalesce()

    return new_matrix

def sparse_boolean_multi_hop(args, A, k, mode='union'):
    selfloop = args.rm_gen_sloop
    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    def sparse_mm_safe(A, B):
        try:
            return torch.sparse.mm(A, B)
        except RuntimeError as e:
            if "CUDA error: insufficient resources" in str(e):
                print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
                return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
            else:
                raise e

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    A_out = sparse_mm_safe(A.t(), A)

    # A_in = sparse_mm_safe(A, A)
    # A_out = sparse_mm_safe(A.t(), A.t())
    if selfloop == 'remove':
        A_in = sparse_remove_self_loops(A_in)
        A_out = sparse_remove_self_loops(A_out)
    num_nonzero_in = A_in._nnz()
    num_nonzero_out = A_out._nnz()
    print('2-order number of edges(in, out):', num_nonzero_in, num_nonzero_out)

    if mode == 'union':
        A_result = A_in + A_out
        A_result = A_result.coalesce()
        A_result._values().clamp_(0, 1)  # Ensuring binary values
        if selfloop == 'remove':
            A_result = sparse_remove_self_loops(A_result)
        all_hops = [A_result]
    elif mode == 'intersection':
        A_result = intersect_sparse_tensors(A_in, A_out)
        if selfloop == 'remove':
            A_result = sparse_remove_self_loops(A_result)
        all_hops = [A_result]
    elif mode == 'separate':
        if selfloop == 'remove':
            A_in = sparse_remove_self_loops(A_in)
            A_out = sparse_remove_self_loops(A_out)
        all_hops = [A_in, A_out]
    else:
        raise NotImplementedError("Not Implemented mode: ", mode)

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        A_in = torch.sparse.mm(A, A_in)
        A_in = torch.sparse.mm(A_in, A.t())
        A_out = torch.sparse.mm(A.t(), A_out)
        A_out = torch.sparse.mm(A_out, A)

        if selfloop == 'remove':
            A_in = sparse_remove_self_loops(A_in)
            A_out = sparse_remove_self_loops(A_out)

        num_nonzero_in = A_in._nnz()
        num_nonzero_out = A_out._nnz()
        print(hop + 2, 'order num of edges(in, out): ', num_nonzero_in, num_nonzero_out)

        if mode == 'union':
            A_result = A_in + A_out
            A_result = A_result.coalesce()
            A_result._values().clamp_(0, 1)  # Ensuring binary values
            num_nonzero = A_result._nnz()
            print(hop + 2, 'order num of edges (union): ', num_nonzero)
            if selfloop == 'remove':
                A_result = sparse_remove_self_loops(A_result)
            all_hops.append(A_result)
        elif mode == 'intersection':
            A_result = intersect_sparse_tensors(A_in, A_out)
            num_nonzero = A_result._nnz()
            print(hop + 2, 'order num of edges (intersection): ', num_nonzero)
            if selfloop == 'remove':
                A_result = sparse_remove_self_loops(A_result)
            all_hops.append(A_result)
        elif mode == 'separate':
            if selfloop == 'remove':
                A_in = sparse_remove_self_loops(A_in)
                A_out = sparse_remove_self_loops(A_out)
            all_hops.extend([A_in, A_out])
        else:
            raise NotImplementedError("Not Implemented mode: ", mode)

        # num_nonzero_result = A_result._nnz()
        # print('num of edges:', num_nonzero_result)


    return tuple(all_hops)

def OneDirect_sparse_boolean_multi_hop(A, k):
    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    def sparse_mm_safe(A, B):
        try:
            return torch.sparse.mm(A, B)
        except RuntimeError as e:
            if "CUDA error: insufficient resources" in str(e):
                print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
                return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
            else:
                raise e

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    # A_out = sparse_mm_safe(A.t(), A)
    num_nonzero_in = A_in._nnz()
    # num_nonzero_out = A_out._nnz()
    print('number of edges:', num_nonzero_in)

    all_hops = [A_in]

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        A_in = torch.sparse.mm(A, A_in)
        A_in = torch.sparse.mm(A_in, A.t())
        # A_out = torch.sparse.mm(A.t(), A_out)
        # A_out = torch.sparse.mm(A_out, A)

        num_nonzero_in = A_in._nnz()
        # num_nonzero_out = A_out._nnz()
        print(hop + 2, 'order num of edges: ', num_nonzero_in)

        # if mode == 'union':
        #     A_result = A_in + A_out
        #     A_result = A_result.coalesce()
        #     A_result._values().clamp_(0, 1)  # Ensuring binary values
        # else:
        #     A_result = intersect_sparse_tensors(A_in, A_out)

        num_nonzero_result = A_in._nnz()
        print('num of edges:', num_nonzero_result)
        all_hops.append(A_in)

    return tuple(all_hops)


def dense_boolean_multi_hop_union(A, k):
    n = A.size(0)
    A_current = A.coalesce()

    # Initialize all_hops list with A (1-hop neighbors)
    all_hops = [A_current]

    # Compute k-hop neighbors using matrix multiplication
    for hop in range(1, k):
        A_next = torch.mm(A_current.to_dense(), A.to_dense())
        A_next = A_next.to_sparse()
        all_hops.append(A_next)
        A_current = A_next

    return tuple(all_hops)


def normalize_row_edges(edge_index, num_nodes, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float, device=device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # deg = torch.where(deg == 0, torch.tensor(float('inf'), device=deg.device), deg)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def sparse_difference(U, I, epsilon=1e-8):
    diff = (U - I).coalesce()
    mask = diff._values().abs() > epsilon
    return torch.sparse_coo_tensor(
        diff._indices()[:, mask],
        diff._values()[mask],
        diff.size()
    )


def sparse_intersection(U, I):
    # Perform element-wise multiplication
    intersection = U * I

    # Coalesce to combine any duplicate indices
    intersection = intersection.coalesce()

    return intersection

def Qin_get_second_directed_adj(args, edge_index, num_nodes, k, IsExhaustive, mode, norm='dir'):     #
    self_loop = args.First_self_loop
    device = edge_index.device
    if self_loop == 'add':
        edge_index, _ = add_self_loops(edge_index.long(), fill_value=1, num_nodes=num_nodes)  # with selfloop, QiG get better
    elif self_loop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    if mode != 'independent':
        if IsExhaustive:
            L_tuple = sparse_boolean_multi_hopExhaust(args, A, k - 1, mode)  # much slower
        else:
            L_tuple = sparse_boolean_multi_hop(args, A, k-1, mode)   # much slower
    else:       # independent
        if IsExhaustive:
            L_tupleU = sparse_boolean_multi_hopExhaust(args, A, k - 1, 'union')  # much slower
            L_tupleI = sparse_boolean_multi_hopExhaust(args, A, k - 1, 'intersection')  # much slower
        else:
            L_tupleU = sparse_boolean_multi_hop(args, A, k-1, 'union')   # much slower
            L_tupleI = sparse_boolean_multi_hop(args, A, k-1, 'intersection')   # much slower
        all_hops = list(L_tupleI)
        for U, I in zip(L_tupleU, L_tupleI):
            all_hops.append(sparse_difference(U, I))
        L_tuple = tuple(all_hops)

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        row, col = L._indices()
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        all_hops_weight.append(adj_norm.storage.value())

    return tuple(all_hop_edge_index), tuple(all_hops_weight)

def dir_normalize_edge_weights_origin(edge_index, num_nodes, edge_weights=None):
# from DirGNN
    # Compute out-degrees and in-degrees
    edge_index = edge_index.long()
    edge_weights = edge_weights.float()

    device = edge_index.device
    row, col = edge_index
    out_deg = torch.zeros(num_nodes, dtype=torch.float).to(device)
    in_deg = torch.zeros(num_nodes, dtype=torch.float).to(device)

    out_deg.scatter_add_(0, row, edge_weights)
    in_deg.scatter_add_(0, col, edge_weights)

    # Compute the inverse square root of the degrees
    out_deg_inv_sqrt = torch.pow(out_deg, -0.5)
    out_deg_inv_sqrt[out_deg_inv_sqrt == float('inf')] = 0

    in_deg_inv_sqrt = torch.pow(in_deg, -0.5)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0

    # Normalize the edge weights
    normalized_edge_weights = edge_weights * out_deg_inv_sqrt[row] * in_deg_inv_sqrt[col]

    return edge_index, normalized_edge_weights


# def dir_normalize_edge_weights(edge_index, num_nodes, edge_weights=None):
def dir_normalize_edge_weights(row, col, value,  num_nodes, edge_weights=None):
# from DirGNN
    # Compute out-degrees and in-degrees
    # device = edge_index.device()
    # edge_index = edge_index.long()
    # if edge_weights is None:
    #     edge_weights = torch.ones((num_nodes, num_nodes), dtype=torch.float).to(device)
    # else:
    #     edge_weights = edge_weights.float()
    edge_weights = value.to(torch.float32)

    device = row.device
    # # row, col = edge_index[:,0], edge_index[:,1]
    # row, col = edge_index.row, edge_index.col
    out_deg = torch.zeros(num_nodes, dtype=torch.float).to(device)
    in_deg = torch.zeros(num_nodes, dtype=torch.float).to(device)

    out_deg.scatter_add_(0, row, edge_weights)
    in_deg.scatter_add_(0, col, edge_weights)

    # Compute the inverse square root of the degrees
    out_deg_inv_sqrt = torch.pow(out_deg, -0.5)
    out_deg_inv_sqrt[out_deg_inv_sqrt == float('inf')] = 0

    in_deg_inv_sqrt = torch.pow(in_deg, -0.5)
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0

    # Normalize the edge weights
    normalized_edge_weights = edge_weights * out_deg_inv_sqrt[row] * in_deg_inv_sqrt[col]

    return normalized_edge_weights

def Qin_get_all_directed_adj(args,  edge_index, num_nodes, k, IsExhaustive, mode, norm='dir'):
    has_1_order = args.has_1_order
    selfloop = args.First_self_loop
    rm_gen_sloop = args.rm_gen_sloop

    device = edge_index.device
    if selfloop == 'add':
        edge_index, _ = add_self_loops(edge_index.long(), fill_value=1, num_nodes=num_nodes)       #
    elif selfloop == 'remove':
        edge_index, _ = remove_self_loops(edge_index)
    else:
        pass
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    L_tuple = sparse_boolean_multi_hop_DirGNN(has_1_order, rm_gen_sloop, A, k - 1)  # much slower

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        row, col = L._indices()
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        all_hops_weight.append(adj_norm.storage.value())

    return tuple(all_hop_edge_index), tuple(all_hops_weight)

def Qin_get_second_adj(edge_index, num_nodes, dtype, k):     #
    device = edge_index.device
    fill_value = 1
    # edge_index, _ = add_self_loops(edge_index.long(), fill_value=fill_value, num_nodes=num_nodes)       # TODO add back after no-selfloop test
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    L_tuple = OneDirect_sparse_boolean_multi_hop(A, k-1)   # much slower

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        edge_indexL = L._indices()
        all_hop_edge_index.append(edge_indexL)
        edge_weightL = normalize_row_edges(edge_indexL, num_nodes).to(device)
        all_hops_weight.append(edge_weightL)

    return tuple(all_hop_edge_index), tuple(all_hops_weight)

def get_second_directed_adj_union(edge_index, num_nodes, dtype, k):
    '''
    Qin change to get union
    '''
    device = edge_index.device
    fill_value = 1
    # edge_index, _ = add_self_loops(edge_index.long(), fill_value=fill_value, num_nodes=num_nodes)     # TODO add back after no-selfloop test
    edge_index, _ = remove_self_loops(edge_index)

    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), dtype=torch.bool).to(device), size=(num_nodes, num_nodes))
    L_tuple = sparse_boolean_multi_hop(A, k-1, mode='union')

    all_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        edge_indexL = L._indices()
        edge_weightL = normalize_row_edges(edge_indexL, num_nodes)
        all_edge_index.append(edge_indexL)
        all_hops_weight.append(edge_weightL)

    return tuple(all_edge_index), tuple(all_hops_weight)

@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def to_undirected(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    return edge_index, edge_weight


def to_undirectedBen(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    # print(edge_index, edge_index.shape)

    edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1])]

    set_edges = set(edges)

    unique_edges = list(set_edges)

    num_list = [0] * len(edges)
    history = []
    count = 0
    for i in range(len(edges)):
        if edges[i] in history:
            num_list[i] += 1
            count += 1
            # print("Duplicate: ", edges[i], num_list[i], count)
        else:
            history.append(edges[i])

    edge_index0 = [i[0] for i in unique_edges]
    edge_index1 = [i[1] for i in unique_edges]
    edge_index = torch.tensor([edge_index0, edge_index1])
    # print(edge_index, edge_index.shape)

    return edge_index





def remove_dupEdge(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    return edge_index, edge_weight


def link_prediction_evaluation(out_val, out_test, y_val, y_test):
    r"""Evaluates link prediction results.

    Args:
        out_val: (torch.FloatTensor) Log probabilities of validation edge output, with 2 or 3 columns.
        out_test: (torch.FloatTensor) Log probabilities of test edge output, with 2 or 3 columns.
        y_val: (torch.LongTensor) Validation edge labels (with 2 or 3 possible values).
        y_test: (torch.LongTensor) Test edge labels (with 2 or 3 possible values).

    :rtype:
        result_array: (np.array) Array of evaluation results, with shape (2, 5).
    """
    out = torch.exp(out_val).detach().to('cpu').numpy()
    y_val = y_val.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis=1)
    val_acc_full = accuracy_score(pred_label, y_val)
    # two-class evaluation
    out = out[y_val < 2, :2]
    y_val = y_val[y_val < 2]

    prob = out[:, 0] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    val_auc = roc_auc_score(y_val, prob)
    pred_label = np.argmax(out, axis=1)
    val_acc = accuracy_score(pred_label, y_val)
    val_f1_macro = f1_score(pred_label, y_val, average='macro')
    val_f1_micro = f1_score(pred_label, y_val, average='micro')

    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis=1)
    test_acc_full = accuracy_score(pred_label, y_test)
    # two-class evaluation
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]

    prob = out[:, 0] / (out[:, 0] + out[:, 1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis=1)
    test_acc = accuracy_score(pred_label, y_test)
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    return [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
            [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro]]

def organize4edgePred(new_x, edges, sampling_src_idx,neighbor_dist_list):
    '''

    Args:
        new_x:
        edges:
        sampling_src_idx:
        neighbor_dist_list:

    Returns:

    '''


    return data