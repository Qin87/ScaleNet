import os
from datetime import datetime

import numpy as np
import torch
from torch_scatter import scatter_add

from nets.gat import StandGATXBN
from nets.gcn import StandGCNXBN
from nets.models import create_MLP

from edge_nets.edge_data import to_undirectedBen
from gens import test_directed
from nets import create_sage

from data.data_utils import random_planetoid_splits, load_directedData
from nets.DiG_NoConv import (create_Di_IB_nhid,  Di_IB_XBN_nhid_ConV,
                             DiSAGE_x_nhid, DiSAGE_xBN_nhid)
import torch.nn.init as init



def init_model(model):
    # Initialize weights and biases of all parameters
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                init.xavier_uniform_(param)  # Initialize weights using Xavier initialization
            else:
                init.constant_(param, 0)  # Initialize biases to zero
        elif 'bias' in name:
            init.constant_(param, 0)  # Initialize biases to zero for bias parameters
    # Initialize parameters of batch normalization layers
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_parameters()  # Res

def CreatModel(args, num_features, n_cls, data_x,device):
    if args.net.lower() == 'mlp':
        model = create_MLP(nfeat=num_features, nhid=args.hid_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer)
    elif args.net.startswith(('Di', 'Ui', 'Ri', 'Ai')):        # GCN  -->  SAGE
        if len(args.net) < 4 :
            if args.BN_model:
                model = DiSAGE_xBN_nhid(args.net, num_features, n_cls, args).to(device)
            else:
                model = DiSAGE_x_nhid(args.net, num_features, n_cls, args).to(device)     # July 24  keep it: the original DiG paper
        else:
            if args.net.startswith('Di'):
                model = create_Di_IB_nhid(m=args.net, nfeat=num_features, nclass=n_cls, args=args).to(device)    # keep this: original DiGib paper
            else:
                model = Di_IB_XBN_nhid_ConV(m=args.net, input_dim=num_features, out_dim=n_cls, args=args).to(device)     # July 24: 1 BN

    else:
        if args.net == 'GCN':
            model = StandGCNXBN(num_features, n_cls, args=args)
        elif args.net in ['GAT', 'RAT', 'UAT']:
            if args.net=='GAT' and args.originGAT:
                model = StandGATXBN(nfeat=num_features, nhid=args.hid_dim, nclass=n_cls, dropout=args.dropout, args=args)
            else:
                model = StandGATXBN(nfeat=num_features, nhid=args.hid_dim, nclass=n_cls, dropout=args.dropout, args=args)
        elif args.net == "SAGE":
            model = create_sage(nfeat=num_features, nhid=args.hid_dim, nclass=n_cls, dropout=args.dropout,nlayer=args.layer)
        else:
            raise NotImplementedError("Not Implemented Architecture!"+ args.net)
    model = model.to(device)
    init_model(model)
    return model

def get_name(args, IsDirectedGraph=1):
    dataset_to_print = args.Dataset.replace('/', '_')
    if not IsDirectedGraph:
        dataset_to_print = dataset_to_print + 'Undire'
    else:
        dataset_to_print = dataset_to_print + 'Direct'
    if args.net.startswith('Ri'):
        net_to_print = args.net + str(args.W_degree) + '_'
    else:
        net_to_print = args.net
    if args.net[:2] == 'Ai' or args.net in ['GAT', 'RAT', 'UAT']:
        net_to_print = net_to_print + '_Head' + str(args.heads)
    if args.BN_model:
        net_to_print = 'LNorm_' + net_to_print
    else:
        net_to_print = 'NoLNorm_' + net_to_print
    if args.net[1] == 'i':
        if args.First_self_loop == 'add':
            net_to_print = net_to_print + '_AddSloop'
        elif args.First_self_loop == 'remove':
            net_to_print = net_to_print + '_RmSloop'
        else:
            net_to_print = net_to_print + '_NoSloop'

    if args.hid_dim != 64:
        net_to_print = net_to_print + '_hid' + str(args.hid_dim)
    if args.net == 'GAT':
        net_to_print += '_ofc' + str(args.originGAT)
    if args.r20_per_class:
        dataset_to_print += '20_perclass'

    return net_to_print, dataset_to_print


def log_file(net_to_print, dataset_to_print, args):
    log_file_name = dataset_to_print+'_'+net_to_print+'_lay'+str(args.layer)+'_lr'+str(args.lr)+'_NoImp'+str(args.NotImproved)+'_norm'+args.inci_norm
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.nonlinear:
        log_file_name_with_timestamp = f"{log_file_name}_{timestamp}_c.log"
    else:
        log_file_name_with_timestamp = f"{log_file_name}_{timestamp}_d.log"
    # d is no relu, e is hid relu for *iGib

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp

import os.path as osp
def load_dataset(args):
    if len(args.Dataset.split('/')) < 2:
        path = args.data_path
        path = osp.join(path, args.Dataset)
        dataset = get_dataset(args.Dataset, path, split_type='full')
        IsDirectedGraph = 0
    else:
        dataset = load_directedData(args)
        IsDirectedGraph = 1
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.Dataset in ['ogbn-arxiv/', 'directed-roman-empire/']:
        data = dataset._data
    else:
        data = dataset[0]

    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        edges_weight = torch.FloatTensor(data.edge_weight)
    except:
        edges_weight = None

    # copy GraphSHA
    if args.Dataset.split('/')[0].startswith('dgl'):
        edge_types = data.etypes
        print("Available edge types:", edge_types)
        num_edge_types = len(data.etypes)

        if num_edge_types == 1:
            # Only one edge type, retrieve edges normally
            edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        else:
            # Multiple edge types
            print("Edge types:", data.etypes)
            all_src = []
            all_dst = []

            for etype in data.etypes:
                src, dst = data.edges(etype=etype)
                all_src.append(src)
                all_dst.append(dst)

            # Concatenate all source and destination nodes
            all_src = torch.cat(all_src)
            all_dst = torch.cat(all_dst)

            # Combine source and destination to form edges
            edges = torch.stack([all_src, all_dst])
        data_y = data.ndata['label']
        print(data.ndata.keys())
        try:
            data_x = data.ndata['feat']
        except:
            data_x = data.ndata['feature']
        if args.Dataset.split('/')[1].startswith(('reddit', 'yelp')):
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())

        elif args.Dataset.split('/')[1].startswith(('Fyelp', 'Famazon')):
            data = random_planetoid_splits(data, data_y, train_ratio=0.7, val_ratio=0.1, Flag=0)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
        else:
            data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        # for multi-label dataset
        if args.Dataset.split('/')[1].startswith('yelp'):
            data_y = data_y[:, 1]
        dataset_num_features = data_x.shape[1]


    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y

        if args.r20_per_class:
            data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
        else:

            if data_y.dtype.is_floating_point:
                data_y = data_y.to(torch.long)
            if not hasattr(data, 'train_mask'):
                data = random_planetoid_splits(data, data_y, train_ratio=0.48, val_ratio=0.1, num_splits=10, Flag=0)
                data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
            elif args.Dataset in ['ogbn-arxiv/', 'arxiv_year', 'fb100/penn94', 'telegram/'] or (len(data.train_mask.shape) > 1 and data.train_mask.size(-1) > 1):
                data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
            else:
                data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
                data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]


    print("data_x", data_x.shape)  # [11701, 300])

    if args.to_undirected:
        if IsDirectedGraph == 0:
            print("Already undirected graph")
        else:
            IsDirectedGraph = test_directed(edges)  # time consuming
            print("This is directed graph: ", IsDirectedGraph)
        if IsDirectedGraph:
            edges = to_undirectedBen(edges)
            IsDirectedGraph = 0
            print("Converted to undirected data")

    return data_x, data_y, edges, edges_weight, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph


def count_homophilic_nodes(edge_index, y):
    num_nodes = y.size(0)
    in_homophilic_count = 0
    out_homophilic_count = 0
    no_in_neighbors = 0
    no_out_neighbors = 0

    for node in range(num_nodes):
        # Find the in-neighbors (nodes that point to the current node)
        in_neighbors = (edge_index[1] == node).nonzero(as_tuple=True)[0]
        in_neighbors = edge_index[0, in_neighbors]

        # Find the out-neighbors (nodes that the current node points to)
        out_neighbors = (edge_index[0] == node).nonzero(as_tuple=True)[0]
        out_neighbors = edge_index[1, out_neighbors]


        # Check in-neighbor homophily
        if len(in_neighbors) > 0:
            in_neighbor_labels = y[in_neighbors]
            in_most_common_label = torch.mode(in_neighbor_labels).values.item()
            if in_most_common_label == y[node]:
                in_homophilic_count += 1
        else:
            no_in_neighbors += 1

            # Check out-neighbor homophily
        if len(out_neighbors) > 0:
            out_neighbor_labels = y[out_neighbors]
            out_most_common_label = torch.mode(out_neighbor_labels).values.item()
            if out_most_common_label == y[node]:
                out_homophilic_count += 1
        else:
            no_out_neighbors += 1

    percent_no_in = (no_in_neighbors / num_nodes) * 100
    percent_in_homo = (in_homophilic_count / num_nodes) * 100
    percent_no_out = (no_out_neighbors / num_nodes) * 100
    percent_out_homo = (out_homophilic_count / num_nodes) * 100

    print('percent of no_in, in_homo, no_out, out_homo', end=':')
    print(f"{percent_no_in:.1f} & {percent_in_homo:.1f} & {percent_no_out:.1f} & {percent_out_homo:.1f}")
    # print(f"{percent_no_in:.1f}% & {percent_in_homo:.1f}% & {percent_no_out:.1f}% & {percent_out_homo:.1f}%")

    return no_in_neighbors, in_homophilic_count, no_out_neighbors, out_homophilic_count


def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Coauthor

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    elif name == 'Coauthor-CS':

        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    elif name == 'Coauthor-physics':

        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
    # elif name == 'ppi':     #
    #     path = '../data/ppi_data'
    #     G = json_graph.node_link_graph(json.load(open(path + "/toy-ppi-G.json")))
    #     labels = json.load(open(path + "/toy-ppi-class_map.json"))
    #     labels = {int(i): l for i, l in labels.iteritems()}
    #
    #     train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    #     test_ids = [n for n in G.nodes() if G.node[n][setting]]
    #     train_labels = np.array([labels[i] for i in train_ids])
    #     if train_labels.ndim == 1:
    #         train_labels = np.expand_dims(train_labels, 1)
    #     test_labels = np.array([labels[i] for i in test_ids])
    #
    #     embeds = np.load(data_dir + "/val.npy")
    #     id_map = {}
    #     with open(data_dir + "/val.txt") as fp:
    #         for i, line in enumerate(fp):
    #             id_map[int(line.strip())] = i
    #     train_embeds = embeds[[id_map[id] for id in train_ids]]
    #     test_embeds = embeds[[id_map[id] for id in test_ids]]
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset

def remove_inner_class_edge(edges, y):
    src, dst = edges
    mask = y[src] != y[dst]
    new_edges = edges[:, mask]
    return new_edges