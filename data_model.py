import os
from datetime import datetime

import numpy as np
import torch
from torch_scatter import scatter_add

from nets.gat import GATConvQin, StandGAT1BN_Qin
from nets.gcn import ParaGCNXBN, StandGCNXBN, StandGCNXBN_Ak, StandGCNX_noRelu
from nets.geometric_baselines import GCN_JKNet, GPRGNN, get_model, Sloop_JKNet, ScaleNet, RandomNet, High_Frequent
from nets.models import JKNet, create_MLP, create_SGC, create_pgnn, GPRGNNNet1, GraphModel

from nets.Signum_quaternion import QuaNet_node_prediction_one_laplacian_Qin
from nets.Signum import SigMaNet_node_prediction_one_laplacian_Qin
from edge_nets.edge_data import to_undirectedBen
from gens import test_directed
from nets import  create_gat

from data.data_utils import random_planetoid_splits, load_directedData
from nets.APPNP_Ben import APPNP_Model, ChebModel, SymModel, GCNModel_Cheb
# from nets.DGCN import SymModel
# from nets.DiGCN import DiModel, DiGCN_IB
from nets.DiG_NoConv import (create_DiG_MixIB_SymCat_Sym_nhid,
                             create_DiG_MixIB_SymCat_nhid, create_DiG_IB_SymCat_nhid, create_DiG_IB_Sym_nhid, create_DiG_IB_Sym_nhid_para,
                             create_DiG_IB_nhid_para, create_DiSAGESimple_nhid, Si_IB_X_nhid, DiGCN_IB_XBN_nhid_para, Di_IB_XBN_nhid_ConV, DiSAGE_xBN_nhid_BN,
                             create_DiSAGESimple_nhid0, DiSAGE_x_nhid, DiSAGE_xBN_nhid, DiGCN_IB_X_nhid_para, Si_IB_X_nhid, DiSAGE_1BN_nhid,
                             DiSAGE_2BN_nhid, DiGCN_IB_X_nhid_para_Jk, Di_IB_XBN_nhid_ConV_JK, Di_IB_X_nhid, Di_IB_1_nhid)
# from nets.DiG_NoConv import  create_DiG_IB
from nets.GIN_Ben import create_GIN
from nets.Sym_Reg import create_SymReg_add, create_SymReg_para_add
from nets.sagcn import SAGCN, SAGCNXBN
from nets.gcn import GraphSAGEXBatNorm
# from nets.UGCL import UGCL_Model_Qin
from nets.sparse_magnet import ChebNet_Ben, ChebNet_BenQin, ChebNet_Ben_05
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

def CreatModel(args, num_features, n_cls, data_x,device, num_edges=None):
    if args.net.lower() == 'pgnn':
        model = create_pgnn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout, layer =args.layer)
    elif args.net.lower() == 'mamba':
        model = GraphModel(channels=64, pe_dim=8, num_layers=10,
                           model_type='mamba',
                           shuffle_ind=1, order_by_degree=False,
                           d_conv=4, d_state=16,
                           ).to(device)
    elif args.net.lower() == 'mlp':
        model = create_MLP(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer)
    elif args.net.lower() == 'sgc':
        model = create_SGC(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer,K=args.K)
    elif args.net == 'Dir-GNN':
        model = get_model(num_features,  n_cls, args)
    elif args.net.lower() == 'jk':
        model = JKNet(in_channels=num_features,
                        out_channels=n_cls,
                        num_hid=args.feat_dim,
                        K=args.K,
                        alpha=args.alpha,
                        dropout=args.dropout, layer=args.layer)
    elif args.net.lower() == 'gprgnn':
        # model = GPRGNNNet1_Qin(in_channels=num_features,      # a bit worse
        model = GPRGNNNet1(in_channels=num_features,
                            out_channels=n_cls,
                            num_hid=args.feat_dim,
                            ppnp=args.ppnp,
                            K=args.K,
                            alpha=args.alpha,
                            Init=args.Init,
                            Gamma=args.Gamma,
                            dprate=args.dprate,
                            dropout=args.dropout)

    elif args.net == 'GIN':
        model = create_GIN(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
    elif args.net == 'Cheb':
        model = ChebModel(num_features, n_cls, K=args.K,filter_num=args.feat_dim, dropout=args.dropout,layer=args.layer).to(device)
    elif args.net == 'ScaleNet':
        model = GCN_JKNet(nfeat=num_features, nclass=n_cls, args=args)
    elif args.net == 'HFNet':
        model = High_Frequent(nfeat=num_features, nclass=n_cls, args=args)
    elif args.net == 'RandomNet':
        model = RandomNet(nfeat=num_features, nclass=n_cls, args=args)
    elif args.net == 'tSNE':
        model = ScaleNet(nfeat=num_features, nclass=n_cls, args=args)
    elif args.net == 'SloopNet':
        model = Sloop_JKNet(nfeat=num_features, nclass=n_cls, args=args)
    elif args.net == 'GPRGNN':
        model = GPRGNN(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, args= args)
    elif args.net == 'APPNP':
        model = APPNP_Model(num_features, n_cls,args.feat_dim, alpha=args.alpha,dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith(('Di', '1i', 'Ri', 'Ui', 'Li', 'Ai', 'Ti', 'Hi','Ii', 'ii')):        # GCN  -->  SAGE
        if len(args.net) < 4 or args.net.startswith(('Ui', 'Li')):
            if args.BN_model:
                model = DiSAGE_xBN_nhid(args.net[2], num_features, n_cls, args).to(device)
            else:
                model = DiSAGE_x_nhid(args.net[2], num_features, n_cls, args).to(device)     # July 24  keep it: the original DiG paper
        else:
            if args.net[3:].startswith(('Sym', '1ym')):
                if args.net[6:].startswith('Cat'):
                    if args.net[9:].startswith('Mix'):
                        if args.net[12:].startswith(('Sym', '1ym')):
                            model = create_DiG_MixIB_SymCat_Sym_nhid(args.net[2], num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                        else:
                            model = create_DiG_MixIB_SymCat_nhid(args.net[2],num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                    else:
                        model = create_DiG_IB_SymCat_nhid(args.net[2], num_features, args.feat_dim, n_cls, args.dropout, args.layer, args.ibx1).to(device)
                else:
                    if args.paraD:
                        model = create_DiG_IB_Sym_nhid_para(args.net[2], num_features,  n_cls, args).to(device)
                    else:
                        model = create_DiG_IB_Sym_nhid(args.net[2], num_features,  n_cls, args).to(device)
            else:
                if args.net.startswith(('Ai', 'Ti')):
                    if args.paraD:
                        model = DiGCN_IB_X_nhid_para_Jk(args.net[2], num_features,  n_cls,  args).to(device)
                    else:
                        model = Di_IB_XBN_nhid_ConV_JK(m=args.net[2], input_dim=num_features, out_dim=n_cls, args=args).to(device)
                else:
                    if args.paraD:
                        model = DiGCN_IB_XBN_nhid_para(args.net[2], num_features,  n_cls,  args).to(device)        # July 25
                    else:
                        if args.net.startswith('Di') or args.net.endswith('ib'):
                            model = Di_IB_X_nhid(args.net[2], num_features, n_cls, args).to(device)    # keep this: original DiGib paper
                        else:
                            model = Di_IB_XBN_nhid_ConV(m=args.net[2], input_dim=num_features, out_dim=n_cls, args=args).to(device)     # July 24: 1 BN
    elif args.net.startswith(('Sym', '1ym')):
        model = SymModel(num_features, n_cls, filter_num=args.feat_dim,dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith(('addSym', 'addQym')):
        if not args.net.endswith('para'):
            model = create_SymReg_add(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        else:
            model = create_SymReg_para_add(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
    elif args.net.startswith('Mag'):
        if args.net[3:].startswith('Qin'):
            model = ChebNet_BenQin(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                                activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
        if args.net[3:].startswith('0_5'):      # 0.5
            model = ChebNet_Ben_05(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                                activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
        else:
            model = ChebNet_Ben(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                            activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
    elif args.net.startswith('Sig'):
        model = SigMaNet_node_prediction_one_laplacian_Qin(num_features, K=args.K, hidden=args.feat_dim, label_dim=n_cls,i_complex=args.i_complex, layer=args.layer,
                                                           activation=args.activation,follow_math=args.follow_math, gcn=args.gcn, net_flow=args.netflow, unwind=True).to(device)
    elif args.net.startswith('Qua'):
        model = QuaNet_node_prediction_one_laplacian_Qin(device, num_features, K=args.K, hidden=args.feat_dim, label_dim=n_cls,
                                                     layer=args.layer, unwind=True,
                                                     quaternion_weights=args.qua_weights, quaternion_bias=args.qua_bias).to(device)

    else:
        if args.net == 'GCN':
            # model = GCNModel_Cheb(num_features, n_cls,filter_num=args.feat_dim, dropout=args.dropout, layer=args.layer).to(device)
            model = StandGCNXBN(num_features, n_cls, args=args)
        elif args.net == 'GCN_NoRelu':
            # model = GCNModel_Cheb(num_features, n_cls,filter_num=args.feat_dim, dropout=args.dropout, layer=args.layer).to(device)
            model = StandGCNX_noRelu(num_features, n_cls, args=args)
        elif args.net == 'GCNAk':  # only 1 layer GCN, other layer are MLP
            model = StandGCNXBN_Ak(num_features, n_cls, args=args)
        elif args.net == 'ParaGCN':
            model = ParaGCNXBN(num_node=data_x.shape[0] ,num_edges=num_edges, nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, norm= args.gcn_norm)
        elif args.net == 'GAT':
            model = create_gat(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, head=args.heads)
        elif args.net == 'SimGAT':
            model = StandGAT1BN_Qin(data_x.shape[0], num_features, args.feat_dim, n_cls, args.dropout, args.layer, head=args.heads)
        elif args.net == "SAGE":
            model = GraphSAGEXBatNorm(nfeat=num_features,  nclass=n_cls, args=args)
            # model = create_sage(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout,nlayer=args.layer)
        elif args.net == "SAGCN":
            model = SAGCNXBN(num_features, args.feat_dim, n_cls, args.dropout, args.layer, norm=True)
            # model = GraphSAGEXBatNorm(nfeat=num_features, nclass=n_cls, args=args)
            # model = SAGCN(num_features,  n_cls,  cached= False, normalize=True, add_self_loops=True)
            # model = SAGCN(nfeat=num_features,  nclass=n_cls, args=args)
        else:
            raise NotImplementedError("Not Implemented Architecture!"+ args.net)
    model = model.to(device)
    # if not args.conv_type == 'dir-sage':
    init_model(model)
    return model

def get_name(args, IsDirectedGraph):
    dataset_to_print = args.Dataset.replace('/', '_')
    if args.degfea:
        dataset_to_print = 'degfea' + str(args.degfea)+ dataset_to_print
    else:
        if args.all1:
            dataset_to_print = 'all1' +str(args.all1d)+ dataset_to_print
    if not IsDirectedGraph:
        dataset_to_print = dataset_to_print + 'Undire'
    else:
        dataset_to_print = dataset_to_print + 'Direct'
    if args.net.startswith('Ri'):
        net_to_print = args.net + str(args.W_degree) + '_'
    elif args.net.startswith('Mag'):
        net_to_print = args.net + str(args.q)
    else:
        net_to_print = args.net
    if args.net[1:3] == 'iA' or args.net == 'GAT':
        net_to_print = net_to_print + '_Head' + str(args.heads)
    if args.BN_model:
        net_to_print = 'BNorm_' + net_to_print
    else:
        net_to_print = 'NoBNorm_' + net_to_print



    if args.net == 'GCN':
        if args.First_self_loop == 1 or args.add_selfloop == 1:
            net_to_print = net_to_print + '_AddSloop'
        elif args.add_selfloop == -1:
            net_to_print = net_to_print + '_RmSloop'
        else:
            net_to_print = net_to_print + '_NoAddSloop'
        if args.gcn_norm == 1:
            net_to_print = net_to_print + '_gcnnorm'
        else:
            net_to_print = net_to_print + '_Nogcnnorm'
    else:
        if args.add_selfloop:
            net_to_print = net_to_print + '_AddSloop'

    if args.net[1] == 'i':
        if args.paraD:
            net_to_print = net_to_print + 'paraD' + str(args.coeflr)

        if args.First_self_loop == 1:
            net_to_print = net_to_print + '_AddSloop'
        elif args.First_self_loop == -1:
            net_to_print = net_to_print + '_RmSloop'
        else:
            net_to_print = net_to_print + '_NoSloop'

        # if args.feat_proximity:
        #     net_to_print = net_to_print + '_feaProx'
    # if args.feat_dim != 64:
    net_to_print = net_to_print + str(args.feat_dim) + 'hid_'
    if args.MakeImbalance:
        net_to_print = net_to_print + '_Imbal' + str(args.imb_ratio)
    else:
        net_to_print = net_to_print + '_Bal'
    if args.net == 'ScaleNet':
        if args.differ_AA or args.differ_AAt:
            if args.differ_AA:
                diff = 'AA'+str(args.alphaDir)
            else:
                diff = 'AAt'+str(args.alphaDir)
            net_to_print = net_to_print + args.conv_type + '_diff'+diff + '_jk'+str(args.jk)+'_norm'+args.inci_norm
        else:
            net_to_print = net_to_print  +'_' + args.conv_type +'_part'+str(args.alphaDir)+'_'+ str(args.betaDir)+'_'+str(
                args.gamaDir)+'_sloop'+str(args.First_self_loop)+str(args.rm_gen_sloop)+'_jk'+str(args.jk)+'_norm'+str(args.inci_norm)

    return net_to_print, dataset_to_print


def log_file(net_to_print, dataset_to_print, args):
    if args.Ak:
        net_to_print = net_to_print + 'Ak' + str(args.Ak)
    log_file_name = dataset_to_print+'_'+net_to_print+'_lay'+str(args.layer)+'_lr'+str(args.lr)+'_NoImp'+str(args.NotImproved)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name_with_timestamp = f"{log_file_name}_{timestamp}.log"

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp

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
    elif name == 'ppi':     # TODO
        dataset_dir = './data/ppi_data'
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
        labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
        labels = {int(i): l for i, l in labels.iteritems()}

        train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
        test_ids = [n for n in G.nodes() if G.node[n][setting]]
        train_labels = np.array([labels[i] for i in train_ids])
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)
        test_labels = np.array([labels[i] for i in test_ids])

        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]]
        test_embeds = embeds[[id_map[id] for id in test_ids]]
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset

import os.path as osp
def load_dataset(args):
    if len(args.Dataset.split('/')) < 2:
        path = args.data_path
        path = osp.join(path, args.Dataset)
        dataset = get_dataset(args.Dataset, path, split_type='full')
    else:
        dataset = load_directedData(args)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.Dataset in ['ogbn-arxiv/', 'directed-roman-empire/', 'snap-patents/', 'arxiv-year/']:
        data = getattr(dataset, '_data', dataset.data)
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
        # if isinstance(data_y[0], float):
        #     data_y = int(data_y)
        if data_y.dtype.is_floating_point:
            data_y = data_y.to(torch.long)
            # data_y = data_y.long()
        if not hasattr(data, 'train_mask'):
            data = random_planetoid_splits(data, data_y, train_ratio=0.48, val_ratio=0.1, num_splits=10, Flag=0)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        elif args.Dataset in ['ogbn-arxiv/'] or (len(data.train_mask.shape) > 1 and data.train_mask.size(-1) > 9):
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())
        else:
            data = random_planetoid_splits(data, data_y, percls_trn=20, val_lb=30, Flag=1)
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone())

        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    IsDirectedGraph = test_directed(edges)        # time consuming
    print("This is directed graph: ", IsDirectedGraph)
    # print("data_x", data_x.shape)  # [11701, 300])

    if IsDirectedGraph and args.to_undirected:
        edges = to_undirectedBen(edges)
        IsDirectedGraph = False
        print("Converted to undirected data")
    try:
        edge_attr = data.edge_attr
        data_batch = data.batch
    except:
        edge_attr = None
        data_batch = None

    return data_x, data_y, edges, edges_weight, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph, edge_attr, data_batch

def feat_proximity(edge_index1, data_x):
    distances = []
    for i in range(edge_index1.size(1)):
        a = edge_index1[0, i]
        b = edge_index1[1, i]
    # for edge in edges:
    #     a, b = edge
        node_a_features = data_x[a]
        node_b_features = data_x[b]
        distance = torch.sum(torch.abs(node_a_features - node_b_features)).item()
        distances.append(distance)

    # Calculate the average distance
    average_distance = sum(distances) / len(distances)

    sorted_distances = sorted(distances)

    # Calculate the threshold for the first 80%
    threshold_index = int(len(sorted_distances) * 0.8) - 1
    threshold_value = sorted_distances[threshold_index]
    return average_distance, threshold_value


def delete_edges(edge_index1, data_x, threshold_value):
    filtered_edges = []
    for i in range(edge_index1.size(1)):
        a = edge_index1[0, i]
        b = edge_index1[1, i]
        node_a_features = data_x[a]
        node_b_features = data_x[b]
        distance = torch.sum(torch.abs(node_a_features - node_b_features)).item()
        if distance < threshold_value:
            filtered_edges.append([a, b])

    # Convert filtered edges back to a tensor
    filtered_edge_index1 = torch.tensor(filtered_edges).t()
    return filtered_edge_index1

def make_imbalanced(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    training split don't influence edge, but make_imbalance will cut edge.
    :param edge_index: all edges in the graph
    :param label: classes of all nodes
    :param n_data:num of train in each class
    :param n_cls:
    :param ratio:
    :param train_mask:
    :return: list(class_num_list), train_mask, idx_info, node_mask, edge_mask
    """
    # Sort from major to minor
    device = edge_index.device
    n_data = torch.tensor(n_data)   # from list to tensor
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    ratio = torch.tensor(ratio, dtype=torch.float32)   # for mu to convert to numpy
    # Move the tensor to CPU before using it in numpy operations
    if not isinstance(n_cls, int):
        ratio = ratio.cpu()
        n_cls = n_cls.cpu()
    mu = np.power(1/ratio.detach().cpu().numpy(), 1/(n_cls - 1))            # mu is ratio of two classes, while args.ratio is ratio of major to minor
    mu = torch.tensor(mu, dtype=torch.float32, device=ratio.device)

    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        temp = int(sorted_n_data[0].item() * np.power(mu, i))
        if temp< 1:
            temp = 1
        class_num_list.append(int(min(temp, sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1:  # We do not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)   # from list to np.array
    class_num_list = class_num_list[inv_indices]    # sorted
    n_round = np.array(n_round)[inv_indices]        # sorted  #

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    # print(remove_idx_list)  # [[], [], [], [], [], [], []]
    cls_idx_list = []   # nodes belong to class i
    index_list = torch.arange(len(train_mask)).to(device)
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool).to(device)
            # new_ones is a PyTorch function used to create a new tensor of ones with the specified shape and data type.
            # print("Initialize all true: ", node_mask[:10])
            node_mask[sum(remove_idx_list, [])] = False
            # print("Setting some as false", node_mask[:10])

            # Remove connection with removed nodes
            row, col = edge_index[0].to(device), edge_index[1].to(device)
            # print("row is ", row.shape, row[:10])
            # # torch.Size([10556]) tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
            # print("col is ", row.shape, col[:10])
            # # torch.Size([10556]) tensor([ 633, 1862, 2582,    2,  652,  654,    1,  332, 1454, 1666])
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask  # elementwise "and"

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]

            # remove_idx_list[i] = list(remove_idx.numpy())
            remove_idx_list[i] = list(remove_idx.cpu().numpy())     # Ben for GPU

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask     #

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def count_homophilic_nodes(edge_index, y):
    in_homophilic_nodes = []
    out_homophilic_nodes = []
    no_in_nodes = []
    no_out_nodes = []
    in_heterophilic_nodes = []
    out_heterophilic_nodes= []

    num_nodes = y.size(0)
    in_homophilic_count = 0
    out_homophilic_count = 0
    no_in_neighbors = 0
    no_out_neighbors = 0

    for node in range(num_nodes):
        y_node = y[node].item()
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
            # if node == 14:
            #     print(in_most_common_label, in_neighbor_labels)
            if in_most_common_label == y_node:
                in_homophilic_count += 1
                in_homophilic_nodes.append(node)
            else:
                in_heterophilic_nodes.append(node)
        else:
            no_in_neighbors += 1
            no_in_nodes.append(node)

        # Check out-neighbor homophily
        if len(out_neighbors) > 0:
            out_neighbor_labels = y[out_neighbors]
            out_most_common_label = torch.mode(out_neighbor_labels).values.item()
            # if node == 2144:
            #     print(out_most_common_label, out_neighbor_labels, y_node)
            if out_most_common_label == y_node:
                out_homophilic_count += 1
                out_homophilic_nodes.append(node)
            else:
                out_heterophilic_nodes.append(node)
        else:
            no_out_neighbors += 1
            no_out_nodes.append(node)

        # if len(in_neighbors) == 0 and len(out_neighbors) == 0:
        #     isolated_nodes.append(node)

    percent_no_in = (no_in_neighbors / num_nodes) * 100
    percent_in_homo = (in_homophilic_count / num_nodes) * 100
    percent_no_out = (no_out_neighbors / num_nodes) * 100
    percent_out_homo = (out_homophilic_count / num_nodes) * 100

    print('percent of no_in, in_homo, no_out, out_homo', end=':')
    print(f"{percent_no_in:.1f}% | {percent_in_homo:.1f}% | {percent_no_out:.1f}% | {percent_out_homo:.1f}%")
    # print(f"{percent_no_in:.1f}% & {percent_in_homo:.1f}% & {percent_no_out:.1f}% & {percent_out_homo:.1f}%")

    return no_in_neighbors, in_homophilic_count, no_out_neighbors, out_homophilic_count, in_homophilic_nodes, out_homophilic_nodes,in_heterophilic_nodes, out_heterophilic_nodes, no_in_nodes,  no_out_nodes


from sklearn.metrics import balanced_accuracy_score, f1_score

def calculate_metrics(logits, data_test_mask, data_y, node_index_lists, edge_index):
    if len(node_index_lists) == 0:
        return 0
    device = data_test_mask.device
    results = []
    node_index_mask = create_mask(node_index_lists, data_y.shape[0]).to(device)
    mask = node_index_mask & data_test_mask

    # test_indices = [i for i in node_index_lists if data_test_mask[i]]
    test_indices = [i for i in node_index_lists if mask[i]]
    test_indices.sort()
    if len(test_indices) == 0:
        return 0
    percentage = round((len(test_indices))/data_test_mask.sum().item() * 100, 2)

    pred_origin = logits.max(1)[1]
    pred = logits[mask].max(1)[1]
    y_pred = pred.cpu().numpy()
    y_true = data_y[mask].cpu().numpy()

    # Calculate metrics
    # acc = round(pred.eq(data_y[test_indices]).sum().item() / len(test_indices), 2)
    acc = round(pred.eq(data_y[mask]).sum().item() / len(test_indices), 2)
    bacc = round(balanced_accuracy_score(y_true, y_pred), 2)
    f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
    correct = pred.eq(data_y[mask]).sum().item()

    # masked_data_y = data_y[mask]

    wrong_indices = [(test_indices[i], y_pred[i], y_true[i]) for i in range(len(pred)) if y_pred[i] != y_true[i]]
    # wrong_i, wrong_indices = zip(*[(i, index) for i, index in enumerate(test_indices) if pred[i] != masked_data_y[i]])
    # wrong_indices = [index for i, index in enumerate(test_indices) if pred[i] != data_y[mask][i]]
    # wrong_here = (pred != data_y[mask]).nonzero()
    wrong_node_info = {}

    # for i, idx in enumerate(wrong_indices):
    # for i, idx in zip(wrong_i, wrong_indices):
        # print(data_y.shape)
        # print(mask.shape)
    i=0
    for idx, pred_val, true_val in wrong_indices:
        # if idx == 184 or idx == 1054:
        #     print(data_y[idx].item())
        if true_val.item() != data_y[idx].item():
            print('fault: node', idx, true_val.item(),  data_y[idx].item())
        wrong_node_info[i] = {
            'node': idx,
            # 'predicted_class': pred_origin[idx].item(),
            # 'predicted_class': pred[i].item(),
            'predicted_class': pred_val.item(),
            # 'true_class': data_y[idx].item(),
            # 'true_class': masked_data_y[i].item(),
            'true_class': true_val.item(),
            'neighbors': {
                'in': {
                    # 'nodes': edge_index[0][edge_index[1] == idx].tolist(),
                    'classes': data_y[edge_index[0][edge_index[1] == idx]].tolist()
                },
                'out': {
                    # 'nodes': edge_index[1][edge_index[0] == idx].tolist(),
                    'classes': data_y[edge_index[1][edge_index[0] == idx]].tolist()
                }
            }
        }
        i += 1

    results.append({
        'num_node': len(node_index_lists),
        'test': len(test_indices),
        'correct': correct,
        # 'percentage': percentage,

        'acc': acc,
        'bacc': bacc,
        'f1': f1,
        'wrong_node_info': wrong_node_info

    })

    return results


def create_mask(node_index_list, total_nodes):
    # Initialize a mask array of False values
    mask = torch.zeros(total_nodes, dtype=torch.bool)

    # Set the positions corresponding to node_index_list to True
    mask[node_index_list] = True

    return mask


def print_x(x):
    x_min = torch.min(x).item()
    x_max = torch.max(x).item()
    zero_count = torch.sum(x == 0).item()
    one_count = torch.sum(x == 1).item()

    print(f"Minimum value: {x_min}")
    print(f"Maximum value: {x_max}")
    print(f"Number of zeros: {zero_count}")
    print(f"Number of ones: {one_count}")

    non_zero_one = x[(x != 0) & (x != 1)]
    print("Values that are not 0 or 1:", end='')
    if non_zero_one.numel() > 0:
        formatted_values = [f"{v:.2f}" for v in non_zero_one.tolist()]
        if len(formatted_values) > 10:
            print(" ( total_num =", len(formatted_values),')\n', formatted_values[:10], "... (showing first 10)")
        else:
            print(formatted_values)
        print(f"Total count of values not 0 or 1: {non_zero_one.numel()}")
    else:
        print("None")

    x = torch.where(x != 0, torch.tensor(1.0, device=x.device), x)
    print("\nall features are 0 or 1 now.")

    x_min = torch.min(x).item()
    x_max = torch.max(x).item()
    zero_count = torch.sum(x == 0).item()
    one_count = torch.sum(x == 1).item()

    print(f"Minimum value: {x_min}")
    print(f"Maximum value: {x_max}")
    print(f"Number of zeros: {zero_count}")
    print(f"Number of ones: {one_count}")

    return x
