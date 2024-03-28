import os
from datetime import datetime
import torch
import sys
sys.path.append('./Signum_quaternion/QuaNet_node_prediction_one_laplacian_Qin')
sys.path.append('./Signum_quaternion/')
print('sys path is',sys.path)

print(os.getcwd())
try:
    from Signum_quaternion import QuaNet_node_prediction_one_laplacian_Qin
except:
    pass
from Signum import SigMaNet_node_prediction_one_laplacian_Qin
from edge_nets.edge_data import to_undirectedBen
from gens import test_directed
from nets import create_gcn, create_gat, create_sage
import os.path as osp

from data_utils import load_directedData, get_dataset, get_step_split
from nets.APPNP_Ben import create_APPNPSimp
from nets.Cheb_Ben import create_Cheb
# from nets.DGCN import SymModel
# from nets.DiGCN import DiModel, DiGCN_IB
from nets.DiG_NoConv import create_DiGSimple, create_DiG_IB_SymCat, create_DiG_MixIB_SymCat, create_DiG_IB_batch, create_DiG_IB_Sym_batch, create_DiG_MixIB_SymCat_Sym
from nets.DiG_NoConv import  create_DiG_IB
from nets.DiG_NoConv import create_DiG_IB_Sym
from nets.GIN_Ben import create_GIN
from edge_nets.SD_GCN import SDGCN_Edge
from nets.Sym_Reg import create_SymReg, create_SymReg_add
from nets.UGCL import UGCL_Model_Qin
from nets.hermitian import hermitian_decomp_sparse, cheb_poly_sparse
from nets.sparse_magnet import ChebNet_Ben, ChebNet_BenQin


def CreatModel(args, num_features, n_cls, data_x,device):
    if args.net == 'GIN':
        model = create_GIN(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        # model = GIN_ModelBen(num_features, n_cls, nhid=args.feat_dim,
        #                      dropout=args.dropout, layer=args.layer)
    elif args.net == 'Cheb':
        model = create_Cheb(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, K=args.K).to(device)
        # model = ChebModelBen(num_features, n_cls, K=args.K,
        #                      filter_num=args.num_filter, dropout=args.dropout,
        #                      layer=args.layer).to(device)
    elif args.net == 'APPNP':
        model = create_APPNPSimp(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, alpha=args.alpha, K=10).to(device)
        # model = create_APPNPGGPT(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, alpha=args.alpha, K=10).to(device)
        # model = create_APPNP(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer, alpha=args.alpha).to(device)
        # model = create_appnp(num_features, n_cls,
        #                        filter_num=args.num_filter, alpha=args.alpha,
        #                        dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith('DiG'):
        if not args.net[-2:] == 'ib':
            model = create_DiGSimple(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        else:
            if args.net[3:].startswith('Sym'):
                if args.net[6:].startswith('Cat'):
                    if args.net[9:].startswith('Mix'):
                        if args.net[12:].startswith('Sym'):
                            model = create_DiG_MixIB_SymCat_Sym(num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                        else:
                            model = create_DiG_MixIB_SymCat(num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                    else:
                        model = create_DiG_IB_SymCat(num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                else:
                    if not args.largeData:
                        model = create_DiG_IB_Sym(num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                    else:
                        print('Shoot, using batch_size:', args.batch_size)
                        model = create_DiG_IB_Sym_batch(num_features, args.feat_dim, n_cls, args.dropout, args.layer, args.batch_size).to(device)
            else:
                if not args.largeData:
                    model = create_DiG_IB(num_features, args.feat_dim, n_cls, args.dropout, args.layer).to(device)
                else:
                    print('Shoot, using batch_size:', args.batch_size)
                    model = create_DiG_IB_batch(num_features, args.feat_dim, n_cls, args.dropout, args.layer, args.batch_size).to(device)

    elif args.net.startswith('Sym'):
        model = create_SymReg(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
        # model = SymModel(num_features, n_cls, filter_num=args.num_filter,dropout=args.dropout, layer=args.layer).to(device)
    elif args.net.startswith('addSym'):
        model = create_SymReg_add(num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer).to(device)
    elif args.net.startswith('Mag'):
        if args.net[3:].startswith('Qin'):
            model = ChebNet_BenQin(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                                activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
        else:
            model = ChebNet_Ben(num_features, K=args.K, label_dim=n_cls, layer=args.layer,
                            activation=args.activation, num_filter=args.feat_dim, dropout=args.dropout).to(device)
    elif args.net.startswith('Sig'):
        model = SigMaNet_node_prediction_one_laplacian_Qin(num_features, K=args.K, hidden=args.feat_dim, label_dim=n_cls,i_complex=args.i_complex, layer=args.layer,
                                                           activation=args.activation,follow_math=args.follow_math, gcn=args.gcn, net_flow=args.netflow, unwind=True).to(device)
    elif args.net.startswith('Qua'):
        # model = QuaNet_node_prediction_one_laplacian(num_features, K=args.K,hidden=args.num_filter, label_dim=n_cls,
        #                                              layer=args.layer, unwind=True, edge_index=edges, \
        #                                              norm_real=norm_real, norm_imag_i=norm_imag_i, norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
        #                                              quaternion_weights=args.qua_weights, quaternion_bias=args.qua_bias).to(device)
        model = QuaNet_node_prediction_one_laplacian_Qin(device, num_features, K=args.K, hidden=args.num_filter, label_dim=n_cls,
                                                     layer=args.layer, unwind=True,
                                                     quaternion_weights=args.qua_weights, quaternion_bias=args.qua_bias).to(device)

    elif args.net.startswith('SD'):
        # model = SDGCN_Edge(X_real.size(-1), L_real, L_img, K=args.K, label_dim=args.num_class_link,
        #            layer=args.layer, num_filter=args.num_filter, dropout=args.dropout)
        model = SDGCN_Edge(X_real.size(-1), L_real, L_img, K=args.K, label_dim=args.num_class_link,
                           layer=args.layer, num_filter=args.num_filter, dropout=args.dropout)
    elif args.net.startswith('UGCL'):
        model = UGCL_Model_Qin(num_hidden=args.feat_dim, num_proj_hidden=args.feat_dim, num_label=n_cls)
    else:
        if args.net == 'GCN':
            model = create_gcn(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer)
        elif args.net == 'GAT':
            model = create_gat(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout, nlayer=args.layer)
        elif args.net == "SAGE":
            model = create_sage(nfeat=num_features, nhid=args.feat_dim, nclass=n_cls, dropout=args.dropout,nlayer=args.layer)
        else:
            raise NotImplementedError("Not Implemented Architecture!")
    # try:
    #     print(model)  # # StandGCN2((conv1): GCNConv(3703, 64)  (conv2): GCNConv(64, 6))
    # except:
    #     pass
    model = model.to(device)
    return model


def load_dataset(args,device, laplacian=True, gcn_appr=False):
    if args.IsDirectedData:
        dataset = load_directedData(args)
    else:
        path = args.data_path
        path = osp.join(path, args.undirect_dataset)
        dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("The Dataset is ", dataset, "from DirectedData: ", args.IsDirectedData)

    # if os.path.isdir(log_path) is False:
    #     os.makedirs(log_path)

    data = dataset[0].to(device)
    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        data.edge_weight = torch.FloatTensor(data.edge_weight)
    except:
        data.edge_weight = None


    # copy GraphSHA
    if args.IsDirectedData and args.Direct_dataset.split('/')[0].startswith('dgl'):
        try:
            edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0).to(device)
        except:
            print(data.canonical_etypes)
            print(data.etypes)
            edges = torch.cat((data.edges(etype='rdftype')[0].unsqueeze(0), data.edges(etype='rdftype')[1].unsqueeze(0)), dim=0).to(device)

        data_y = data.ndata['label'].to(device)
        print(data.ndata.keys())

        try:
            data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
        except:
            data_train_maskOrigin = data.ndata['train_mask']
            data_val_maskOrigin = data.ndata['val_mask']
            data_test_maskOrigin = data.ndata['test_mask']
        data_x = data.ndata['feat']
        dataset_num_features = data_x.shape[1]
    # elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
    elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        edges = data.edge_index.to(device)  # for torch_geometric librar
        data_y = data.y.to(device)
        data_x = data.x.to(device)
        dataset_num_features = dataset.num_features

        data_y = data_y.long()
        n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
        n_cls = torch.tensor(n_cls).to(device)

        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio,
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls),
                                                                    labeling_ratio=0.1,
                                                                    all_idx=[i for i in range(data.x.shape[0])],
                                                                    all_label=data.y.cpu().detach().numpy(),
                                                                    nclass=n_cls)

        data_train_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_maskOrigin[train_idx] = True
        data_val_maskOrigin[valid_idx] = True
        data_test_maskOrigin[test_idx] = True
        train_idx = data_train_maskOrigin.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]
    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(),data.test_mask.clone())
        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    IsDirectedGraph = test_directed(edges)
    print("This is directed graph: ", IsDirectedGraph)
    print("data_x", data_x.shape)  # [11701, 300])

    if IsDirectedGraph and args.to_undirected:
        edges = to_undirectedBen(edges)
        print("Converted to undirected data")

    data = data.to(device)
    data_x = data_x.to(device)
    data_y = data_y.long().to(device)
    edges = edges.to(device)
    data_train_maskOrigin = data_train_maskOrigin.to(device)
    data_val_maskOrigin = data_val_maskOrigin.to(device)
    data_test_maskOrigin = data_test_maskOrigin.to(device)
    return data, data_x, data_y, edges, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin

def log_file(args):
    if args.IsDirectedData:
        dataset_to_print = args.Direct_dataset.split('/')[0]+'_'+args.Direct_dataset.split('/')[1] if len(args.Direct_dataset.split('/')) > 1 else \
        args.Direct_dataset.split('/')[0]
    else:
        dataset_to_print = args.undirect_dataset
    log_file_name = dataset_to_print+args.net+'_Aug'+str(args.AugDirect)+'_lr'+str(args.lr)+'_l2'+str(args.l2)+'_NotImprovEpoch'+str(args.NotImproved)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name_with_timestamp = f"{log_file_name}_{timestamp}.log"

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp

def geometric_dataset_sparse_Ben(q, K, args,load_only=False,  laplacian=True, gcn_appr=False):
    # if subset == '':
    #     dataset = dataset(root=root)
    # else:
    #     dataset = dataset(root=root, name=subset)
    if args.IsDirectedData:
        dataset = load_directedData(args)
    else:
        path = args.data_path
        path = osp.join(path, args.undirect_dataset)
        dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    # dataset = load_directedData(args)

    size = dataset[0].y.size(-1)
    # adj = torch.zeros(size, size).data.numpy().astype('uint8')
    # adj[dataset[0].edge_index[0], dataset[0].edge_index[1]] = 1

    f_node, e_node = dataset[0].edge_index[0], dataset[0].edge_index[1]

    label = dataset[0].y.data.numpy().astype('int')
    X = dataset[0].x.data.numpy().astype('float32')
    train_mask = dataset[0].train_mask.data.numpy().astype('bool_')
    val_mask = dataset[0].val_mask.data.numpy().astype('bool_')
    test_mask = dataset[0].test_mask.data.numpy().astype('bool_')

    if load_only:
        return X, label, train_mask, val_mask, test_mask

    try:
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian,
                                    max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=dataset[0].edge_weight)
    except AttributeError:
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian,
                                    max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)

    multi_order_laplacian = cheb_poly_sparse(L, K)

    # save_name = root + '/data' + str(q) + '_' + str(K)
    # if laplacian == False:
    #     save_name += '_P'
    # if save_pk:
    #     data = {}
    #     data['L'] = multi_order_laplacian
    #     pk.dump(data, open(save_name + '_sparse.pk', 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return X, label, train_mask, val_mask, test_mask, multi_order_laplacian
