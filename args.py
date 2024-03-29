import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='BitCoinAlpha', help='data set selection')
    parser.add_argument('--ensemble', type=int, default=5, help='number of ensemble model')
    parser.add_argument('--ratio', type=int, default=3, help='pos_neg ratio')
    parser.add_argument('--to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to train large graph')
    parser.add_argument('--largeData', '--large', action='store_false', help='train in batches for large graph')
    # parser.add_argument('--to_undirected', '-tud', action='store_false', help='if convert graph to undirecteds')

    # parser.add_argument('--IsDirectedData', action='store_true', help='The dataset is a directed graph')
    parser.add_argument('--IsDirectedData', type=bool, default=True, help='the dataset is directed graph')
    parser.add_argument('--AugDirect', type=int, default=0, help='0 for noAug, 1 for one direction, 2 for bidirection aug edges, 100 for link prediction'
                                                                 '4 for bidegree and bidirection, 20 for my bidegree(best), 21 for graphSHA bidegree, 2311 is trainmask use row-degree instead of 231 use col-deg, '
                                                                 '301 based on original direction')
    parser.add_argument('--net', type=str, default='QuaNet', help='addSym, UGCL,DiGSymib, DiGSymCatib, DiGSymCatMixib, DiGSymCatMixSymib, MagQin, QuaNet')
    parser.add_argument('--GPUdevice', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=100, help='seed')
    parser.add_argument('--NotImproved', type=int, default=410, help='consecutively Not Improved, break, 500, 450, 410, 210, 60')
    parser.add_argument('--undirect_dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS'], default='Cora', help='dataset name')
    parser.add_argument('--Direct_dataset', type=str, default='dgl/cora'
                                                              '', help='dgl/cora, dgl/citeseer, dgl/pubmed..., '
                                                                               'citeseer_npz/ , cora_ml/,  WikiCS/, '
                                                                              ' WikipediaNetwork/squirrel, WikipediaNetwork/chameleon '
                                                                              'WebKB/texas, WebKB/Cornell, WebKB/wisconsin, telegram/telegram')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')
    # parser.add_argument('--MakeImbalance', type=bool, default=False, help='True for turn dataset into imbalanced')
    parser.add_argument('--MakeImbalance', type=bool, default=True, help='True for turn dataset into imbalanced')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    parser.add_argument('-K', '--K', default=2, type=int)  # for cheb
    parser.add_argument('-AP_K', '--AP_K', default=10, type=int)  # for APPNP

    parser.add_argument('--feat_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
    parser.add_argument('--epoch', type=int, default=1500, help='epoch1500,')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer, 5e-4')
    parser.add_argument('--tau', type=int, default=2, help='temperature in the softmax function when calculating confidence-based node hardness')
    parser.add_argument('--max', action="store_true", help='synthesizing to max or mean num of training set. default is mean') 
    parser.add_argument('--no_mask', action="store_true", help='whether to mask the self class in sampling neighbor classes. default is mask')
    parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr', help='how to convert to weighted graph')

    parser.add_argument('-hds', '--heads', default=8, type=int)

    #  from Magnet
    parser.add_argument('--q', type=float, default=0.25, help='q value for the phase matrix')
    parser.add_argument('--p_q', type=float, default=0.95, help='Direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='Inter-cluster edge probabilities.')
    parser.add_argument('-norm', '-n', action='store_true', help='if use activation function')
    parser.add_argument('-activation', '-a', action='store_true', help='if use activation function')
    # parser.add_argument('-activation', '-a', action='store_false', help='if use activation function')

    # for SigManet
    parser.add_argument('--netflow', '-N', action='store_false', help='if use net flow')
    parser.add_argument('--follow_math', '-F', action='store_false', help='if follow math')
    parser.add_argument('--gcn',  action='store_false', help='...')
    parser.add_argument('--i_complex',  action='store_false', help='...')

    # for edge prediction
    parser.add_argument('--task', type=str, default='three_class_digraph', help='Task: three_class_digraph,  direction, existence, ...')
    parser.add_argument('--method_name', type=str, default='DiG', help='method name')
    parser.add_argument('--num_class_link', type=int, default=3,
                        help='number of classes for link direction prediction(2 or 3).')

    # for quaNet
    parser.add_argument('--qua_weights', '-W', action='store_true', help='quaternion weights option')
    parser.add_argument('--qua_bias', '-B', action='store_true', help='quaternion bias options')

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=64, help='num of filters')

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    args = parser.parse_args()

    return args
