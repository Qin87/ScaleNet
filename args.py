import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_best_hyperparams", type=int, help="use parameters in best_hyperparameters.yml", default=1)
    parser.add_argument('--GPU', type=int, default=0, help='device')
    parser.add_argument('--r20_per_class', type=int, default=0, help='1 train split is random 20 nodes_per_class')

    parser.add_argument('--originGAT', type=int, default=0, help='1 use official GAT')
    parser.add_argument('--XW', type=int, default=1, help='1 A(XW), 0 (AX)W')
    parser.add_argument('--posweight', type=str, default='abs', help='positive attention: abs, 2, e for exp,or 0 for None, softplus, softmax')
    parser.add_argument('--CPU', action='store_true', help='use CPU even has GPU')
    parser.add_argument("--BN_model", type=int, help="whether use layer normalization in model:0/1", default=0)
    parser.add_argument("--nonlinear", type=int, help="whether use activation(relu) in ScaleNet model:0/1", default=1)
    parser.add_argument("--First_self_loop", type=str, choices=["add", "remove",  0], default=0, help="Whether to add self-loops to the graph")
    parser.add_argument("--rm_gen_sloop", type=str, choices=["remove", 0], default=0, help="Whether to remove generated self-loops to the graph")

    parser.add_argument("--has_scheduler", type=int, default=1, help="Whether Optimizer has a scheduler")
    parser.add_argument('--patience', type=int, default=80, help='patience to reduce lr,')

    # for DirGNN
    parser.add_argument("--normalize", type=int, help="whether use layer normalization in ScaleNet, model:0/1", default=1)
    parser.add_argument("--inci_norm", type=str, choices=["dir", "sym", 'row', '0', 'softmax'], default='0')
    parser.add_argument('--num_split', type=int, default=10, help='num of run in spite of many splits')

    parser.add_argument('--net', type=str, default='GATv2', help='GAT, RAT, UAT, DAT, RiGib, AiGib, UiGib, DATib')
    parser.add_argument('--gt', type=int, default=0, help='whether graph transformer, 1 means all connected graph')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--Dataset', type=str, default='Coauthor-physics', help='telegram/ , cora_ml/, citeseer/,  WikiCS/'
                'PubMed, Coauthor-physics, Coauthor-CS, Amazon-Computers, Amazon-Photo, snap-patents/, WikipediaNetwork/filter_dir_chameleon')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')

    parser.add_argument('--hid_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--epoch', type=int, default=1500, help='epoch1500,')
    parser.add_argument('--NotImproved', type=int, default=810, help='consecutively Not Improved, break, 500, 450, 410, 210, 60')
    parser.add_argument('--gcn_norm', '-gcnnorm', type=int, default=1, help='GCNConv forward, normalize edge_index during training')


    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer, 5e-4')
    parser.add_argument('-hds', '--heads', default=1, type=int)

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')

    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='./dataset/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    parser.add_argument('--W_degree', type=int, default=5, help='using in-degree_0, out-degree_1, full-degree_2 for DiG edge-weight, 3 is random[1,100], 4 is random[0.1,1], 5 is random[0.0001, '
                                                                '10000], 50 is abs(sin(random5))')
    parser.add_argument('--to_undirected', '-tud', type=int, default=0, help='if convert graph to undirected')
    parser.add_argument('--to_reverse_edge', '-tre', type=int, default=0, help='if reverse direction of edges')


    args = parser.parse_args()

    return args
