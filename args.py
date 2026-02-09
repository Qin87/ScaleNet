import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_best_hyperparams", type=int, help="use parameters in best_hyperparameters.yml", default=1)
    parser.add_argument('--GPUdevice', type=int, default=0, help='device')
    parser.add_argument('--new_edge', type=int, default=0, help='1 remove edges connect inter-class nodes')
    parser.add_argument('--r20_per_class', type=int, default=0, help='1 train split is random 20 nodes_per_class')

    parser.add_argument('--originGAT', type=int, default=1, help='1 use official GAT')
    parser.add_argument('--CPU', action='store_true', help='use CPU even has GPU')
    parser.add_argument("--BN_model", type=int, help="whether use layer normalization in model:0/1", default=0)
    parser.add_argument("--nonlinear", type=int, help="whether use activation(relu) in ScaleNet model:0/1", default=0)
    parser.add_argument("--First_self_loop", type=str, choices=["add", "remove",  0], default=0, help="Whether to add self-loops to the graph")
    parser.add_argument("--rm_gen_sloop", type=str, choices=["remove", 0], default=0, help="Whether to remove generated self-loops to the graph")


    parser.add_argument("--has_scheduler", type=int, default=1, help="Whether Optimizer has a scheduler")
    parser.add_argument('--patience', type=int, default=80, help='patience to reduce lr,')

    # for DirGNN
    parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="dir-gcn")
    parser.add_argument("--normalize", type=int, help="whether use layer normalization in ScaleNet, model:0/1", default=1)
    parser.add_argument("--inci_norm", type=str, choices=["dir", "sym", 'row', '0', 'softmax'], default='row')
    parser.add_argument('--num_split', type=int, default=10, help='num of run in spite of many splits')

    parser.add_argument('--net', type=str, default='AiGib', help='GAT, RAT, UAT, RiGib, AiGib, UiGib')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Dataset', type=str, default='citeseer/', help='telegram/ ,citeseer/,  cora_ml/, telegram/,  WikiCS/'
                'PubMed, Coauthor-physics, Coauthor-CS, Amazon-Computers, Amazon-Photo')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--layer', type=int, default=4, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')

    parser.add_argument('--hid_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--epoch', type=int, default=1500, help='epoch1500,')
    parser.add_argument('--NotImproved', type=int, default=810, help='consecutively Not Improved, break, 500, 450, 410, 210, 60')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer, 5e-4')
    parser.add_argument('-hds', '--heads', default=1, type=int)

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')

    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    parser.add_argument('--W_degree', type=int, default=5, help='using in-degree_0, out-degree_1, full-degree_2 for DiG edge-weight, 3 is random[1,100], 4 is random[0.1,1], 5 is random[0.0001, '
                                                                '10000], 50 is abs(sin(random5))')
    parser.add_argument('--to_undirected', '-tud', type=int, default=0, help='if convert graph to undirected')


    args = parser.parse_args()

    return args
