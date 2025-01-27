import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=str, help="optimiser monitor: val_acc(acc), val_loss(loss)", default="acc")
    parser.add_argument("--all1", type=int, help="feature all 1 ", default=0)
    parser.add_argument("--all1d", type=int, help="feature dimention in all 1, 0 is keep original d ", default=1)

    parser.add_argument("--degfea", type=int, help="degree as feature: in-degree 1, out-degree -1, both 2, no 0  ", default=0)
    parser.add_argument("--use_best_hyperparams", type=int, default=0, help="whether use parameters in best_hyperparameters.yml")
    parser.add_argument('--GPU', type=int, default=0, help='GPU device')
    parser.add_argument('--CPU', action='store_true', help='use CPU even has GPU')
    parser.add_argument("--mlpIn", type=int, help="in ScaleNet, whether include mlp ", default=0)
    parser.add_argument("--mlpOut", type=int, help="in ScaleNet, whether include mlp ", default=0)
    parser.add_argument("--BN_model", type=int, help="whether use layer normalization in model:0/1", default=1)
    parser.add_argument("--nonlinear", type=int, help="whether use activation(relu) in ScaleNet model:0/1", default=1)
    parser.add_argument("--First_self_loop", type=int, choices=[1, -1,  0], default=0, help="1 is add, -1 is remove, Whether to add self-loops to the graph")
    parser.add_argument("--rm_gen_sloop", type=int, choices=[1, 0], default=0, help="Whether to remove generated self-loops to the graph")

    parser.add_argument("--has_scheduler", type=int, default=1, help="Whether Optimizer has a scheduler")
    parser.add_argument('--patience', type=int, default=80, help='patience to reduce lr,80')

    # for DirGNN
    parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="dir-gcn")
    parser.add_argument("--normalize", type=int, help="whether use batch normalization in ScaleNet, model:0/1", default=0)
    parser.add_argument("--jk", choices=["max", "cat", 'weighted',  0], default=0)
    parser.add_argument("--jk_inner", choices=["max", "cat", 'lstm', 0, 'weighted'], default=0)
    parser.add_argument("--inci_norm", choices=["dir", "sym", 'row', 0], default="sym")
    parser.add_argument("--fs", type=str, choices=["sum", "cat", 'weight_sum', 'linear'], default="dir", help='fusion method')
    parser.add_argument("--alphaDir", type=float, help="Direction convex combination params", default=0.5)
    parser.add_argument("--betaDir", type=float, help="Direction convex combination params", default=-1)
    parser.add_argument("--gamaDir", type=float, help="Direction convex combination params", default=-1)
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--differ_AA", type=int, default=0, help="Whether test AA-A-At")
    parser.add_argument("--differ_AAt", type=int, default=0,  help="Whether test AAt-A-At")
    parser.add_argument('--num_split', type=int, default=20, help='num of run in spite of many splits')

    parser.add_argument('--MakeImbalance', '-imbal', action='store_true', help='if convert graph to undirecteds')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')

    parser.add_argument('--net', type=str, default='GCN', help='mlp, Dir-GNN, ParaGCN, SimGAT, ScaleNet, SloopNet, tSNE, RandomNet, HFNet '
                     'Mag, Sig, QuaNet, '
                    'GCN, GAT, SAGE, Cheb, APPNP, GPRGNN, pgnn, mlp, sgc,'
                    'DiGib, DiGub,DiGi3, DiGi4 (1iG, RiG replace DiG)''Sym, 1ym')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Dataset', type=str, default='dgl/reddit', help='citeseer/ , cora_ml/, dgl/pubmed, telegram/,  WikiCS/, dgl/cora ,film/'
        'WikipediaNetwork/squirrel, WikipediaNetwork/chameleon, WikipediaNetwork/crocodile, WebKB/Cornell, WebKB/Texas,  WebKB/Wisconsin'
        'ogbn-arxiv/, directed-roman-empire/, arxiv-year/, snap-patents/, Amazon-Photo, Amazon-citeseer/Computers, malnet/tiny, dgl/Flickr, ')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--layer', type=int, default=1, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')

    parser.add_argument('-AP_K', '--AP_K', default=10, type=int)  # for APPNP

    parser.add_argument('--feat_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--epoch', type=int, default=1500, help='epoch1500,')
    parser.add_argument('--NotImproved', type=int, default=810, help='consecutively Not Improved, break, 500, 450, 410, 210, 60')

    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lrweight', type=float, default=0.4, help='learning rate for edge_weight')
    parser.add_argument('--coeflr', type=float, default=2, help='coef lr get multiplied with it')
    parser.add_argument('--wd4coef', type=float, default=5e-2, help='coef change slower with weight decay')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularizer, 5e-4, 0 is better')
    parser.add_argument('-hds', '--heads', default=1, type=int)

    #  from Magnet
    parser.add_argument('--q', type=float, default=0, help='q value for the phase matrix')
    parser.add_argument('--p_q', type=float, default=0.95, help='Direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='Inter-cluster edge probabilities.')
    parser.add_argument('-norm', '-n', type=int, default=0, help='if use activation function')          # no diff in 0 or 1
    parser.add_argument('-activation', '-a', type=int, default=0, help='if use activation function')        # 0 is better
    parser.add_argument('-K', '--K', default=2, type=int)  # for cheb and Mag K=1(K=2 is better for citeseer, k=3 for telegram)

    # for SigManet
    parser.add_argument('--netflow', '-N', action='store_false', help='if use net flow')
    parser.add_argument('--follow_math', '-F', action='store_false', help='if follow math')
    parser.add_argument('--gcn',  action='store_false', help='...')
    parser.add_argument('--i_complex',  action='store_false', help='...')

    # for quaNet
    parser.add_argument('--qua_weights', '-W', action='store_true', help='quaternion weights option')
    parser.add_argument('--qua_bias', '-B', action='store_true', help='quaternion bias options')

    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    # for GPRGN
    parser.add_argument('--ppnp', default='GPR_prop',choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],default='PPR')

    # for pGCN
    parser.add_argument('--p',type=float,  default=2,help='p.')
    parser.add_argument('--mu',   type=float,default=0.1,help='mu.')

    parser.add_argument('--W_degree', type=int, default=5, help='using in-degree_0, out-degree_1, full-degree_2 for DiG edge-weight, 3 is random[1,100], 4 is random[0.1,1], 5 is random[0.0001, '
                                                                '10000], 50 is abs(sin(random5))')


    # not use for ScaleNet
    parser.add_argument("--has_1_order", type=int, help="Whether Ai* has 1-order edges:0/1", default=0)
    parser.add_argument('--paraD', action='store_true', help='ib is weighted sum')
    parser.add_argument('--gcn_norm', '-gcnnorm', type=int, default=1, help='GCNConv forward, normalize edge_index during training')
    parser.add_argument('--add_selfloop',  type=int, default=1, help='add selfloop in before model 1, remove -1, 0')
    parser.add_argument('--to_undirected', '-tud', type=int, default=0, help='if convert graph to undirected')
    parser.add_argument('--to_reverse_edge', '-tre', type=int, default=0, help='if reverse direction of edges')
    parser.add_argument('--rm_bidirect_edge', '-rbe', type=int, default=0, help='make all edges directed')
    parser.add_argument('--Ak', '-Ak', type=int, default=0, help='use A^k as adjacency matrix, 0 or 1 is not')
    parser.add_argument('--num_edge', '-nedge', type=int, default=0, help='1 to print number of edges of Ak')

    parser.add_argument('--feat_proximity', action='store_true', help='filter out non similar nodes in scaled graph')
    parser.add_argument('--ibx1', action='store_true', help='share the same ibx block in DiGSymCatib')
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')


    args = parser.parse_args()

    return args
