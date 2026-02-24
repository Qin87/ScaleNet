import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=str, help="optimiser monitor: val_acc(acc), val_loss(loss)", default="val_acc")

    parser.add_argument("--use_best_hyperparams", type=int, default=1, help="whether use parameters in best_hyperparameters.yml")
    parser.add_argument('--GPU', type=int, default=0, help='GPU device number')
    parser.add_argument('--CPU', action='store_true', help='use CPU even has GPU')
    parser.add_argument("--BN_model", type=int, help="whether use layer normalization in model:0/1", default=0)
    parser.add_argument("--nonlinear", type=int, help="whether use activation(relu) in ScaleNet model:0/1", default=1)

    # for DirGNN
    parser.add_argument("--conv_type", type=str, help="DirGNN Model, scale, ", default="dir-gcn")
    parser.add_argument("--normalize", type=int, help="whether use batch normalization in ScaleNet, model:0/1", default=0)
    parser.add_argument("--jk", type=str, choices=["max", "cat", 'weighted',  0], default='max')
    parser.add_argument("--inci_norm", type=str, choices=["dir", "sym", 'row', 0], default="row")
    parser.add_argument("--fs", type=str, choices=["sum", "cat", 'weight_sum', 'linear'], default="dir", help='fusion method')
    parser.add_argument("--alphaDir", type=float, help="Direction convex combination params", default=0)
    parser.add_argument("--betaDir", type=float, help="Direction convex combination params", default=-1)
    parser.add_argument("--gamaDir", type=float, help="Direction convex combination params", default=-1)
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--differ_AA", type=int, default=0, help="Whether test AA-A-At")
    parser.add_argument("--differ_AAt", type=int, default=0,  help="Whether test AAt-A-At")
    parser.add_argument('--num_split', type=int, default=1, help='num of run in spite of many splits')

    parser.add_argument('--net', type=str, default='1iGib', help='Mag, Sig, QuaNet, '
                    'GCN, GAT, SAGE, Cheb, APPNP, GPRGNN, pgnn, mlp, sgc, ParaGCN, SimGAT, SloopNet, tSNE,RandomNet, HFNet '
                    'DiGib, DiGub,DiGi3, DiGi4 (1iG, RiG replace DiG), Sym, 1ym' 
                    'mlp, link, linkxcat, linkxadd, linkx, linkxgit, Dir-GNN, '
                    'ScaleNet, LargeScaleNet, FaberNet')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Dataset', type=str, default='telegram/', help=
    'telegram/, citeseer/ , cora_ml/, dgl/pubmed, WikiCS/, dgl/cora ,film/'
        'WikipediaNetwork/squirrel, WikipediaNetwork/chameleon, WebKB/Cornell, WebKB/Texas,  WebKB/Wisconsin'
        'ogbn-arxiv/, directed-roman-empire/, arxiv-year/, snap-patents/, '
        'fb100/penn94, pokec/, genius/ ')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob in DiG(ib)')

    parser.add_argument('-AP_K', '--AP_K', default=10, type=int)  # for APPNP

    parser.add_argument('--hid_dim', type=int, default=256, help='feature dimension')
    parser.add_argument('--epoch', type=int, default=10000, help='epoch1500,')
    parser.add_argument("--has_scheduler", type=int, default=0, help="Whether Optimizer has a scheduler")
    parser.add_argument('--patience', type=int, default=10, help='patience to reduce lr,80')
    parser.add_argument('--NotImproved', type=int, default=1, help='consecutively Not Improved, break, 500, 450, 410, 210, 60')

    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lrweight', type=float, default=0.4, help='learning rate for edge_weight')
    parser.add_argument('--coeflr', type=float, default=2, help='coef lr get multiplied with it')
    parser.add_argument('--wd4coef', type=float, default=5e-2, help='coef change slower with weight decay')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer, 5e-4, 0 is better')
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


    # for GPRGN
    parser.add_argument('--ppnp', default='GPR_prop',choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],default='PPR')

    # for pGCN
    parser.add_argument('--p',type=float,  default=2,help='p.')
    parser.add_argument('--mu',   type=float,default=0.1,help='mu.')

    parser.add_argument('--W_degree', type=int, default=5, help='using in-degree_0, out-degree_1, full-degree_2 for DiG edge-weight, 3 is random[1,100], 4 is random[0.1,1], 5 is random[0.0001, '
                                                                '10000], 50 is abs(sin(random5))')

    # for linkx
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')

    # for scale_big
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--checkpoint_directory", type=str, help="Directory to save checkpoints", default="checkpoint")
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=1e-3)
    parser.add_argument("--lrelu_slope", type=float, help="negative slope of Leaky Relu", default=-1.0)
    parser.add_argument("--conv_type2", type=str, help="scale, faber ", default="scale")
    parser.add_argument("--weight_penalty", type=str, choices=["exp", "lin", "None"], default="exp")
    parser.add_argument("--k_plus", type=int, help="Polynomial order", default=2)
    parser.add_argument("--exponent", type=float, help="exponent in norm, -0.25, -0.5", default=-0.5)
    parser.add_argument("--zero_order", type=int, help="If include zero order", default=0)
    parser.add_argument("--cat_A_X", type=int, help="If include concatenate A and X", default=0)
    parser.add_argument("--structure", type=float, default=0, help="1 pure structure, 0 pure feature, 0.5 structure is feature too")

    # not use for ScaleNet
    parser.add_argument("--has_1_order", type=int, help="Whether Ai* has 1-order edges:0/1", default=0)
    parser.add_argument('--paraD', action='store_true', help='ib is weighted sum')
    parser.add_argument('--gcn_norm', '-gcnnorm', type=int, default=1, help='GCNConv forward, normalize edge_index during training')
    parser.add_argument('--add_selfloop',  type=int, default=1, help='add selfloop in before model')
    parser.add_argument('--rm_gen_sloop',  type=int, default=0, help='rm selfloop in high order')

    parser.add_argument("--all1", type=int, help="feature all 1 ", default=0)
    parser.add_argument('--to_undirected', '-tud', type=int, default=0, help='if convert graph to undirected')
    parser.add_argument('--to_reverse_edge', '-tre', type=int, default=0, help='if reverse direction of edges')

    parser.add_argument('--feat_proximity', action='store_true', help='filter out non similar nodes in scaled graph')
    parser.add_argument('--ibx1', action='store_true', help='share the same ibx block in DiGSymCatib')

    parser.add_argument('--data_path', type=str, default='../dataset/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')

    parser.add_argument('--MakeImbalance', '-imbal', action='store_true', help='if convert graph to undirecteds')
    parser.add_argument('--imb_ratio', type=float, default=20, help='imbalance ratio')

    args = parser.parse_args()

    return args
