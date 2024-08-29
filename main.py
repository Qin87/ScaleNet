################################
# this version to ensure that when I stop the process half way, it still could print the result.
################################
import os
import signal
import statistics
import sys
import time


import torch
import torch.nn.functional as F

from args import parse_args
from data.data_utils import keep_all_data, seed_everything, set_device
from edge_nets.edge_data import get_second_directed_adj, get_second_directed_adj_union, \
    WCJ_get_directed_adj, Qin_get_second_directed_adj, Qin_get_directed_adj, get_appr_directed_adj2, Qin_get_second_directed_adj0, Qin_get_second_adj, Qin_get_all_directed_adj, normalize_row_edges
from data_model import CreatModel, log_file, get_name, load_dataset, feat_proximity, delete_edges, make_imbalanced, count_homophilic_nodes
from nets.DiG_NoConv import union_edges
from nets.src2 import laplacian
from nets.src2.quaternion_laplacian import process_quaternion_laplacian
from data.preprocess import  F_in_out, F_in_out0
from utils import CrossEntropy, use_best_hyperparams
from sklearn.metrics import balanced_accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore")


def signal_handler(sig, frame):
    global end_time
    end_time = time.time()
    print("Process interrupted!")
    # calculate_time()
    log_results()
    sys.exit(0)

def log_results():
    global start_time, end_time
    if start_time is not None and end_time is not None:
        with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
            elapsed_time = end_time - start_time
            print("Total time: {:.2f} seconds".format(elapsed_time), file=log_file)
            print("Total time: {:.2f} seconds".format(elapsed_time))
            if len(macro_F1) > 1:
                average = statistics.mean(macro_F1)
                std_dev = statistics.stdev(macro_F1)
                average_acc = statistics.mean(acc_list)
                std_dev_acc = statistics.stdev(acc_list)
                average_bacc = statistics.mean(bacc_list)
                std_dev_bacc = statistics.stdev(bacc_list)
                print(net_to_print +'_'+ str(args.layer) + '_'+dataset_to_print + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits")
                print(net_to_print +'_'+ str(args.layer) + '_'+dataset_to_print + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits", file=log_file)
            elif len(macro_F1) == 1:
                print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print +"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split", file=log_file)
                print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print +"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split")
            else:
                print("not a single split is finished")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def train(edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real, norm_imag,
          X_img_i, X_img_j, X_img_k,norm_img_i,norm_img_j, norm_img_k, Quaedge_index):
    global class_num_list, idx_info, prev_out, biedges
    global data_train_mask, data_val_mask, data_test_mask
    new_edge_index=None
    new_x = None
    new_y = None
    new_y_train = None
    model.train()
    if args.net.endswith('ymN1'):   # without 1st-order edges
        biedges = edge_in
    optimizer.zero_grad()
    if args.net.startswith(('Sym', 'addSym', '1ym', 'addQym')):
        out = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight)
    elif args.net.startswith(('Di', '1i', 'Ri', 'Ui', 'Li', 'Ai', 'Ti',  'Hi', 'Ii', 'ii')) and not args.net.startswith('Dir'):
        if args.net[3:].startswith(('Sym', '1ym')):
            out = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
        else:
            out = model(data_x, SparseEdges, edge_weight)
    elif args.net.startswith('Mag'):
        out = model(X_real, X_img, edges, args.q, edge_weight)  # (1,5,183)
    elif args.net.startswith('Sig'):
        out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    elif args.net.startswith('Qua'):
        out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
    else:
        out = model(data_x, edges)
    criterion(out[data_train_mask], data_y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        if args.net.startswith(('Sym', 'addSym', '1ym', 'addQym')):
            out = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight)
        elif args.net.startswith(('Di', '1i', 'Ri', 'Ui', 'Li', 'Ti', 'Ai', 'Hi', 'Ii', 'ii')) and not args.net.startswith('Dir'):
            if args.net[3:].startswith(('Sym', '1ym')):
                out = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
            else:
                out = model(data_x, SparseEdges, edge_weight)
        elif args.net.startswith('Mag'):
            out = model(X_real, X_img, edges, args.q, edge_weight)
        elif args.net.startswith('Sig'):
            out = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
        elif args.net.startswith('Qua'):  #
            out = model(X_real, X_img_i, X_img_j, X_img_k,norm_img_i, norm_img_j, norm_img_k, norm_real,Quaedge_index)
        else:
            out = model(data_x, edges)
        val_loss = F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
    optimizer.step()
    if args.has_scheduler:
        scheduler.step(val_loss, epoch)

    return val_loss, new_edge_index, new_x, new_y, new_y_train

@torch.no_grad()
def test():
    global edge_in, in_weight, edge_out, out_weight
    model.eval()
    if args.net.startswith(('Sym', 'addSym', '1ym', 'addQym')):
        logits = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight)
    elif args.net.startswith(('Di', '1i', 'Ri', 'Ui', 'Li', 'Ti', 'Ai', 'Hi', 'Ii', 'ii')) and not args.net.startswith('Dir'):
        if args.net[3:].startswith(('Sym', '1ym')):
            logits = model(data_x, biedges, edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight)
        else:
            logits = model(data_x, SparseEdges, edge_weight)
    elif args.net.startswith('Mag'):
        logits = model(X_real, X_img, edges, args.q, edge_weight)
    elif args.net.startswith('Sig'):  #
        logits = model(X_real, X_img, norm_real, norm_imag, Sigedge_index)
    elif args.net.startswith('Qua'):  #
        logits = model(X_real, X_img_i, X_img_j, X_img_k, norm_imag_i, norm_imag_j, norm_imag_k, norm_real, Quaedge_index)
    else:
        logits = model(data_x, edges[:, train_edge_mask])
    accs, baccs, f1s = [], [], []
    for mask in [data_train_mask, data_val_mask, data_test_mask]:
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data_y[mask].cpu().numpy()
        acc = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
    return accs, baccs, f1s


start_time = time.time()
args = parse_args()
args = use_best_hyperparams(args, args.Dataset) if args.use_best_hyperparams else args

data_x, data_y, edges, edges_weight, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph = load_dataset(args)
net_to_print, dataset_to_print = get_name(args, IsDirectedGraph)
load_time = time.time()

log_directory, log_file_name_with_timestamp = log_file(net_to_print, dataset_to_print, args)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
print(args)
with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
    print(args, file=log_file)

seed_everything(args.seed)

no_in, homo_ratio_A, no_out,   homo_ratio_At = count_homophilic_nodes(edges, data_y)

biedges = None
edge_in = None
in_weight = None
edge_out = None
out_weight = None
SparseEdges = None
edge_weight = None
X_img = None
X_real = None
edge_Qin_in_tensor = None
edge_Qin_out_tensor = None
Sigedge_index = None
norm_real = None
norm_imag = None
norm_imag_i =None
norm_imag_j = None
norm_imag_k = None
Quaedge_index = None
X_img_i = None
X_img_j = None
X_img_k = None

gcn = True
IsExhaustive = False
proximity_threshold = 0.0

macro_F1 = []
acc_list = []
bacc_list = []

device = set_device(args)

data_x = data_x.to(device)
data_y = data_y.to(device)
edges = edges.to(device)
data_train_maskOrigin = data_train_maskOrigin.to(device)
data_val_maskOrigin = data_val_maskOrigin.to(device)
data_test_maskOrigin = data_test_maskOrigin.to(device)


criterion = CrossEntropy().to(device)
n_cls = data_y.max().item() + 1

if args.net.startswith(('1i', 'Ri', 'Di', 'pan', 'Ui', 'Li', 'Ti', 'Ai', 'Hi','Ii', 'ii')) and not args.net.startswith('Dir'):
    if args.feat_proximity:
        average_distance, threshold_value = feat_proximity(edges, data_x)
        proximity_threshold = threshold_value
    if args.net.startswith('Ri'):
        edge_index1, edge_weights1 = WCJ_get_directed_adj(args, edges.long(), data_y.size(-1), data_x.dtype)
    elif args.net.startswith(('1i', 'pan', 'Ui', 'Li', 'Ti', 'Ai', 'Hi', 'Ii', 'ii')):
        edge_index1, edge_weights1 = Qin_get_directed_adj(args, edges.long(), data_y.size(-1), data_x.dtype)
    elif args.net.startswith('Di'):
        edge_index1, edge_weights1 = get_appr_directed_adj2(args.First_self_loop, args.alpha, edges.long(), data_y.size(-1), data_x.dtype)  # consumiing for large graph
    else:
        raise NotImplementedError("Not Implemented" + args.net)
    if args.net[-1].isdigit() or args.net[-2:] == 'ib':
        if args.net[-1] == 'b':
            k = 2
        else:
            k = int(args.net[-1])
        if args.net.startswith(('Ti', 'Ai', 'Hi')):       # Hi is heterogeneous
            IsExhaustive = True
        if IsDirectedGraph:
            if args.net.startswith('Ai'):
                edge_index_tuple, edge_weights_tuple = Qin_get_all_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='independent', norm=args.inci_norm)
            elif args.net.startswith('Ii'):
                IsExhaustive = True
                edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='independent', norm=args.inci_norm)
            elif args.net.startswith('ii'):
                IsExhaustive = False
                edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='independent', norm=args.inci_norm)
            elif args.net[-2] == 'i':
                if k == 2 and args.net.startswith('Di'):
                    edge_list = []
                    if args.net.startswith('Di'):
                        edge_index_tuple, edge_weights_tuple = get_second_directed_adj(args, edges.long(), data_y.size(-1), data_x.dtype)
                    else:   # just for debug
                        edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj0(edges.long(), data_y.size(-1), data_x.dtype)
                    edge_list.append(edge_index_tuple)
                    edge_index_tuple = tuple(edge_list)
                    edge_weights_tuple = (edge_weights_tuple, )
                    del edge_list
                else:
                    edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='intersection', norm=args.inci_norm)
            elif args.net[-2] == 'u':
                edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='union', norm=args.inci_norm)
            elif args.net[-2] == 's':  # separate tuple for A_in, and A_out
                edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='separate', norm=args.inci_norm)
            else:
                raise NotImplementedError("Not Implemented" + args.net)
        else:    # undirected graph
            edge_index_tuple, edge_weights_tuple = Qin_get_second_adj(edges.long(), data_y.size(-1), k, IsExhaustive)
        if args.net.startswith(('Hi', 'Ai')):
            SparseEdges = edge_index_tuple
            edge_weight = edge_weights_tuple
        else:
            SparseEdges = (edge_index1,) + edge_index_tuple
            edge_weight = (edge_weights1,) + edge_weights_tuple
        del edge_index_tuple, edge_weights_tuple
        if args.net.startswith('Ui'):
            SparseEdges, edge_weight = union_edges(data_x.size()[0], SparseEdges, device, mode='union')
        elif args.net.startswith('Li'):
            SparseEdges, edge_weight = union_edges(data_x.size()[0], SparseEdges, device, mode='last')
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1
    if args.feat_proximity:
        if not isinstance(SparseEdges, tuple):
            SparseEdges = delete_edges(SparseEdges, data_x, threshold_value).to(device)
            edge_weight = normalize_row_edges(SparseEdges, data_x.size()[0]).to(device)
            # SparseEdges = (SparseEdges,)  normalize_row_edges(edge_index, num_nodes, edge_weight)
        else:
            proximity_edges = []
            proximity_weights = []
            for edge_index1 in SparseEdges:
                filtered_edges = delete_edges(edge_index1, data_x, threshold_value).to(device)
                filtered_edge_weights = normalize_row_edges(filtered_edges, data_x.size()[0]).to(device)
                print("num_edge change from {} to {}".format(edge_index1.shape[1], filtered_edge_weights.shape[0]))
                proximity_edges.append(filtered_edges)
                proximity_weights.append(filtered_edge_weights)
            SparseEdges = tuple(proximity_edges)
            edge_weight = tuple(proximity_weights)
            del proximity_edges, proximity_weights

    if args.net[3:].startswith('1ym'):
        biedges, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(), data_y.size(-1), edges_weight)
    elif args.net[3:].startswith('Sym'):
        biedges, edge_in, in_weight, edge_out, out_weight = F_in_out0(edges.long(), data_y.size(-1), edges_weight)

elif args.net.startswith(('Sym', 'addSym', '1ym', 'addQym')):
    if args.net.startswith(('1ym', 'addQym')):
        biedges, edge_in, in_weight, edge_out, out_weight = F_in_out(edges.long(),data_y.size(-1),edges_weight)
    else:
        biedges, edge_in, in_weight, edge_out, out_weight = F_in_out0(edges.long(),data_y.size(-1),edges_weight)
elif args.net.startswith(('Mag', 'Sig', 'Qua')):
    data_x_cpu = data_x.cpu()
    X_img_i = torch.FloatTensor(data_x_cpu).to(device)
    X_img_j = torch.FloatTensor(data_x_cpu).to(device)
    X_img_k = torch.FloatTensor(data_x_cpu).to(device)
    X_img = torch.FloatTensor(data_x_cpu).to(device)
    X_real = torch.FloatTensor(data_x_cpu).to(device)
    if args.net.startswith('Sig'):
        Sigedge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edges, gcn=gcn, net_flow=args.netflow, x_real=X_real, edge_weight=edge_weight,
                                                                                   normalization='sym', return_lambda_max=False)
    elif args.net.startswith('Qua'):
        Quaedge_index, norm_real, norm_imag_i, norm_imag_j, norm_imag_k = process_quaternion_laplacian(edge_index=edges, x_real=X_real, edge_weight=edge_weight,
                                                                                                    normalization='sym', return_lambda_max=False)
else:
    pass
try:
    splits = data_train_maskOrigin.shape[1]
    print("splits", splits)
    if len(data_test_maskOrigin.shape) == 1:
        data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
except:
    splits = 1
Set_exit = False

num_run = args.num_split if args.num_split<splits else splits
preprocess_time = time.time()
try:
    with open(log_directory + log_file_name_with_timestamp, 'a') as log_file:
        print('Using Device: ', device, file=log_file)
        for split in range(num_run):
            model = CreatModel(args, num_features, n_cls, data_x, device).to(device)
            if split==0:
                print('no_in, homo_in, no_out, homo_out:', no_in, homo_ratio_A, no_out, homo_ratio_At, file=log_file)
                print(model, file=log_file)
                print(model)
                if args.net.startswith('ym'):
                    print('Sym edge size(biedge, edge_in, edge_out):', biedges.size(),  in_weight.size(),  out_weight.size(), file=log_file)
                    print('Sym edge size(biedge, edge_in, edge_out):', biedges.size(),  in_weight.size(),  out_weight.size())
                elif args.net[1:].startswith('i'):
                    print(args.net, 'edge size:', end=' ', file=log_file)
                    print(args.net, 'edge size:', end=' ')
                    if isinstance(edge_weight, tuple):
                        for i in edge_weight:
                            print(i.size()[0], end=' ', file=log_file)
                            print(i.size()[0], end=' ')
                    else:
                        if edge_weight is not None:
                            print(edge_weight.size()[0], end=' ', file=log_file)
                            print(edge_weight.size()[0], end=' ')

            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
            if hasattr(model, 'coefs'):     # parameter without weight_decay will typically change faster
                optimizer = torch.optim.Adam(
                    [dict(params=model.reg_params, lr=args.lr, weight_decay=5e-4), dict(params=model.non_reg_params, lr=args.lr, weight_decay=0),
                     dict(params=model.coefs, lr=args.coeflr * args.lr, weight_decay=args.wd4coef), ],
                )
            elif hasattr(model, 'reg_params'):
                optimizer = torch.optim.Adam(
                    [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            if args.has_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=False)

            if splits == 1:
                data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),data_val_maskOrigin.clone(),data_test_maskOrigin.clone())
            else:
                try:
                    data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin[:, split].clone(),
                                                                      data_val_maskOrigin[:, split].clone(),
                                                                      data_test_maskOrigin[:, split].clone())
                except IndexError:
                    print("testIndex ,", data_test_mask.shape, data_train_mask.shape, data_val_mask.shape)
                    data_train_mask, data_val_mask = (
                        data_train_maskOrigin[:, split].clone(), data_val_maskOrigin[:, split].clone())
                    try:
                        data_test_mask = data_test_maskOrigin[:, 1].clone()
                    except:
                        data_test_mask = data_test_maskOrigin.clone()

            n_data0 = []  # num of train in each class
            for i in range(n_cls):
                data_num = (data_y == i).sum()
                n_data0.append(int(data_num.item()))
            if split == 0:
                print('class in data: ', sorted(n_data0))

            stats = data_y[data_train_mask]  # this is selected y. only train nodes of y
            n_data = []  # num of train in each class
            for i in range(n_cls):
                data_num = (stats == i).sum()
                n_data.append(int(data_num.item()))
            # idx_info = get_idx_info(data_y, n_cls, data_train_mask, device)  # torch: all train nodes for each class
            node_train = torch.sum(data_train_mask).item()

            if args.MakeImbalance:
                print("make imbalanced")
                print("make imbalanced", file=log_file)
                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                    make_imbalanced(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
                if split==0:
                    print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNodeBal_' + str(node_train) + '\t trainNodeImbal_' + str(torch.sum(
                        data_train_mask).item()), file=log_file)
                    print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdgeBal_' + str(train_edge_mask.size()[0]) + '\t trainEdgeImbal_' + str(torch.sum(
                        train_edge_mask).item()), file=log_file)
                    print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNodeBal_' + str(node_train) + '\t trainNodeImbal_' + str(torch.sum(
                        data_train_mask).item()))
                    print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdgeBal_' + str(train_edge_mask.size()[0]) + '\t trainEdgeImbal_' + str(torch.sum(
                        train_edge_mask).item()))
            else:
                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                    keep_all_data(edges, data_y, n_data, n_cls, data_train_mask)
                if split == 0:
                    print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNode_' + str(node_train), file=log_file)
                    print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdge_' + str(train_edge_mask.size()[0]), file=log_file)
                    print(dataset_to_print + '\ttotalNode_' + str(data_train_mask.size()[0]) + '\t trainNodeBal_' + str(node_train) + '\t trainNodeNow_' + str(torch.sum(
                        data_train_mask).item()))
                    print(dataset_to_print + '\ttotalEdge_' + str(edges.size()[1]) + '\t trainEdgeBal_' + str(train_edge_mask.size()[0]) + '\t trainEdgeNow_' + str(
                        torch.sum(train_edge_mask).item()))
                    sorted_list = sorted(class_num_list, reverse=True)
                    sorted_list_original = sorted(n_data, reverse=True)
                    print('class_num_list is ', n_data)
                    print('sorted class_num_list is ', sorted_list_original)

            sorted_list = sorted(class_num_list, reverse=True)
            sorted_list_original = sorted(n_data, reverse=True)
            if split == 0:
                if sorted_list[-1]:
                    imbalance_ratio_origin = sorted_list_original[0] / sorted_list_original[-1]
                    print('Origin Imbalance ratio is {:.1f}'.format(imbalance_ratio_origin))
                    # imbalance_ratio = sorted_list[0] / sorted_list[-1]
                    # print('New    Imbalance ratio is {:.1f}'.format(imbalance_ratio))
                else:
                    print('the minor class has no training sample')

            train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
            val_idx = data_val_mask.nonzero().squeeze()  # get the index of training data
            test_idx = data_test_mask.nonzero().squeeze()  # get the index of training data
            labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
            train_idx_list = train_idx.cpu().tolist()
            local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
            global2local = dict([val, key] for key, val in local2global.items())
            idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
            idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                              idx_info_list]  # train nodes position inside train

            best_val_loss = 100
            best_val_acc_f1 = 0
            best_val_acc = 0
            best_val_f1 = 0
            best_test_f1 = 0
            saliency, prev_out = None, None
            test_acc, test_bacc, test_f1 = 0.0, 0.0, 0.0
            CountNotImproved = 0
            end_epoch = 0
            for epoch in range(args.epoch):
                val_loss, new_edge_index, new_x, new_y, new_y_train = train(edge_in, in_weight, edge_out, out_weight, SparseEdges, edge_weight, X_real, X_img, Sigedge_index, norm_real,norm_imag,
                                                                                X_img_i, X_img_j, X_img_k,norm_imag_i, norm_imag_j, norm_imag_k, Quaedge_index)
                accs, baccs, f1s = test()
                train_acc, val_acc, tmp_test_acc = accs
                train_f1, val_f1, tmp_test_f1 = f1s
                val_acc_f1 = (val_acc + val_f1) / 2.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    test_acc = accs[2]
                    test_bacc = baccs[2]
                    test_f1 = f1s[2]
                    CountNotImproved = 0
                    # print('test_f1 CountNotImproved reset to 0 in epoch', epoch, file=log_file)
                else:
                    CountNotImproved += 1
                if not epoch % 100 :
                    # end_time = time.time()
                    print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100))
                    # print(end_time - start_time, file=log_file)
                    # print(end_time - start_time)
                    # print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100),file=log_file)
                end_epoch = epoch
                if CountNotImproved > args.NotImproved:
                    # print("No improved for consecutive {:3d} epochs, break.".format(args.NotImproved))
                    break
            dataset_to_print = args.Dataset.replace('/', '_') + str(args.to_undirected)
            print(net_to_print+'layer'+str(args.layer), dataset_to_print, 'EndEpoch', str(end_epoch), 'lr', args.lr)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100))
            print(net_to_print, args.layer, dataset_to_print, 'EndEpoch', str(end_epoch), 'lr', args.lr, file=log_file)
            print('Split{:3d}, acc: {:.2f}, bacc: {:.2f}, f1: {:.2f}'.format(split, test_acc * 100, test_bacc * 100, test_f1 * 100), file=log_file)
            macro_F1.append(test_f1*100)
            acc_list.append(test_acc*100)
            bacc_list.append(test_bacc*100)
            if Set_exit:
                sys.exit(1)

        last_time = time.time()
        elapsed_time0 = last_time-start_time
        print("Time(s): Total_{}= Load_{} + Preprocess_{} + Train_{}".format(int(last_time-start_time), int(load_time-start_time), int(preprocess_time-load_time), int(last_time-preprocess_time)),
              file=log_file)
        print(
            "Time(s): Total_{}= Load_{} + Preprocess_{} + Train_{}".format(int(last_time - start_time), int(load_time - start_time), int(preprocess_time - load_time), int(last_time - preprocess_time)))
        if len(macro_F1) > 1:
            average = statistics.mean(macro_F1)
            std_dev = statistics.stdev(macro_F1)
            average_acc = statistics.mean(acc_list)
            std_dev_acc = statistics.stdev(acc_list)
            average_bacc = statistics.mean(bacc_list)
            std_dev_bacc = statistics.stdev(bacc_list)
            print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print+"_acc"+f"{average_acc:.1f}±{std_dev_acc:.1f}"+"_bacc"+f"{average_bacc:.1f}±{std_dev_bacc:.1f}"+'_Macro F1:'+f"{average:.1f}±{std_dev:.1f}")
            print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print+"_acc"+f"{average_acc:.1f}±{std_dev_acc:.1f}"+"_bacc"+f"{average_bacc:.1f}±{std_dev_bacc:.1f}"+'_Macro F1:'+f"{average:.1f}±{std_dev:.1f}", file=log_file)



except KeyboardInterrupt:
    # If interrupted, the signal handler will be triggered
    pass

