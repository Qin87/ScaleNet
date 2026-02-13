################################
# GAT Variants and DiGib Variants, clean version
################################
import os
import signal
import socket
import statistics
import sys
import time
import uuid

import torch
import torch.nn.functional as F

from args import parse_args
from data.data_utils import keep_all_data, seed_everything, set_device
from edge_nets.edge_data import get_second_directed_adj, \
    WCJ_get_directed_adj, Qin_get_second_directed_adj, Qin_get_directed_adj, get_appr_directed_adj2
from data_model import CreatModel, log_file, get_name, load_dataset, count_homophilic_nodes, remove_inner_class_edge
from utils import CrossEntropy, use_best_hyperparams
from sklearn.metrics import balanced_accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore")


def signal_handler(sig, frame):
    global end_time
    end_time = time.time()
    print("Process interrupted!")
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
                result_str = f"{average_acc:.1f}±{std_dev_acc:.1f}_{len(macro_F1):2d}splits"
                print(net_to_print +'_'+ str(args.layer) + '_'+dataset_to_print + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits")
                print(net_to_print +'_'+ str(args.layer) + '_'+dataset_to_print + "_acc" + f"{average_acc:.1f}±{std_dev_acc:.1f}" + "_bacc" + f"{average_bacc:.1f}±{std_dev_bacc:.1f}" + '_MacroF1:' + f"{average:.1f}±{std_dev:.1f},{len(macro_F1):2d}splits", file=log_file)
            elif len(macro_F1) == 1:
                result_str = f"{acc_list[0]:.1f}"
                print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print +"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split", file=log_file)
                print(net_to_print+'_'+str(args.layer)+'_'+dataset_to_print +"_acc"+f"{acc_list[0]:.1f}"+"_bacc" + f"{bacc_list[0]:.1f}"+'_MacroF1_'+f"{macro_F1[0]:.1f}, 1split")
            else:
                print("not a single split is finished")

            # Rename log file
            old_path = os.path.join(log_directory, log_file_name_with_timestamp)
            new_file_name = f"{result_str}_{log_file_name_with_timestamp}"
            new_path = os.path.join(log_directory, new_file_name)

            os.rename(old_path, new_path)
            print(f"Log file renamed to: {new_path}", file=sys.__stdout__)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def train(SparseEdges, edge_weight):
    global class_num_list, idx_info, prev_out, biedges
    global data_train_mask, data_val_mask, data_test_mask
    new_edge_index=None
    new_x = None
    new_y = None
    new_y_train = None
    model.train()
    optimizer.zero_grad()
    if args.net.startswith(('Di', 'Ui', 'Ri', 'Ai')) and not args.net.startswith('Dir'):
        out = model(data_x, SparseEdges, edge_weight)
    else:
        out = model(data_x, edges)
    criterion(out[data_train_mask], data_y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        if args.net.startswith(('Di', 'Ui', 'Ri', 'Ai')) and not args.net.startswith('Dir'):
            out = model(data_x, SparseEdges, edge_weight)
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
    if args.net.startswith(('Di', 'Ui', 'Ri', 'Ai')) and not args.net.startswith('Dir'):
        logits = model(data_x, SparseEdges, edge_weight)
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
edge_Qin_in_tensor = None
edge_Qin_out_tensor = None

gcn = True
IsExhaustive = False
proximity_threshold = 0.0

macro_F1 = []
acc_list = []
bacc_list = []

device = set_device(args)

data_x = data_x.to(device)
# print(data_x[0])
print(min(data_x[0]), max(data_x[0]), max(data_x[1]))
data_y = data_y.to(device)
edges = edges.to(device)
data_train_maskOrigin = data_train_maskOrigin.to(device)
data_val_maskOrigin = data_val_maskOrigin.to(device)
data_test_maskOrigin = data_test_maskOrigin.to(device)

if args.new_edge:
    edges=remove_inner_class_edge(edges, data_y)

criterion = CrossEntropy().to(device)
n_cls = data_y.max().item() + 1
args.num_nodes = data_y.size(-1)
if args.net.startswith(('Ui', 'Ri', 'Di', 'Ai')) and not args.net.startswith('Dir'):
    if args.net.startswith('Ri'):
        edge_index1, edge_weights1 = WCJ_get_directed_adj(args, edges.long(), data_x.dtype)
    elif args.net.startswith(('Ui', 'Ai')):
        edge_index1, edge_weights1 = Qin_get_directed_adj(args, edges.long())
    elif args.net.startswith('Di'):
        edge_index1, edge_weights1 = get_appr_directed_adj2(args.First_self_loop, args.alpha, edges.long(), data_y.size(-1), data_x.dtype)  # consumiing for large graph
    else:
        raise NotImplementedError("Not Implemented" + args.net)
    if args.net[-1].isdigit() or args.net[-2:] == 'ib':
        if args.net[-1] == 'b':
            k = 2
        else:
            k = int(args.net[-1])
        # if IsDirectedGraph:
        if args.net[-2] == 'i':
            if k == 2 and args.net.startswith('Di'):
                edge_list = []
                if args.net.startswith('Di'):
                    edge_index_tuple, edge_weights_tuple = get_second_directed_adj(args, edges.long(), data_y.size(-1), data_x.dtype)
                # elif args.net.startswith('Ui'):   # just for debug
                #     edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj0(edges.long(), data_y.size(-1), data_x.dtype)

                edge_list.append(edge_index_tuple)
                edge_index_tuple = tuple(edge_list)
                edge_weights_tuple = (edge_weights_tuple, )
                del edge_list
            elif args.net.startswith(('Ui', 'Ri', 'Ai')):
                edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='intersection', norm=args.inci_norm)

        elif args.net[-2] == 'u':
            edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='union', norm=args.inci_norm)
        elif args.net[-2] == 's':  # separate tuple for A_in, and A_out
            edge_index_tuple, edge_weights_tuple = Qin_get_second_directed_adj(args, edges.long(), data_y.size(-1), k, IsExhaustive, mode='separate', norm=args.inci_norm)
        else:
            raise NotImplementedError("Not Implemented" + args.net)
        # else:    # undirected graph  # TODO big BUG!
        #     edge_index_tuple, edge_weights_tuple = Qin_get_second_adj(edges.long(), data_y.size(-1), k, IsExhaustive)
        SparseEdges = (edge_index1,) + edge_index_tuple
        edge_weight = (edge_weights1,) + edge_weights_tuple
        del edge_index_tuple, edge_weights_tuple
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    del edge_index1, edge_weights1

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
        print(f"Machine ID: {socket.gethostname()}-{':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])}", file=log_file)
        print('Using Device: ', device, file=log_file)
        for split in range(num_run):
            model = CreatModel(args, num_features, n_cls, data_x, device).to(device)
            if split==0:
                print('no_in, homo_in, no_out, homo_out:', no_in, homo_ratio_A, no_out, homo_ratio_At, file=log_file)
                print(model, file=log_file)
                print(model)
                if args.net[1:].startswith('i'):
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
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)

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
            node_train = torch.sum(data_train_mask).item()

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
                val_loss, new_edge_index, new_x, new_y, new_y_train = train(SparseEdges, edge_weight)
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
                # if not epoch % 100 :            # TODO comment out
                #     end_time = time.time()
                #     print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100))
                #     print(end_time - start_time, file=log_file)
                #     print(end_time - start_time)
                #     print('epoch: {:3d}, val_loss:{:2f}, acc: {:.2f}, bacc: {:.2f}, tmp_test_f1: {:.2f}, f1: {:.2f}'.format(epoch, val_loss, test_acc * 100, test_bacc * 100, tmp_test_f1*100, test_f1 * 100),file=log_file)
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

            result_str = f"{average_acc:.1f}±{std_dev_acc:.1f}"
        else:
            result_str = f"{test_acc * 100:.1f}"


        # Rename log file
        old_path = os.path.join(log_directory, log_file_name_with_timestamp)
        new_file_name = f"{result_str}_{log_file_name_with_timestamp}"
        new_path = os.path.join(log_directory, new_file_name)

        os.rename(old_path, new_path)
        print(f"Log file renamed to: {new_path}", file=sys.__stdout__)


except KeyboardInterrupt:
    # If interrupted, the signal handler will be triggered
    pass

