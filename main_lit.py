################################
# PyTorch Lightning Version: Multi-scale Learning for big-sized graph
################################

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
import gc
import socket
import uuid
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint
from torch.utils.data import DataLoader
from ogb.nodeproppred import Evaluator

from utils.args import parse_args
from data.data_utils import  set_device, seed_everything
from utils.data_model import CreatModel, load_dataset, name_file, free_space, rename_log
from nets.lit_model import FullBatchGraphDataset, LightingFullBatchModelWrapper
from utils.utils import use_best_hyperparams

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # supress: oneDNN custom operations are on
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 supress warning:Unable to register cuFFT factory...
import warnings   # ScaleNet2
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)   #
import time


def main():
    seed_everything(args.seed)
    device = set_device(args)

    data_x, data_y, edges, edges_weight, num_features, data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin, IsDirectedGraph, edge_attr, data_batch = load_dataset(args)
    if args.all1:
        data_x = torch.ones_like(data_x)
    n_cls = data_y.max().item() + 1
    args.num_features, args.num_classes, args.edge_index, args.num_nodes = data_x.shape[1], n_cls, edges, data_x.shape[0]

    log_directory, log_file_name_with_timestamp = name_file(args, IsDirectedGraph)
    log_file_name_with_timestamp = 'lit_' + log_file_name_with_timestamp
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    evaluator = None
    if len(args.Dataset.split('/')) == 2:
        name = args.Dataset.split('/')[0]
        if  name in ["ogbn-arxiv", "arxiv-year"] :
            evaluator = Evaluator(name="ogbn-arxiv")

    start_time = time.time()
    with open(log_directory + log_file_name_with_timestamp, 'w') as logfile:
        print(args, file=logfile)
        print(f"Script: {__file__}", file=logfile)
        print(f"Machine ID: {socket.gethostname()}-{':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])}", file=logfile)
        sys.stdout = logfile

        # if not args.multiple_GPU:
        #     graph_data = (data_x, edges, data_y)
        #     dataset = FullBatchGraphDataset(graph_data)
        #     loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
        # else:
        data = Data(x=data_x, edge_index=edges, y=data_y)
        cluster_data = ClusterData(data, num_parts=4, recursive=False)  # 4 partitions for 4 GPUs
        loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)

        val_accs, test_accs = [], []
        for split in range(args.num_split):
            if args.num_split == 1:
                data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(), data_val_maskOrigin.clone(), data_test_maskOrigin.clone())
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

            print("\nstart split: ", split)
            model = CreatModel(args, num_features, n_cls, data_x, device, edges.shape[1]).to(device)
            lit_model = LightingFullBatchModelWrapper(
                model=model,
                args=args,
                evaluator=evaluator,
                train_mask=data_train_mask,
                val_mask=data_val_mask,
                test_mask=data_test_mask,
            )

            monitor_metric = args.monitor  # "val_loss"   "val_acc"   "train_loss"
            if "loss" in monitor_metric:
                mode = "min"
            else:
                mode = "max"

            early_stopping_callback = EarlyStopping(monitor=monitor_metric, mode=mode, patience=args.NotImproved)
            model_summary_callback = ModelSummary(max_depth=-1)
            model_checkpoint_callback = ModelCheckpoint(
                monitor=monitor_metric,
                mode=mode,
                dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
            )

            trainer = pl.Trainer(
                log_every_n_steps=1,
                enable_progress_bar=False,
                enable_model_summary=False,  # suppresses the model table  # ScaleNet2
                max_epochs=args.epoch,
                callbacks=[
                    early_stopping_callback,  # comment out will be much slower!
                    # model_summary_callback,
                    model_checkpoint_callback,  # delete will not working
                ],
                profiler="simple" if args.profiler else None,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                # devices=[args.GPU] if torch.cuda.is_available() else None,

                devices="auto",  # use all available GPUs
                strategy="ddp_find_unused_parameters_true",
            )
            if split==0:
                print(lit_model)

            # trainer.fit(lit_model, train_dataloaders=loader)
            for batch in loader:
                print('Qin', type(batch), len(batch))   # debug
            trainer.fit(lit_model, train_dataloaders=(batch for batch in loader))

            val_acc = model_checkpoint_callback.best_model_score.item()
            test_acc = trainer.test(ckpt_path="best", dataloaders=loader)[0]["test_acc"]
            test_accs.append(test_acc)
            val_accs.append(val_acc)
            print(f"Test Acc: {test_acc * 100:.2f}", file=sys.__stdout__)

            del model
            del lit_model
            del trainer
            del early_stopping_callback
            del model_summary_callback
            del model_checkpoint_callback
            torch.cuda.empty_cache()
            gc.collect()

            print('Used time: ', time.time() - start_time)

        print(f"Test Acc: {np.mean(test_accs) * 100:.2f}±{np.std(test_accs) * 100:.2f}")
        print(f"Test Acc: {np.mean(test_accs) * 100:.2f}±{np.std(test_accs) * 100:.2f}", file=sys.__stdout__)
        result_str = f"{np.mean(test_accs) * 100:.2f}±{np.std(test_accs) * 100:.2f}"

    # Rename log file
    rename_log(log_directory, log_file_name_with_timestamp, result_str)

    free_space()



if __name__ == "__main__":
    args = parse_args()
    args = use_best_hyperparams(args, args.Dataset) if args.use_best_hyperparams else args
    print(args)
    main()

