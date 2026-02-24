import os
import random

import torch_geometric.transforms as transforms
from torch_geometric.datasets import Actor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
try:
    import dgl
    from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset, FraudDataset, \
    FlickrDataset, YelpDataset, RedditDataset
except:
    print("dgl not imported, install chardet!")
import torch
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS

from data.Citation import citation_datasets
from data.preprocess import load_syn

def keep_all_data(edge_index, label, n_data, n_cls, train_mask):
    device = edge_index.device
    class_num_list = n_data
    data_train_mask = train_mask

    index_list = torch.arange(len(train_mask)).to(device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    train_node_mask = train_mask.to(device)

    edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)   # Qin revise May16
    return class_num_list, data_train_mask, idx_info, train_node_mask, edge_mask

def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def load_directedData(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_func, subset = args.Dataset.split('/')[0], args.Dataset.split('/')[1]
    if load_func == "Cora" or load_func == "CiteSeer" or load_func == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(args.data_path, load_func, transform=T.NormalizeFeatures(), split='full')

    elif load_func in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(name=load_func, transform=transforms.ToSparseTensor(), root=args.data_path)
        # evaluator = Evaluator(name=name)
        split_idx = dataset.get_idx_split()
        dataset.data.train_mask = get_mask(split_idx["train"], dataset.data.num_nodes)
        dataset.data.val_mask = get_mask(split_idx["valid"], dataset.data.num_nodes)
        dataset.data.test_mask = get_mask(split_idx["test"], dataset.data.num_nodes)
        num_train_nodes = dataset._data.train_mask.sum().item()
        num_train_nodes0 = dataset._data.val_mask.sum().item()
        num_train_nodes1 = dataset._data.test_mask.sum().item()
        print(num_train_nodes, num_train_nodes0, num_train_nodes1)

    elif load_func in ["directed-roman-empire"]:
        dataset = DirectedHeterophilousGraphDataset(name=load_func, transform=transforms.NormalizeFeatures(), root=args.data_path)
    elif load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path, is_undirected=False)
    elif load_func == 'WikiCS_U':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)        # get undirected
    elif load_func == 'cora_ml':
        dataset = citation_datasets(root='./cora_ml.npz')
    elif load_func == 'citeseer':
        dataset = citation_datasets(root='./citeseer_npz.npz')
    elif load_func in ['film']:
        dataset = Actor(root='../data/film', transform=T.NormalizeFeatures())

    elif load_func == 'dgl':    # Ben
        subset = subset.lower()
        dataset = load_dgl_graph(subset)
    elif load_func == 'telegram':
        dataset = load_syn(root='./telegram')
    else:
        dataset = load_syn(load_func+ '/'+ subset, None)

    return dataset

def load_dgl_graph(subset):
    if subset == 'citeseer':    # Nodes: 3327, Edges: 9228, Number of Classes: 6
        return CiteseerGraphDataset(reverse_edge=False)
    elif subset == 'cora':  # Nodes: 2708, Edges: 10556, Number of Classes: 7
        return CoraGraphDataset(reverse_edge=False)
    elif subset == 'pubmed':    # Nodes: 19717, Edges: 88651
        dataset = PubmedGraphDataset(reverse_edge=False)
    elif subset== 'coauthor-cs':   # bidirected
        dataset = CoauthorCSDataset()
    elif subset== 'coauthor-ph':   # bidirected
        dataset = CoauthorPhysicsDataset()
    elif subset == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif subset == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif subset == 'reddit':
        dataset = RedditDataset()
    elif subset == 'fyelp':
        dataset = FraudDataset('yelp')
    elif subset == 'famazon':
        dataset = FraudDataset('amazon')
    elif subset == 'flickr':
        dataset = FlickrDataset()
    elif subset == 'yelp':
        dataset = YelpDataset()
    # all below not working
    elif subset == 'aifb':  # Nodes: 7262, Edges: 48810 (including reverse edges)
        dataset = dgl.data.rdf.AIFBDataset(insert_reverse=False)    # don't have data_x  #
        #  assortative , node classification
    elif subset =='mutag':  # Nodes: 27163, Edges: 148100 (including reverse edges), 2 class
        dataset = dgl.data.rdf.MUTAGDataset(insert_reverse=False)   # for graph classification
    elif subset == 'bgs':   # Nodes: 94806,  Edges: 672884 (including reverse edges), 2 class
        dataset = dgl.data.rdf.BGSDataset(insert_reverse=False)     # not work to load
    elif subset == 'am':   # Nodes: 881680  Edges: 5668682 (including reverse edges)
        dataset = dgl.data.rdf.AMDataset(insert_reverse=False)
    else:
        raise NotImplementedError
    return dataset

def random_planetoid_splits(data, y, train_ratio=0.7, val_ratio=0.1, percls_trn=20,  val_lb=30, num_splits=10, Flag=1):
    # Set new random planetoid splits based on provided ratios
    num_node = y.size()[0]
    data.train_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)
    data.val_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)
    data.test_mask = torch.zeros(num_node, num_splits, dtype=torch.bool)

    for split_idx in range(num_splits):
        for i in range(y.max().item() + 1):
            index = (y == i).nonzero().view(-1)

            if Flag == 1:
                train_size = percls_trn
                val_size = val_lb
            else:       # If Flag is 0, use ratio split
                total = index.size(0)
                train_size = int(train_ratio * total)
                val_size = int(val_ratio * total)

            train_indices = index[:train_size]
            val_indices = index[train_size:train_size + val_size]
            test_indices = index[train_size + val_size:]

            # Assign masks
            data.train_mask[train_indices, split_idx] = 1
            data.val_mask[val_indices, split_idx] = 1
            data.test_mask[test_indices, split_idx] = 1
        index = index[torch.randperm(index.size(0))]

    return data

import os.path as osp
from typing import Callable, Optional
try:
    import gdown
except:
    pass
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
class DirectedHeterophilousGraphDataset(InMemoryDataset):
    r"""The directed heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace("-", "_")
        assert self.name in [
            "directed_roman_empire",
            "directed_amazon_ratings",
            "directed_questions",
        ]

        self.url = {
            "directed_roman_empire": "https://drive.google.com/uc?id=1atonwA1YqKMV3xWS7T04dRgfmDrsyRj8",
            "directed_amazon_ratings": "https://drive.google.com/uc?id=12Cyw0oZXLjPrebCficporBcIKiAgU5kc",
            "directed_questions": "https://drive.google.com/uc?id=1EnOvBehgLN3uCAQBXrGzGB1d3-aXS2Lk",
        }

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        gdown.download(self.url[self.name], f"{self.raw_dir}/{self.name}.npz", fuzzy=True)

    def process(self):
        raw = np.load(self.raw_paths[0], "r")
        x = torch.from_numpy(raw["node_features"])
        y = torch.from_numpy(raw["node_labels"])
        edge_index = torch.from_numpy(raw["edges"]).t().contiguous()
        train_mask = torch.from_numpy(raw["train_masks"]).t().contiguous()
        val_mask = torch.from_numpy(raw["val_masks"]).t().contiguous()
        test_mask = torch.from_numpy(raw["test_masks"]).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

def set_device(args):
    cuda_device = args.GPU
    if torch.cuda.is_available():
        print("cuda Device Index:", cuda_device)
        device = torch.device("cuda:%d" % cuda_device)
    else:
        print("cuda is not available, using CPU.")
        device = torch.device("cpu")
    if args.CPU:
        device = torch.device("cpu")
        print("args.CPU true, using CPU.")

    return device

def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.qinchmark = False
    random.seed(seed)
    np.random.seed(seed)
