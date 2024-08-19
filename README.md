# ScaleNet
Implementation of paper [ScaleNet: Scale Invariance Learning in Directed Graphs](??).

## Requirements

This repository has been tested with the following packages:
- Python == 3.9 or 3.10
- PyTorch == 2.1.2
- PyTorch Geometric == 2.4.0
- torch-scatter==2.1.2
- torch-sparse==0.6.18

## Installation Instructions

1. Follow the official instructions to install [PyTorch](https://pytorch.org/get-started/previous-versions/).
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. For `pytorch-scatter` and `pytorch-sparse`, download the packages from [this link](https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html) according to your PyTorch, Python, and OS version. Then, use `pip` to install them.

By following these steps, you can resolve compatibility issues and avoid segmentation faults.


## Important Hyper-parameters
### Dataset
Specify the name of the dataset you want to use. The available datasets are categorized as follows:

- **Directed Datasets**:
  - **Assortative Graph**:
    - `citeseer_npz/`
    - `cora_ml/`
    - `WikiCS/`
    - `telegram/telegram`
    - `dgl/pubmed`
  
  - **Disassortative Graph**:
    - `WikipediaNetwork/squirrel`
    - `WikipediaNetwork/chameleon`

### GNN Backbone
Choose the GNN backbone to use. The available options are:
- **Models invented in this paper is composed of three parts**:
  - **(1) Graph Transformation Part**:
    - `Ti` for Tranformation Inception: each type of edges belong to one group
    - `Ui` for Union of all edges
    - `Li` for Last Edges
    - `Ii` for exhaustive independent
    - `ii` for independent
    
  - **(2) Backbone GNN part**:
    - `G` for GCN
    - `A` for GAT
    - `S` for SAGE
    - `C` for Cheb
  - **(3) Inception part**:
    - `i2` interception of 2-order edges
    - `u3` union of 3-order edges
    
  Number 2, 3 can be replaced with any number k>1.
  
- **GNN baselines**:
  - `GCN`
  - `GAT`
  - `SAGE` for GraphSAGE
  - `Cheb`
  - `Mag` for MagNet
  - `Sig`  for SigMaNet
  - `DiG`, `DiGi2`
  - `APPNP`
  - `Sym` for DGCN
  - 

Please refer to [args.py](args.py) for the full hyper-parameters.

## How to Run

Pass the above hyper-parameters to `main.py`. For example:

```
python main.py  --net GCN  --layer=3 --Dataset='cora_ml/'
```

```
python main.py --net TiGi3  --layer=2 --Dataset='citeseer_npz/'
```

(1)To compare ScaleNet with the enumeration of the parameters alpha, beta, and gamma, use the following command:

```
./scale.h
```
(2) diff_AAt
(3) diff_AA



To run in batches, revise net_nest.h by kicking in all the nets in net_values, all the layers in layer_values,
all the datasets in Direct_dataset. Then in terminal, run: 

```
./net_nest.h
```



## License
MIT License

[//]: # (## Contact )

[//]: # (Feel free to email &#40;&#41; for any questions about this work.)

## Acknowledgements

The code is implemented based on [GraphSHA](https://github.com/wenzhilics/GraphSHA), [DiGCN](https://github.com/flyingtango/DiGCN),  [DirGNN](https://github.com/emalgorithm/directed-graph-neural-network)and 
[MagNet](https://github.com/matthew-hirn/magnet).

## Citation

If you find this work is helpful to your research, please consider citing our paper:???

