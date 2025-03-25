# ScaleNet
After paper [Scale Invariance of Graph Neural Networks](https://arxiv.org/abs/2411.19392).

## Requirements

This repository has been tested with the following packages:
- Python == 3.9 or 3.10
- PyTorch == 2.1.2
- PyTorch Geometric == 2.4.0
- torch-scatter==2.1.2
- torch-sparse==0.6.18

Performance may vary slightly with different version of Python, GPU, cuda, PyTorch.
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
    - `citeseer/`
    - `cora_ml/`
    - `WikiCS/`
    - `telegram/`
    - `dgl/pubmed`
  
  - **Disassortative Graph**:
    - `WikipediaNetwork/squirrel`
    - `WikipediaNetwork/chameleon`

### GNN Backbone 
- **GNN baselines**:
  - `MLP`  
  - `GCN`
  - `GAT`
  - `SAGE` for GraphSAGE
  - `Cheb`
  - `APPNP`

- **Hermitian Matrix GNNs**:
  - `Mag` for MagNet
  - `Sig`  for SigMaNet
  - `Qua` for QuaNet
- **Symmetric models**: 
  -  *`Sym`* for DGCN
      - `1ym`
  - *`DiG`, `DiGi2`*  for DiGCN(ib) 
    - `1iG`
    - `RiG`
  
    In DiG\*, 1iG\*, RiG\*,  * can be nothing or exampled as follows**:
      - `i2` interception of 2-order edges
      - `u3` union of 3-order edges 
    Number 2, 3 can be replaced with any number k>1.
- **BiDirectional models**:
  - `Dir-GNN`

### How to Run

- **(1) To run and get the best performance for each model**:
  - On original datasets:

    ```
    python3 main.py  --net='ScaleNet' --use_best_hyperparams=1  --Dataset='cora_ml/'
    ```
  
    ```
    python3 main.py --net='Dir-GNN' --use_best_hyperparams=1   --Dataset='citeseer_npz/'
    
    ```
  - For imbalanced datasets:
    ```
    python3 main.py  --net='ScaleNet' --use_best_hyperparams=1  --Dataset='cora_ml/'   --MakeImbalance   --imb_ratio=100
    ```
- **(2)Run in batches**:
To run with your own configurations, revise net_nest.h by kicking in all the nets in net_values, all the layers in layer_values,
all the datasets in Direct_dataset. Then in terminal, run: 

  ```
  ./net_nest.h
  ```



- **(3) To compare ScaleNet with the enumeration of the parameters alpha, beta, and gamma, use the following command**:

  ```
  ./scale.h &
  ```

- **(4) To get performance of removing shared edges with lower-scale graphs**:
  - To get performance of AAt-A-At, AtA-A-At, AAt+AtA-A-At ('-' means removing the shared edges with A or At):
    
    args.differ_AAt=1    args.differ_AA=0
  - To get performance of AA-A-At, AtAt-A-At, AA+AtAt-A-At ('-' means removing the shared edges with A or At):
    
    args.differ_AA=1
- **(5) Wilcoxon test** 
To run the Wilcoxon test on each dataset, execute the corresponding script. For example:
  ```
  python3 ./wilcoxon/wilcoxon_cham.py  &
  ```



## License
MIT License

## Acknowledgements

The code is implemented based on [GraphSHA](https://github.com/wenzhilics/GraphSHA), [DiGCN](https://github.com/flyingtango/DiGCN),  [DirGNN](https://github.com/emalgorithm/directed-graph-neural-network)and 
[MagNet](https://github.com/matthew-hirn/magnet).

[//]: # (## Citation)
