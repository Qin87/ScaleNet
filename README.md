# Position: Attention Offers Little Benefit for Graph Node Classification

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
    - `citeseer/`
    - `cora_ml/`
    - `WikiCS/`
    - `telegram/telegram`

- **Undirected Datasets**:  
  - `Amazon-Photo`
  - `Amazon-Computers`
  - `PubMed`
  - `Coauthor-physics`
  - `Coauthor-CS`


### GNN models
- **GAT variants**:
  - `GAT`  for attention weights
  - `UAT`  for uniform weights(all 1)
  - `RAT`  for random weights
  
  set --inci_norm: softmax  dir sym row 0 for different normalizations

  
- **DiGib variants**: 

  - `DiGib`
  - `UiGib`
  - `RiGib`
  
  set --inci_norm: softmax  dir sym row 0 for different normalizations



### How to Run

- **(1) To run and get the best performance for each model**:
  - On original datasets:

    ```
    python3 main.py  --net='RAT' --use_best_hyperparams=1  --Dataset='cora_ml/'   --inci_norm='sym'
    ```
  
    ```
    python3 main.py --net='UAT' --use_best_hyperparams=1   --Dataset='citeseer/'   --inci_norm='softmax'
    
    ```

- **(2)Run in batches**:
To run with your own configurations, revise net_nest.h by kicking in all the nets in net_values, all the layers in layer_values,
all the datasets in Direct_dataset. Then in terminal, run: 

  ```
  ./net_nest.h
  ```





## License
MIT License

## Acknowledgements

The code is implemented based on [GraphSHA](https://github.com/wenzhilics/GraphSHA), [DiGCN](https://github.com/flyingtango/DiGCN),  [DirGNN](https://github.com/emalgorithm/directed-graph-neural-network)and 
[MagNet](https://github.com/matthew-hirn/magnet).



