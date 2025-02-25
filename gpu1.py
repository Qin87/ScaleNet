import torch
import torch_sparse
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)

# num_nodes = 5
# row = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4])
# col = torch.tensor([1, 2, 0, 3, 4, 1, 2, 3])
# adj = SparseTensor(row=row.contiguous(), col=col.contiguous(),
#                    sparse_sizes=(num_nodes, num_nodes))
# # inv_deg = torch.ones((5, 1))
# inv_deg = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)

num_nodes = 5
row = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4])
col = torch.tensor([1, 2, 0, 3, 4, 1, 2, 3])
edge_index = torch.stack((row, col), dim=0)
edge_index_t = torch.stack((col, row), dim=0)
edge_weight = torch.ones((edge_index.size(1),))

from torch_sparse import SparseTensor
adj = SparseTensor(row=row.contiguous(), col=col.contiguous(),
                   sparse_sizes=(num_nodes, num_nodes))
adj_t = SparseTensor(row=col.contiguous(), col=row.contiguous(),
                   sparse_sizes=(num_nodes, num_nodes))

print("Original adjacency matrix:")
print(adj.to_dense())

x = torch.ones((6,1))
print("adj @ x :")
print(adj @ x)

print("adj_t @ x :")
print(adj_t @ x)

from torch_sparse import sum as sparsesum
row_sum1 = sparsesum(adj, dim=1)  # row count
print("\nSum along dimension 1 (outgoing edges per node):")
print(row_sum1)

deg = torch_sparse.sum(adj, dim=1)
print("\ntorch_sparse Sum along dimension 1:")
print(deg)

deg0 = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
print("\nscatter Sum along col :")
print(deg0)

print('----------------')


row_sum0 = sparsesum(adj, dim=0)   # col count
print("\nSum along dimension 0 (ingoing edges per node):")
print(row_sum0)

row_sum1 = sparsesum(adj_t, dim=1)  # row count
print("\nadj_t Sum along dimension 1 (outgoing edges per node):")
print(row_sum1)

row_sum0 = sparsesum(adj_t, dim=0)   # col count
print("\nadj_t Sum along dimension 0 (ingoing edges per node):")
print(row_sum0)



row_scatter = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
print("\nRow Scatter using SparseTensor sum(dim=0):")
print(row_scatter)

col_scatter = scatter(edge_weight, col, 0, dim_size=num_nodes, reduce='sum')
print("\nCol Scatter using SparseTensor sum(dim=0):")
print(col_scatter)

