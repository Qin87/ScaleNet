import itertools
from scipy.sparse import  csr_matrix
from torch import Tensor
from torch_sparse import SparseTensor, coalesce
import torch
import torch.distributions as dist
import scipy
from torch_geometric.utils import  add_self_loops, remove_self_loops
from torch_scatter import scatter_add

from utils.utils import get_norm_adj


def get_second_directed_adj(args,  edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    if args.inci_norm is not None:
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def Qin_get_second_directed_adj0(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    p_dense = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L = L_in
    L[L_out == 0] = 0        # intersection

    # L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    edge_index = L_indices
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def Qin_get_directed_adj(args, edge_index, num_nodes, dtype, edge_weight=None):
    norm = args.inci_norm
    device = edge_index.device
    edge_index = torch.unique(edge_index, dim=1).to(device)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1).to(device)

    # type 1: conside different inci-norm
    if norm =='dir' or norm =='row':
        row, col = edge_index
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        # all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        edge_weight = adj_norm.storage.value()
    # type 2: only GCN_norm
    elif norm == 'sym':
        edge_weight = normalize_row_edges(edge_index, num_nodes).to(device)
    elif norm == 0 or norm is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

    return edge_index,  edge_weight


def WCJ_get_directed_adj(args, edge_index, num_nodes, dtype, edge_weight=None):
    norm = args.inci_norm
    # norm = 'sym'
    # norm = 0
    W_degree = args.W_degree
    # random value to edge weights
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    row, col = edge_index
    deg0 = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)  # row degree
    deg1 = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes).to(device)  # col degree
    deg2 = deg0 + deg1

    # plt.hist(deg0.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Original Distribution of  degree0:NPZ')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if W_degree == 0:  # in-degree
        edge_weight = deg0[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg0")
    elif W_degree == 1:     # out-degree
        edge_weight = deg1[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg1")
    elif W_degree == 2:     # total-degree
        edge_weight = deg2[edge_index[0]] + deg0[edge_index[1]]
        print("Using deg2")
    elif W_degree == 3:     # random number in [1,100]
        edge_weight = torch.randint(1, 101, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 300:     # random number in [1,10000]
        edge_weight = torch.randint(1, 10001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 30000:     # random number in [1,1000000]
        edge_weight = torch.randint(1, 1000001, (edge_index.size(1),), dtype=dtype, device=edge_index.device)
        print("proximity weight is random number in [1,100]")
    elif W_degree == 400:  # random number in [0.001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.999 + 0.001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 40000:  # random number in [0.00001,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.99999 + 0.00001
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 4:  # random number in [0.1,1]       # random number in [0.1,1]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * 0.9 + 0.1
        print("proximity weight is random number in [0.1,1]")
    elif W_degree == 5:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        # edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (1 - 0.0001)-0.5

        # indices = torch.randperm(edge_index.size(1), device=edge_index.device)[edge_index.size(1) // 2:]
        # edge_weight[indices] = 0

        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == 50:  # random number in [0.00001,100000]
        edge_weight = torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (10000 - 0.0001) + 0.0001
        edge_weight = torch.abs(torch.sin(edge_weight))
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
        # edge_weight = random_values * (10000 - 0.0001) + 0.0001
    elif W_degree == -3:   # three peaks
        edge_weight = trimodal_distribution(edge_index.size(1), edge_index.device, dtype)
    elif W_degree == -2:   # two peaks
        edge_weight = trimodal_distribution2(edge_index.size(1), edge_index.device, dtype)
        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Original Edge weight range: [{min_val}, {max_val}]")
    elif W_degree == -4:   # two peaks
        edge_weight = trimodal_distribution4(edge_index.size(1), edge_index.device, dtype)
    else:
        NotImplementedError('Not Implemented edge-weight type')

    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Original Distribution of  WiG-2 edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    if norm == 'sym':
        # row normalization
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif norm == 'dir':
        # type 1: conside different inci-norm
        row, col = edge_index
        deg_row = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_col = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

        row_deg_inv_sqrt = deg_row.pow(-0.5)
        row_deg_inv_sqrt[row_deg_inv_sqrt == float('inf')] = 0

        col_deg_inv_sqrt = deg_col.pow(-0.5)
        col_deg_inv_sqrt[col_deg_inv_sqrt == float('inf')] = 0

        edge_weight = row_deg_inv_sqrt[row] * edge_weight * col_deg_inv_sqrt[col]


        # adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        # # all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        # edge_weight = adj_norm.storage.value()

        # type 2: only GCN_norm
        # edge_weight = normalize_row_edges(edge_index, num_nodes).to(device)
    else:
        pass

    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Normalized Edge weight range: [{min_val}, {max_val}]")

    # # plt.xlim(0, 2)
    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Normalized Distribution of  WiG-2 edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    return edge_index,  edge_weight


def get_appr_directed_adj2(args, edge_index, num_nodes, dtype, edge_weight=None):
    selfloop, alpha = args.add_selfloop, args.alpha
    device = edge_index.device

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    if selfloop == 1:
        edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value=1, num_nodes=num_nodes)  # with selfloop, QiG get better
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).to(device)
    deg_inv = deg.pow(-1).to(device)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # plt.hist(deg.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('degree')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of degree:Cora')       # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense().to(device)
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1])).to(device)     # dummy node
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense      # original P
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v.cpu()   # for p_ppr.numpy()  # this is new P with one dummy node

    p_ppr_sparse = csr_matrix(p_ppr.numpy())
    eig_value, left_vector = scipy.linalg.eig(p_ppr_sparse.toarray(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real).to(device)     # Qin ask: why only real?       # converting a NumPy array containing eigenvalues to a PyTorch tensor
    left_vector = torch.from_numpy(left_vector.real).to(device)
    val, ind = eig_value.sort(descending=True)
    #  sort the tensor eig_value in descending order and also get the corresponding indices of the sorted elements
    #
    pi = left_vector[:,ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]    # X+1 back to X  # remove the dummy node
    p_ppr = p_dense.to(device)
    pi = pi/pi.sum()  # norm pi
    #
    # # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0
    pi_inv_sqrt = pi.pow(-0.5)      # (183,) to (183,)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag().to(device)     # (183,) to (183, 183)
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag().to(device)

    # L_appr   # actually, L_appr= I-L, so this L is the equivalent their version of symmetric_A of digraph
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0       # a bit time consuming
    L[torch.isnan(L)] = 0    # make nan to 0
    # L[torch.isnan(L)] = 1  # make nan to 1   # TODO delete it after testing(Qin use 1, original is 0)---worse

    # L = (p_ppr + p_ppr.t()) / 2.0       # TODO delete it after testing

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()     # the indices of all nonzero elements in the input tensor L, arranged as a tensor where each column represents the indices of a nonzero element
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices      # their transformed edges of this symmetric_A of digraph
    edge_weight = L_values

    # edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    # edge_weight= torch.rand(edge_index.size(1), dtype=dtype, device=edge_index.device) * (0.1 - 0.00001) + 0.0001   # TODO delete this just for test
    # perm = torch.randperm(edge_weight.size(0), device=edge_weight.device)
    # edge_weight = edge_weight[perm]

    # edge_weight = torch.abs(torch.sin(edge_weight))     # TODO delete
    min_val = torch.min(edge_weight).item()
    max_val = torch.max(edge_weight).item()

    print(f"Original Edge weight range: [{min_val}, {max_val}]")



    # row normalization
    if args.inci_norm is not None:
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col].to(device)

        min_val = torch.min(edge_weight).item()
        max_val = torch.max(edge_weight).item()

        print(f"Normalized Edge weight range: [{min_val}, {max_val}]")
    # plt.hist(edge_weight.cpu(), bins=50, edgecolor='k')
    # plt.xlabel('Absolute Edge Weight')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of  DiG edge weights_F1=()')  # Shuffled Absolute Value-Transformed Edge Weights
    # plt.show()

    # delete TODO
    # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    # edge_weight = torch.cat([edge_weight, torch.ones((num_nodes,), device=edge_weight.device)])
    # edge_index = edge_index.to(device)
    # edge_weight = edge_weight.to(device)

    return edge_index, edge_weight


def trimodal_distribution(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 100.0, 1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001, 0.1, 100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(3, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples


def trimodal_distribution4(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001, 0.01, 900, 5000.0], device=device, dtype=dtype)
    stds = torch.tensor([10, 100, 1000, 5000], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(4, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples


def trimodal_distribution2(size, device, dtype):
    # Define the means and standard deviations for our three peaks
    means = torch.tensor([0.001,  1000.0], device=device, dtype=dtype)
    stds = torch.tensor([0.0001,  100.0], device=device, dtype=dtype)

    # Create a categorical distribution to choose between the three peaks
    mix = dist.Categorical(torch.ones(2, device=device))

    # Create three normal distributions
    comp = dist.Normal(means, stds)

    # Create a mixture of these distributions
    gmm = dist.MixtureSameFamily(mix, comp)

    # Sample from the mixture
    samples = gmm.sample((size,))

    # Clip the values to ensure they're within [0.0001, 10000]
    samples = torch.clamp(samples, min=0.0001, max=10000)

    return samples


def intersect_sparse_tensors_noDense(A_in, A_out):
    device = A_in.device

    try:
        indices_in = A_in.indices()
        indices_out = A_out.indices()

        A_in = A_in.to_dense()
        A_out = A_out.to_dense()

        A_in_hat = A_in.to(device)
        A_out_hat = A_out.to(device)

        A_in_hat[A_out == 0] = 0  # intersection
        A_out_hat[A_in == 0] = 0

        # L^{(2)}
        # intersection = (A_in_hat + A_out_hat) / 2.0
        intersection = A_in_hat

        indices = intersection.nonzero().t()
        values = torch.ones(indices.size(1), dtype=torch.float32).to(device)

        num_intersecting_edges = torch.count_nonzero(intersection)
        # print('!!!!',  time.time())

        return torch.sparse_coo_tensor(indices, values, size=intersection.size(), dtype=torch.float32)
    except:
        # Get indices and values for A_in and A_out
        indices_in = A_in.indices()
        indices_out = A_out.indices()

        # Create sets of tuples for the indices of A_in and A_out
        set_in = set(map(tuple, indices_in.t().tolist()))
        set_out = set(map(tuple, indices_out.t().tolist()))

        # Find the intersection of these sets
        intersect_indices = list(set_in & set_out)

        if len(intersect_indices) == 0:
            return torch.sparse_coo_tensor([], [], size=A_in.size(), dtype=torch.float32, device=device)

        # Convert intersecting indices back to a tensor
        intersect_indices = torch.tensor(intersect_indices, dtype=torch.long, device=device).t()

        # Create the intersection sparse tensor
        values = torch.ones(intersect_indices.size(1), dtype=torch.float32, device=device)

        # Number of intersecting edges
        num_intersecting_edges = intersect_indices.size(1)

        return torch.sparse_coo_tensor(intersect_indices, values, size=A_in.size(), dtype=torch.float32).coalesce()


def sparse_mm_chunked(A, B, chunk_size):
    """
    Perform sparse matrix multiplication in chunks to manage memory usage.
    """
    A = A.coalesce()
    B = B.coalesce()

    A_indices = A.indices()
    A_values = A.values()
    B_indices = B.indices()
    B_values = B.values()

    result_indices = []
    result_values = []

    num_chunks = (A.size(1) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, A.size(1))

        A_chunk = torch.sparse_coo_tensor(
            A_indices[:, (A_indices[1] >= start) & (A_indices[1] < end)],
            A_values[(A_indices[1] >= start) & (A_indices[1] < end)],
            (A.size(0), end - start)
        )

        B_chunk = torch.sparse_coo_tensor(
            B_indices[:, (B_indices[0] >= start) & (B_indices[0] < end)],
            B_values[(B_indices[0] >= start) & (B_indices[0] < end)],
            (end - start, B.size(1))
        )

        AB_chunk = torch.sparse.mm(A_chunk.to_dense(), B_chunk.to_dense()).to_sparse()

        result_indices.append(AB_chunk.indices())
        result_values.append(AB_chunk.values())

    result_indices = torch.cat(result_indices, dim=1)
    result_values = torch.cat(result_values)
    result = torch.sparse_coo_tensor(result_indices, result_values, (A.size(0), B.size(1))).coalesce()

    return result


def sparse_mm_safe(A, B):
    try:
        return torch.sparse.mm(A, B)
    except RuntimeError as e:
        if "CUDA error: insufficient resources" in str(e):
            print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
            return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
        else:
            raise e


def generate_possible_B_products(A, m):
    # List of matrices to be used in combinations (A and A transpose)
    elements = [A, A.t()]

    # Generate all possible combinations of A and A transpose of length k
    all_combinations = list(itertools.product(elements, repeat=m))

    # Compute the product for each combination
    results1 = []
    results2 = []
    for combination in all_combinations:
        B = combination[0]
        for mat in combination[1:]:
            B = sparse_mm_safe(B, mat)
        B1 = sparse_mm_safe(A, B)        # make sure the first is A
        B2 = sparse_mm_safe(A.t(), B)  # make sure the first is A.t()
        results1.append(sparse_mm_safe(B1, B1.t()))
        results2.append(sparse_mm_safe(B2, B2.t()))

    return [results1, results2]


def sparese_remove_self_loops(sparse_matrix):
    # Ensure the sparse matrix is in coalesced format (no duplicate entries)
    sparse_matrix = sparse_matrix.coalesce()

    # Create a mask to filter out diagonal elements (self-loops)
    mask = sparse_matrix.indices()[0] != sparse_matrix.indices()[1]

    # Apply the mask and create a new sparse tensor without self-loops
    return torch.sparse.FloatTensor(
        sparse_matrix.indices()[:, mask],
        sparse_matrix.values()[mask],
        sparse_matrix.size()
    )


def sparse_boolean_multi_hopExhaust(args, A, k, mode='union'):
    selfloop = args.rm_gen_sloop

    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    A_out = sparse_mm_safe(A.t(), A)
    # num_nonzero_in = A_in._nnz()
    # num_nonzero_out = A_out._nnz()
    # print('number of edges:', num_nonzero_in, num_nonzero_out)

    if mode == 'union':
        A_result = A_in + A_out
        A_result = A_result.coalesce()
        A_result._values().clamp_(0, 1)  # Ensuring binary values
    else:   # intersection
        A_result = intersect_sparse_tensors_noDense(A_in, A_out)

    if selfloop == -1:
        A_result = sparese_remove_self_loops(A_result)
    all_hops = [A_result]

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        [in_list, out_list] = generate_possible_B_products(A, hop)
        for A_in, A_out in zip(in_list, out_list):
            if mode == 'union':
                A_result = A_in + A_out
                A_result = A_result.coalesce()
                A_result._values().clamp_(0, 1)  # Ensuring binary values
            else:
                A_result = intersect_sparse_tensors_noDense(A_in, A_out)

            # num_nonzero_result = A_result._nnz()
            # print('num of edges:', num_nonzero_result)
            if selfloop == -1:
                A_result = sparese_remove_self_loops(A_result)
            all_hops.append(A_result)

    return tuple(all_hops)


def sparse_boolean_multi_hop_DirGNN(has_1_order, rm_gen_self_loop, A, k):
    order_tuple_list = []

    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    order_tuple_0 = [A, A.t()]

    if k<1:
        return tuple(order_tuple_0)

    if has_1_order:
        order_tuple_list.append(order_tuple_0)

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A)
    # A_out = sparse_mm_safe(A.t(), A.t())
    A_out = A_in.t()

    B_in = sparse_mm_safe(A, A.t())
    # B_out = sparse_mm_safe(A, A.t())
    B_out = B_in.t()
    num_nonzero_in = A_in._nnz()
    num_nonzero_out = B_in._nnz()
    print('number of edges:', num_nonzero_in, num_nonzero_out)

    if rm_gen_self_loop == -1:
        order_tuple_1 = [sparse_remove_self_loops(A_in), sparse_remove_self_loops(B_in), sparse_remove_self_loops(A_out), sparse_remove_self_loops(B_out)]
    else:
        order_tuple_1 = [A_in, B_in, A_out, B_out]
    order_tuple_list.append(order_tuple_1)

    for hop in range(1, k):
        order_tuple_temp = []
        for edge_matrix in order_tuple_list[-1]:        # TODO : might improve efficiency for symmetry
            N_in = sparse_mm_safe(edge_matrix, A)
            N_out = sparse_mm_safe(edge_matrix, A.t())
            if rm_gen_self_loop == -1:
                order_tuple_temp.extend([sparse_remove_self_loops(N_in), sparse_remove_self_loops(N_out)])
            else:
                order_tuple_temp.extend([N_in, N_out])
        order_tuple_list.append(order_tuple_temp)

    return tuple(tensor for sub_list in order_tuple_list for tensor in sub_list)


def sparse_remove_self_loops(matrix):
    # Function to remove self-loops by setting diagonal elements to zero.
    # Extract the indices and values of the sparse matrix
    indices = matrix._indices()
    values = matrix._values()

    # Filter out diagonal elements (self-loops)
    mask = indices[0] != indices[1]
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Create a new sparse matrix without self-loops
    new_matrix = torch.sparse_coo_tensor(new_indices, new_values, matrix.size(), dtype=matrix.dtype)
    new_matrix = new_matrix.coalesce()

    return new_matrix


def sparse_boolean_multi_hop(args, A, k, mode='union'):
    selfloop = args.rm_gen_sloop
    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    def sparse_mm_safe(A, B):
        try:
            return torch.sparse.mm(A, B)
        except RuntimeError as e:
            # if "CUDA error: insufficient resources" in str(e):
            if "CUDA out of memory" in str(e) or "CUDA error: insufficient resources" in str(e):
                try:
                    print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
                    A_cpu = A.to(torch.device("cpu"))
                    B_cpu = B.to(torch.device("cpu"))
                    return torch.sparse.mm(A_cpu, B_cpu).to(A.device)
                except:
                    print("CPU operation failed. Attempting chunked multiplication.")
                    return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
            else:
                raise e

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    A_out = sparse_mm_safe(A.t(), A)

    # A_in = sparse_mm_safe(A, A)
    # A_out = sparse_mm_safe(A.t(), A.t())
    if selfloop == -1:
        A_in = sparse_remove_self_loops(A_in)
        A_out = sparse_remove_self_loops(A_out)
    num_nonzero_in = A_in._nnz()
    num_nonzero_out = A_out._nnz()
    print('2-order number of edges(in, out):', num_nonzero_in, num_nonzero_out)

    if mode == 'union':
        A_result = A_in + A_out
        A_result = A_result.coalesce()
        A_result._values().clamp_(0, 1)  # Ensuring binary values
        if selfloop == -1:
            A_result = sparse_remove_self_loops(A_result)
        all_hops = [A_result]
    elif mode == 'intersection':
        A_result = intersect_sparse_tensors_noDense(A_in, A_out)
        if selfloop == -1:
            A_result = sparse_remove_self_loops(A_result)
        all_hops = [A_result]
    elif mode == 'separate':
        if selfloop == -1:
            A_in = sparse_remove_self_loops(A_in)
            A_out = sparse_remove_self_loops(A_out)
        all_hops = [A_in, A_out]
    else:
        raise NotImplementedError("Not Implemented mode: ", mode)

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        A_in = sparse_mm_safe(A, A_in)
        A_in = sparse_mm_safe(A_in, A.t())
        A_out = sparse_mm_safe(A.t(), A_out)
        A_out = sparse_mm_safe(A_out, A)

        if selfloop == -1:
            A_in = sparse_remove_self_loops(A_in)
            A_out = sparse_remove_self_loops(A_out)

        num_nonzero_in = A_in._nnz()
        num_nonzero_out = A_out._nnz()
        print(hop + 2, 'order num of edges(in, out): ', num_nonzero_in, num_nonzero_out)

        if mode == 'union':
            A_result = A_in + A_out
            A_result = A_result.coalesce()
            A_result._values().clamp_(0, 1)  # Ensuring binary values
            num_nonzero = A_result._nnz()
            print(hop + 2, 'order num of edges (union): ', num_nonzero)
            if selfloop == -1:
                A_result = sparse_remove_self_loops(A_result)
            all_hops.append(A_result)
        elif mode == 'intersection':
            A_result = intersect_sparse_tensors_noDense(A_in, A_out)
            num_nonzero = A_result._nnz()
            print(hop + 2, 'order num of edges (intersection): ', num_nonzero)
            if selfloop == -1:
                A_result = sparse_remove_self_loops(A_result)
            all_hops.append(A_result)
        elif mode == 'separate':
            if selfloop == -1:
                A_in = sparse_remove_self_loops(A_in)
                A_out = sparse_remove_self_loops(A_out)
            all_hops.extend([A_in, A_out])
        else:
            raise NotImplementedError("Not Implemented mode: ", mode)

        # num_nonzero_result = A_result._nnz()
        # print('num of edges:', num_nonzero_result)


    return tuple(all_hops)


def OneDirect_sparse_boolean_multi_hop(A, k):
    # Ensure A is in canonical form
    A = A.coalesce().to(torch.float32)

    def sparse_mm_safe(A, B):
        try:
            return torch.sparse.mm(A, B)
        except RuntimeError as e:
            if "CUDA error: insufficient resources" in str(e):
                print("Switching to CPU for sparse matrix multiplication due to insufficient GPU resources.")
                return sparse_mm_chunked(A, B, chunk_size=1000).to(A.device)
            else:
                raise e

    # Initialize all_hops list with the intersection of A*A.T and A.T*A
    A_in = sparse_mm_safe(A, A.t())
    # A_out = sparse_mm_safe(A.t(), A)
    num_nonzero_in = A_in._nnz()
    # num_nonzero_out = A_out._nnz()
    print('number of edges:', num_nonzero_in)

    all_hops = [A_in]

    # Compute k-hop neighbors using sparse matrix multiplication and intersections
    for hop in range(1, k):
        A_in = torch.sparse.mm(A, A_in)
        A_in = torch.sparse.mm(A_in, A.t())
        # A_out = torch.sparse.mm(A.t(), A_out)
        # A_out = torch.sparse.mm(A_out, A)

        num_nonzero_in = A_in._nnz()
        # num_nonzero_out = A_out._nnz()
        print(hop + 2, 'order num of edges: ', num_nonzero_in)

        # if mode == 'union':
        #     A_result = A_in + A_out
        #     A_result = A_result.coalesce()
        #     A_result._values().clamp_(0, 1)  # Ensuring binary values
        # else:
        #     A_result = intersect_sparse_tensors(A_in, A_out)

        num_nonzero_result = A_in._nnz()
        print('num of edges:', num_nonzero_result)
        all_hops.append(A_in)

    return tuple(all_hops)


def normalize_row_edges(edge_index, num_nodes, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float, device=device)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # deg = torch.where(deg == 0, torch.tensor(float('inf'), device=deg.device), deg)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def sparse_difference(U, I, epsilon=1e-8):
    diff = (U - I).coalesce()
    mask = diff._values().abs() > epsilon
    return torch.sparse_coo_tensor(
        diff._indices()[:, mask],
        diff._values()[mask],
        diff.size()
    )


def Qin_get_second_directed_adj(args, edge_index, num_nodes, k, IsExhaustive, mode, norm='dir'):     #
    device = edge_index.device
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    if mode != 'independent':
        if IsExhaustive:
            L_tuple = sparse_boolean_multi_hopExhaust(args, A, k - 1, mode)  # much slower
        else:
            L_tuple = sparse_boolean_multi_hop(args, A, k-1, mode)   # much slower
    else:       # independent
        if IsExhaustive:
            L_tupleU = sparse_boolean_multi_hopExhaust(args, A, k - 1, 'union')  # much slower
            L_tupleI = sparse_boolean_multi_hopExhaust(args, A, k - 1, 'intersection')  # much slower
        else:
            L_tupleU = sparse_boolean_multi_hop(args, A, k-1, 'union')   # much slower
            L_tupleI = sparse_boolean_multi_hop(args, A, k-1, 'intersection')   # much slower
        all_hops = list(L_tupleI)
        for U, I in zip(L_tupleU, L_tupleI):
            all_hops.append(sparse_difference(U, I))
        L_tuple = tuple(all_hops)

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        row, col = L._indices()
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        if adj_norm.storage.value() is None:
            all_hops_weight.append(torch.ones(row.size(0), dtype=torch.int).to(device))
        else:
            all_hops_weight.append(adj_norm.storage.value())

    return tuple(all_hop_edge_index), tuple(all_hops_weight)


def Qin_get_all_directed_adj(args,  edge_index, num_nodes, k, IsExhaustive, mode, norm='dir'):
    has_1_order = args.has_1_order
    selfloop = args.add_selfloop
    rm_gen_sloop = args.rm_gen_sloop

    device = edge_index.device
    if selfloop == 1:
        edge_index, _ = add_self_loops(edge_index.long(), fill_value=1, num_nodes=num_nodes)       #
    elif selfloop == -1:
        edge_index, _ = remove_self_loops(edge_index)
    else:
        pass
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    L_tuple = sparse_boolean_multi_hop_DirGNN(has_1_order, rm_gen_sloop, A, k - 1)  # much slower

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        row, col = L._indices()
        adj_norm = get_norm_adj(SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)), norm=norm).coalesce()
        all_hop_edge_index.append(torch.stack(adj_norm.coo()[:2]))
        all_hops_weight.append(adj_norm.storage.value())

    return tuple(all_hop_edge_index), tuple(all_hops_weight)


def Qin_get_second_adj(edge_index, num_nodes, dtype, k):     #
    device = edge_index.device
    fill_value = 1
    # edge_index, _ = add_self_loops(edge_index.long(), fill_value=fill_value, num_nodes=num_nodes)       # TODO add back after no-selfloop test
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = edge_index.to(device)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.bool).to(device)
    A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).to(device)
    L_tuple = OneDirect_sparse_boolean_multi_hop(A, k-1)   # much slower

    all_hop_edge_index = []
    all_hops_weight = []
    for L in L_tuple:  # Skip L1 if not needed
        edge_indexL = L._indices()
        all_hop_edge_index.append(edge_indexL)
        edge_weightL = normalize_row_edges(edge_indexL, num_nodes).to(device)
        all_hops_weight.append(edge_weightL)

    return tuple(all_hop_edge_index), tuple(all_hops_weight)

@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def to_undirected(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    return edge_index, edge_weight


def to_undirectedBen(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    # print(edge_index, edge_index.shape)

    edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1])]

    set_edges = set(edges)

    unique_edges = list(set_edges)

    num_list = [0] * len(edges)
    history = []
    count = 0
    for i in range(len(edges)):
        if edges[i] in history:
            num_list[i] += 1
            count += 1
            # print("Duplicate: ", edges[i], num_list[i], count)
        else:
            history.append(edges[i])

    edge_index0 = [i[0] for i in unique_edges]
    edge_index1 = [i[1] for i in unique_edges]
    edge_index = torch.tensor([edge_index0, edge_index1])
    # print(edge_index, edge_index.shape)

    return edge_index



