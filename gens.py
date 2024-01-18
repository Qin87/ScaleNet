import math

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch

def sampling_idx_individual_dst(class_num_list, idx_info, device):
    """
    to get the source nodes
    :param class_num_list:
    :param idx_info: # all train nodes for each class
    :param device:
    :return:
    """
    # print("Warmup: ", class_num_list, np.shape(class_num_list))  #  [14, 0, 1, 45, 4] (5,)
    # Selecting src & dst nodes
    max_num, n_cls = max(class_num_list), len(class_num_list)
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # print(sampling_list)    # tensor([307., 334., 183.,   0., 268., 326., 338.])
    new_class_num_list = torch.Tensor(class_num_list).to(device)
    # print("new : ", new_class_num_list)   # tensor([ 34.,   7., 158., 341.,  73.,  15.,   3.])

    # Compute # of source nodes
    # print("samp: ", sampling_list)   # tensor([34., 48., 48.,  0., 45.])

    sampling_dst_idx = []
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    for cls_idx, samp_num in zip(idx_info, sampling_list):
        samp_num = int(samp_num.item())
        if samp_num <= 0:
            continue
        if len(cls_idx) <=0:
            continue
        # Sampling indices for dst
        # print("What::", samp_num, len(cls_idx))   # What:: 45 0
        dst_idx_local = cls_idx[torch.randint(len(cls_idx), (samp_num,))]
        sampling_dst_idx.append(dst_idx_local)
    # Concatenate the sampled indices
    sampling_dst_idx = torch.cat(sampling_dst_idx)
    # Sample indices for src
    src_idx = torch.multinomial(prob, sampling_dst_idx.shape[0], True)
    sampling_src_idx = temp_idx_info[src_idx]
    # print("\nChatGPT samp_src_idx", sampling_src_idx.shape)

    # sampling_src_idx = [cls_idx[torch.randint(len(cls_idx), (int(samp_num.item()),))]
    #                     for cls_idx, samp_num in zip(idx_info, sampling_list)]   # this code err when sam_num, or cls_idx is 0
    # sampling_src_idx = torch.cat(sampling_src_idx)  # this means 7 combines into one tensor
    # print("\noriginal samp_src_idx", sampling_src_idx.shape)
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()   # why use this as prob?
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]

    # Sorting src idx with corresponding dst idx
    # the first is ascending ordered new tensor, the second is the original index
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]
    # print(sampling_src_idx, sampling_dst_idx)

    return sampling_src_idx, sampling_dst_idx

@torch.no_grad()
# def sampling_idx_individual_dst(class_num_list, idx_info, device):
#     # Selecting src & dst nodes
#     max_num, n_cls = max(class_num_list), len(class_num_list)
#     sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
#     new_class_num_list = torch.Tensor(class_num_list).to(device)
#
#     # Compute # of source nodes
#     sampling_src_idx =[cls_idx[torch.randint(len(cls_idx),(int(samp_num.item()),))]
#                         for cls_idx, samp_num in zip(idx_info, sampling_list)]
#     sampling_src_idx = torch.cat(sampling_src_idx)
#
#     # Generate corresponding destination nodes
#     prob = torch.log(new_class_num_list.float())/ new_class_num_list.float()
#     prob = prob.repeat_interleave(new_class_num_list.long())
#     temp_idx_info = torch.cat(idx_info)
#     dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
#     sampling_dst_idx = temp_idx_info[dst_idx]
#
#     # Sorting src idx with corresponding dst idx
#     sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
#     sampling_dst_idx = sampling_dst_idx[sorted_idx]
#
#     return sampling_src_idx, sampling_dst_idx

def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x

@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    device = edge_index.device

    # Assign node index for augmented nodes
    row, col = edge_index[0], edge_index[1] 
    row, sort_idx = torch.sort(row)
    col = col[sort_idx] 
    degree = scatter_add(torch.ones_like(row), row)
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    # Duplicate the edges of source nodes
    node_mask = torch.zeros(total_node, dtype=torch.bool)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True 
    row_mask = node_mask[row] 
    edge_mask = col[row_mask] 
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


def neighbor_sampling_reverse(total_node, edge_index, sampling_src_idx,
                      neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node - len(degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(
        torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)  # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    # inv_edge_index = torch.stack([new_col, new_row], dim=0)
    inv_edge_index = torch.stack([new_row, new_col], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_BiEdge(total_node, edge_index, sampling_src_idx,
                      neighbor_dist_list, train_node_mask=None):
    """
    add inverse edges based on GraphSHA
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)

    # Find the nearest nodes and mix target pool
    # mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx.long()]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node - len(degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(
        torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1)  # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    inv_edge_index_inverse = torch.stack([new_row, new_col], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index, inv_edge_index_inverse], dim=1)

    return new_edge_index

def neighbor_sampling_bidegreeOrigin(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in row and col.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    # sampling_src_idx = sampling_src_idx.clone().to(device)
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute col_degree
    col = edge_index[1]
    row = edge_index[0]
    col_degree = scatter_add(torch.ones_like(col), col)  # Ben only col col_degree (count in the end of edge)
    row_degree = scatter_add(torch.ones_like(row), row)  # Ben only col col_degree   ( count in the source of edge)
    # print(col_degree)

    if len(col_degree) < total_node:
        col_degree = torch.cat([col_degree, col_degree.new_zeros(total_node - len(col_degree))], dim=0)
    if len(row_degree) < total_node:
        row_degree = torch.cat([row_degree, row_degree.new_zeros(total_node - len(row_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(col_degree,
                                          dtype=torch.bool)  # the same shape as the col_degree tensor, and all elements in the mask are set to True.
    col_degree_dist = scatter_add(torch.ones_like(col_degree[train_node_mask]), col_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    row_degree_dist = scatter_add(torch.ones_like(row_degree[train_node_mask]), row_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample col_degree for augmented nodes
    col_prob = col_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    row_prob = row_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    col_aug_degree = torch.multinomial(col_prob, 1).to(device).squeeze(dim=1)  # (m)
    row_aug_degree = torch.multinomial(row_prob, 1).to(device).squeeze(dim=1)  # (m)
    col_max_degree = col_degree.max().item() + 1
    row_max_degree = row_degree.max().item() + 1
    col_aug_degree = torch.min(col_aug_degree, col_degree[sampling_src_idx])
    row_aug_degree = torch.min(row_aug_degree, row_degree[sampling_src_idx])

    # Sample neighbors
    col_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, col_max_degree)
    row_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, row_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', col_new_tgt)
    col_tgt_index = torch.arange(col_max_degree).unsqueeze(dim=0).to(device)
    row_tgt_index = torch.arange(row_max_degree).unsqueeze(dim=0).to(device)
    col_new_col = col_new_tgt[(col_tgt_index - col_aug_degree.unsqueeze(dim=1) < 0)]
    row_new_row = row_new_tgt[(row_tgt_index - row_aug_degree.unsqueeze(dim=1) < 0)]
    col_new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    row_new_col = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    col_new_row = col_new_row.repeat_interleave(col_aug_degree)
    row_new_col = row_new_col.repeat_interleave(row_aug_degree)
    col_inv_edge_index = torch.stack([col_new_col, col_new_row], dim=0)
    row_inv_edge_index = torch.stack([row_new_col, row_new_row], dim=0)
    # col_inv_edge_index = torch.stack([col_new_row, col_new_col], dim=0)  # Ben change direction
    # row_inv_edge_index = torch.stack([row_new_row, row_new_col], dim=0)
    new_edge_index = torch.cat([edge_index, col_inv_edge_index, row_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in row and col.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute col_degree
    col = edge_index[1]
    row = edge_index[0]
    col_degree = scatter_add(torch.ones_like(col), col)  # Ben only col col_degree
    row_degree = scatter_add(torch.ones_like(row), row)  # Ben only col col_degree

    if len(col_degree) < total_node:
        col_degree = torch.cat([col_degree, col_degree.new_zeros(total_node - len(col_degree))], dim=0)
    if len(row_degree) < total_node:
        row_degree = torch.cat([row_degree, row_degree.new_zeros(total_node - len(row_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(col_degree,
                                          dtype=torch.bool)  # the same shape as the col_degree tensor, and all elements in the mask are set to True.
    col_degree_dist = scatter_add(torch.ones_like(col_degree[train_node_mask]), col_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    row_degree_dist = scatter_add(torch.ones_like(row_degree[train_node_mask]), row_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample col_degree for augmented nodes
    col_prob = col_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    row_prob = row_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    col_aug_degree = torch.multinomial(col_prob, 1).to(device).squeeze(dim=1)  # (m)
    row_aug_degree = torch.multinomial(row_prob, 1).to(device).squeeze(dim=1)  # (m)
    col_max_degree = col_degree.max().item() + 1
    row_max_degree = row_degree.max().item() + 1
    col_aug_degree = torch.min(col_aug_degree, col_degree[sampling_src_idx])
    row_aug_degree = torch.min(row_aug_degree, row_degree[sampling_src_idx])

    # Sample neighbors
    col_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, col_max_degree)
    row_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, row_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', col_new_tgt)
    col_tgt_index = torch.arange(col_max_degree).unsqueeze(dim=0).to(device)
    row_tgt_index = torch.arange(row_max_degree).unsqueeze(dim=0).to(device)
    col_new_col = col_new_tgt[(col_tgt_index - col_aug_degree.unsqueeze(dim=1) < 0)]
    row_new_row = row_new_tgt[(row_tgt_index - row_aug_degree.unsqueeze(dim=1) < 0)]
    col_new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    row_new_col = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    col_new_row = col_new_row.repeat_interleave(col_aug_degree)
    row_new_col = row_new_col.repeat_interleave(row_aug_degree)
    # col_inv_edge_index = torch.stack([col_new_col, col_new_row], dim=0)
    # row_inv_edge_index = torch.stack([row_new_col, row_new_row], dim=0)
    col_inv_edge_index = torch.stack([col_new_row, col_new_col], dim=0)  # Ben change direction
    row_inv_edge_index = torch.stack([row_new_row, row_new_col], dim=0)
    new_edge_index = torch.cat([edge_index, col_inv_edge_index, row_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_variant1(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in row and col.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute col_degree
    col = edge_index[1]
    row = edge_index[0]
    col_degree = scatter_add(torch.ones_like(col), col)  # Ben only col col_degree
    row_degree = scatter_add(torch.ones_like(row), row)  # Ben only col col_degree

    if len(col_degree) < total_node:
        col_degree = torch.cat([col_degree, col_degree.new_zeros(total_node - len(col_degree))], dim=0)
    if len(row_degree) < total_node:
        row_degree = torch.cat([row_degree, row_degree.new_zeros(total_node - len(row_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(col_degree,
                                          dtype=torch.bool)  # the same shape as the col_degree tensor, and all elements in the mask are set to True.
    col_degree_dist = scatter_add(torch.ones_like(col_degree[train_node_mask]), col_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    row_degree_dist = scatter_add(torch.ones_like(row_degree[train_node_mask]), row_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample col_degree for augmented nodes
    col_prob = col_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    row_prob = row_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    col_aug_degree = torch.multinomial(col_prob, 1).to(device).squeeze(dim=1)  # (m)
    row_aug_degree = torch.multinomial(row_prob, 1).to(device).squeeze(dim=1)  # (m)
    col_max_degree = col_degree.max().item() + 1
    row_max_degree = row_degree.max().item() + 1
    col_aug_degree = torch.min(col_aug_degree, col_degree[sampling_src_idx])
    row_aug_degree = torch.min(row_aug_degree, row_degree[sampling_src_idx])

    # Sample neighbors
    col_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, col_max_degree)
    row_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, row_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', col_new_tgt)
    col_tgt_index = torch.arange(col_max_degree).unsqueeze(dim=0).to(device)
    row_tgt_index = torch.arange(row_max_degree).unsqueeze(dim=0).to(device)
    col_new_col = col_new_tgt[(col_tgt_index - col_aug_degree.unsqueeze(dim=1) < 0)]
    row_new_row = row_new_tgt[(row_tgt_index - row_aug_degree.unsqueeze(dim=1) < 0)]
    col_new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    row_new_col = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    col_new_row = col_new_row.repeat_interleave(col_aug_degree)
    row_new_col = row_new_col.repeat_interleave(row_aug_degree)
    # col_inv_edge_index = torch.stack([col_new_col, col_new_row], dim=0)
    row_inv_edge_index = torch.stack([row_new_col, row_new_row], dim=0)
    col_inv_edge_index = torch.stack([col_new_row, col_new_col], dim=0)  # Ben change direction
    # row_inv_edge_index = torch.stack([row_new_row, row_new_col], dim=0)
    new_edge_index = torch.cat([edge_index, col_inv_edge_index, row_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_bidegree_variant2(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in row and col.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]
    # print(neighbor_dist_list)

    # Compute col_degree
    col = edge_index[1]
    row = edge_index[0]
    col_degree = scatter_add(torch.ones_like(col), col)  # Ben only col col_degree
    row_degree = scatter_add(torch.ones_like(row), row)  # Ben only col col_degree

    if len(col_degree) < total_node:
        col_degree = torch.cat([col_degree, col_degree.new_zeros(total_node - len(col_degree))], dim=0)
    if len(row_degree) < total_node:
        row_degree = torch.cat([row_degree, row_degree.new_zeros(total_node - len(row_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(col_degree,
                                          dtype=torch.bool)  # the same shape as the col_degree tensor, and all elements in the mask are set to True.
    col_degree_dist = scatter_add(torch.ones_like(col_degree[train_node_mask]), col_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    row_degree_dist = scatter_add(torch.ones_like(row_degree[train_node_mask]), row_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample col_degree for augmented nodes
    col_prob = col_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    row_prob = row_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    col_aug_degree = torch.multinomial(col_prob, 1).to(device).squeeze(dim=1)  # (m)
    row_aug_degree = torch.multinomial(row_prob, 1).to(device).squeeze(dim=1)  # (m)
    col_max_degree = col_degree.max().item() + 1
    row_max_degree = row_degree.max().item() + 1
    col_aug_degree = torch.min(col_aug_degree, col_degree[sampling_src_idx])
    row_aug_degree = torch.min(row_aug_degree, row_degree[sampling_src_idx])

    # Sample neighbors
    col_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, col_max_degree)
    row_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, row_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', col_new_tgt)
    col_tgt_index = torch.arange(col_max_degree).unsqueeze(dim=0).to(device)
    row_tgt_index = torch.arange(row_max_degree).unsqueeze(dim=0).to(device)
    col_new_col = col_new_tgt[(col_tgt_index - col_aug_degree.unsqueeze(dim=1) < 0)]
    row_new_row = row_new_tgt[(row_tgt_index - row_aug_degree.unsqueeze(dim=1) < 0)]
    col_new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    row_new_col = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    col_new_row = col_new_row.repeat_interleave(col_aug_degree)
    row_new_col = row_new_col.repeat_interleave(row_aug_degree)
    col_inv_edge_index = torch.stack([col_new_col, col_new_row], dim=0)
    # row_inv_edge_index = torch.stack([row_new_col, row_new_row], dim=0)
    # col_inv_edge_index = torch.stack([col_new_row, col_new_col], dim=0)  # Ben change direction
    row_inv_edge_index = torch.stack([row_new_row, row_new_col], dim=0)
    new_edge_index = torch.cat([edge_index, col_inv_edge_index, row_inv_edge_index], dim=1)

    return new_edge_index

def neighbor_sampling_BiEdge_bidegree(total_node, edge_index, sampling_src_idx,
                               neighbor_dist_list, train_node_mask=None):
    """
    two degrees in row and col.
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        the sources nodes has no direction
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        may well no direction
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device).to(torch.long)

    # Find the nearest nodes and mix target pool
    sampling_src_idx = torch.tensor(sampling_src_idx, dtype=torch.long)  # Adjust dtype as needed
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # print(neighbor_dist_list)

    # Compute col_degree
    col = edge_index[1]
    row = edge_index[0]
    col_degree = scatter_add(torch.ones_like(col), col)  # Ben only col col_degree
    row_degree = scatter_add(torch.ones_like(row), row)  # Ben only col col_degree

    if len(col_degree) < total_node:
        col_degree = torch.cat([col_degree, col_degree.new_zeros(total_node - len(col_degree))], dim=0)
    if len(row_degree) < total_node:
        row_degree = torch.cat([row_degree, row_degree.new_zeros(total_node - len(row_degree))], dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(col_degree,
                                          dtype=torch.bool)  # the same shape as the col_degree tensor, and all elements in the mask are set to True.
    # print("col degree\n", col_degree, "\nrow degree\n", row_degree, row_degree.shape)
    # if torch.equal(col_degree, row_degree):
    #     print("the same")
    # else:
    #     print("different")    # here
    col_degree_dist = scatter_add(torch.ones_like(col_degree[train_node_mask]), col_degree[train_node_mask]).to(
        device).type(
        torch.float32)
    row_degree_dist = scatter_add(torch.ones_like(row_degree[train_node_mask]), row_degree[train_node_mask]).to(
        device).type(torch.float32)

    # Sample col_degree for augmented nodes
    col_prob = col_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    row_prob = row_degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
    col_aug_degree = torch.multinomial(col_prob, 1).to(device).squeeze(dim=1)  # (m)
    row_aug_degree = torch.multinomial(row_prob, 1).to(device).squeeze(dim=1)  # (m)
    col_max_degree = col_degree.max().item() + 1
    row_max_degree = row_degree.max().item() + 1
    col_aug_degree = torch.min(col_aug_degree, col_degree[sampling_src_idx])
    row_aug_degree = torch.min(row_aug_degree, row_degree[sampling_src_idx])

    # Sample neighbors
    col_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, col_max_degree)
    row_new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, row_max_degree)
    # print("hhh", mixed_neighbor_dist, 'eewe', col_new_tgt)
    col_tgt_index = torch.arange(col_max_degree).unsqueeze(dim=0).to(device)
    row_tgt_index = torch.arange(row_max_degree).unsqueeze(dim=0).to(device)
    col_new_col = col_new_tgt[(col_tgt_index - col_aug_degree.unsqueeze(dim=1) < 0)]
    row_new_row = row_new_tgt[(row_tgt_index - row_aug_degree.unsqueeze(dim=1) < 0)]
    col_new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    row_new_col = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
    col_new_row = col_new_row.repeat_interleave(col_aug_degree)
    row_new_col = row_new_col.repeat_interleave(row_aug_degree)
    col_inv_edge_index = torch.stack([col_new_col, col_new_row], dim=0)
    row_inv_edge_index = torch.stack([row_new_col, row_new_row], dim=0)
    col_inv_edge_index_inv = torch.stack([col_new_row, col_new_col], dim=0)  # Ben change direction
    row_inv_edge_index_inv = torch.stack([row_new_row, row_new_col], dim=0)
    new_edge_index = torch.cat([edge_index, col_inv_edge_index, row_inv_edge_index, col_inv_edge_index_inv, row_inv_edge_index_inv], dim=1)

    return new_edge_index

@torch.no_grad()
# def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False, no_mask=False):
#     max_num, n_cls = max(class_num_list), len(class_num_list)
#     if not max_flag: # mean
#         max_num = sum(class_num_list) / n_cls
#     sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
#
#     prev_out_local = F.softmax(prev_out_local/tau, dim=1)
#     prev_out_local = prev_out_local.cpu()
#
#     src_idx_all = []
#     dst_idx_all = []
#     for cls_idx, num in enumerate(sampling_list):
#         num = int(num.item())
#         if num <= 0:
#             continue
#
#         # first sampling
#         prob = 1 - prev_out_local[idx_info_local[cls_idx]][:,cls_idx].squeeze()
#         src_idx_local = torch.multinomial(prob + 1e-12, num, replacement=True)
#         src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]]
#
#         # second sampling
#         conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]]
#         if not no_mask:
#             conf_src[:,cls_idx] = 0
#         neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist()
#
#         # third sampling
#         neighbor = [prev_out_local[idx_info_local[cls]][:,cls_idx] for cls in neighbor_cls]
#         dst_idx = []
#         for i, item in enumerate(neighbor):
#             dst_idx_local = torch.multinomial(item + 1e-12, 1)[0]
#             dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
#         dst_idx = torch.tensor(dst_idx).to(src_idx.device)
#
#         src_idx_all.append(src_idx)
#         dst_idx_all.append(dst_idx)
#
#     src_idx_all = torch.cat(src_idx_all)
#     dst_idx_all = torch.cat(dst_idx_all)
#
#     return src_idx_all, dst_idx_all


def sampling_node_source(class_num_list, prev_out_local, idx_info_local, train_idx, tau=2, max_flag=False,
                         no_mask=False):
    """
    sampling for subsequent epochs
    :param class_num_list:
    :param prev_out_local: predicted label of train nodes
    :param idx_info_local:  # train nodes position inside train
    :param train_idx:
    :param tau:
    :param max_flag:
    :param no_mask:whether to mask the self class in sampling neighbor classes. default is mask
    :return:src_idx_all, dst_idx_all
    """
    max_num, n_cls = max(class_num_list), len(class_num_list)
    if not max_flag:  # mean
        max_num = math.ceil(sum(class_num_list) / n_cls)  # determined by args
    sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)
    # print("After warm", sampling_list)    # tensor([ 21.6667,  -2.3333,  -8.3333, -11.3333,  -3.3333,   3.6667])
    prev_out_local = F.softmax(prev_out_local / tau, dim=1)
    # softmax is to transform a vector of real numbers into a probability distribution.
    prev_out_local = prev_out_local.cpu()

    src_idx_all = []
    dst_idx_all = []
    # print("sampling_list: ", sampling_list)
    for cls_idx, num in enumerate(sampling_list):
        num = int(num.item())
        if num <= 0:
            continue
        # first sampling
        if idx_info_local[cls_idx].numel() == 0:
            continue
        # print(cls_idx)  # 1
        # print(idx_info_local[cls_idx])   # tensor([])
        prob = 1 - prev_out_local[idx_info_local[cls_idx].long()][:, cls_idx].squeeze()
        # print(prob)   # tensor(0.7912)
        if prob.shape == torch.Size([]):
            prob = torch.tensor([prob])
        src_idx_local = torch.multinomial(prob + 1e-12, num,
                                          replacement=True)  # the harder the sample, the more likely to be sampled
        src_idx = train_idx[idx_info_local[cls_idx][src_idx_local]]  # each minor class has src_idx

        # second sampling
        conf_src = prev_out_local[idx_info_local[cls_idx][src_idx_local]]
        if not no_mask:
            conf_src[:, cls_idx] = 0
        neighbor_cls = torch.multinomial(conf_src + 1e-12, 1).squeeze().tolist()

        # third sampling
        neighbor = [prev_out_local[idx_info_local[cls].long()][:, cls_idx] for cls in neighbor_cls if
                    idx_info_local[cls].numel() != 0]
        dst_idx = []
        new_src_idx = []
        for i, item in enumerate(neighbor):
            dst_idx_local = torch.multinomial(item + 1e-12, 1)[0]
            # dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])    # index 32 is out of bounds for dimension 0 with size 0
            # Check if idx_info_local[neighbor_cls[i]] has non-zero size along dimension 0
            if idx_info_local[neighbor_cls[i]].numel() != 0:
                # Check if dst_idx_local is within valid bounds
                if 0 <= dst_idx_local < idx_info_local[neighbor_cls[i]].numel():
                    dst_idx.append(train_idx[idx_info_local[neighbor_cls[i]][dst_idx_local]])
                    new_src_idx.append(src_idx[i])
                else:
                    # Handle the case where dst_idx_local is out of bounds
                    # You can print a warning or take appropriate action
                    pass
                    # print(f"Warning: dst_idx_local {dst_idx_local} is out of bounds.")

            else:
                # Handle the case where idx_info_local[neighbor_cls[i]] has size 0
                # You can print a warning or take appropriate action
                # print(f"Warning: idx_info_local[neighbor_cls[i]] has size 0.")   # happened
                pass
        dst_idx = torch.tensor(dst_idx).to(src_idx.device)
        # new_src_idx = torch.tensor(dst_idx).to(src_idx.device)
        new_src_idx = torch.tensor(dst_idx, device=src_idx.device).clone().detach()

        src_idx_all.append(new_src_idx)
        dst_idx_all.append(dst_idx)

    src_idx_all = torch.cat(src_idx_all)
    dst_idx_all = torch.cat(dst_idx_all)

    return src_idx_all, dst_idx_all
