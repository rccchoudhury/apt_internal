import math

import torch
import torch.nn.functional as F

def generate_edge_dict(h, w):
    edge_dict = {}
    for i in range(h-1):
        for j in range(w):
            edge_dict[len(edge_dict)] = (i, j, i+1, j)
    for i in range(h):
        for j in range(w-1):
            edge_dict[len(edge_dict)] = (i, j, i, j+1)

    return edge_dict

def create_edge_embeds(token_embeds, window_size=0):
    B, N, C = token_embeds.shape
    h, w = int(math.sqrt(N)), int(math.sqrt(N))
    
    token_embeds = token_embeds.view(B, h, w, -1)
    
    edge_embeds_h = F.cosine_similarity(token_embeds[:, :-1, :, :], token_embeds[:, 1:, :, :], dim=3, eps=1e-8)
    edge_embeds_w = F.cosine_similarity(token_embeds[:, :, :-1, :], token_embeds[:, :, 1:, :], dim=3, eps=1e-8)
    
    token_embeds = token_embeds.view(B, -1, C)
    
    edge_embeds_h = (edge_embeds_h + 1) / 2
    edge_embeds_w = (edge_embeds_w + 1) / 2
    
    # make similarity 0 for window boundary edges
    if window_size > 0:
        row_indices = torch.arange(h-1, device=token_embeds.device).unsqueeze(1)
        horizontal_mask = ((row_indices%window_size)!=window_size-1).float().expand(h-1, w)
        edge_embeds_h = edge_embeds_h * horizontal_mask
        
        col_indices = torch.arange(w-1, device=token_embeds.device).unsqueeze(0)
        vertical_mask = ((col_indices%window_size)!=window_size-1).float().expand(h, w-1)
        edge_embeds_w = edge_embeds_w * vertical_mask
    
    edge_embeds_h = edge_embeds_h.view(B, -1, 1)
    edge_embeds_w = edge_embeds_w.view(B, -1, 1)
    
    edge_embeds = torch.cat([edge_embeds_h, edge_embeds_w], dim=1)
    
    return edge_embeds

def group_patches(edge: torch.Tensor, edge_dict, num_patches):
    """
    edge: [B, num_edges], where edge[b, e] = 1 if edge e is retained in batch b.
    Returns: labels of shape [B, H, W] with connected components assigned.
    """
    B = edge.shape[0]
    num_patches_h, num_patches_w = int(math.sqrt(num_patches)), int(math.sqrt(num_patches))
    device = edge.device

    all_edges = []
    for e in range(len(edge_dict)): 
        i1, j1, i2, j2 = edge_dict[e]
        p1 = i1 * num_patches_w + j1
        p2 = i2 * num_patches_w + j2
        all_edges.append([p1, p2])
    edge_index = torch.tensor(all_edges, dtype=torch.long, device=device)
    num_edges = edge_index.shape[0]

    labels = torch.arange(num_patches).unsqueeze(0).expand(B, -1).to(device)

    edge_index_expanded = edge_index.unsqueeze(0).expand(B, -1, -1)
    p1_idx = edge_index_expanded[..., 0]  # shape [B, num_edges]
    p2_idx = edge_index_expanded[..., 1]

    active_edges = edge

    max_iter = 100
    for _ in range(max_iter):
        old_labels = labels.clone()

        # Gather labels for each endpoint
        p1_labels = labels.gather(1, p1_idx)
        p2_labels = labels.gather(1, p2_idx)

        # Proposed new label = min of both
        min_labels = torch.minimum(p1_labels, p2_labels)

        # Inactivate edges => big number so it won't override
        infinity = num_patches + 999999
        candidate_labels = torch.where(active_edges.bool(), min_labels, torch.full_like(min_labels, infinity, device=device))

        # scatter_reduce for p1
        labels = labels.scatter_reduce(
            dim=1,
            index=p1_idx,
            src=candidate_labels,
            reduce="min",
            include_self=True,
        )
        # scatter_reduce for p2
        labels = labels.scatter_reduce(
            dim=1,
            index=p2_idx,
            src=candidate_labels,
            reduce="min",
            include_self=True,
        )

        # Check convergence
        if torch.equal(labels, old_labels):
            break

    # Finally, reshape [B, num_patches] => [B, H, W]
    labels = labels.view(B, num_patches_h, num_patches_w)
    return labels

def calculate_merge_ratio(initial_group, num_patches):
    B = initial_group.shape[0]
    h, w = int(math.sqrt(num_patches)), int(math.sqrt(num_patches))
    device = initial_group.device
    
    presence = torch.zeros((B, h*w), dtype=torch.long, device=device)
    initial_group_flat = initial_group.view(B, -1)
    src = torch.ones_like(initial_group_flat)
    presence.scatter_add_(1, initial_group_flat, src)
    presence = (presence>0).float()
    merge_ratio = 1 - torch.mean(presence, dim=1)
    
    return merge_ratio