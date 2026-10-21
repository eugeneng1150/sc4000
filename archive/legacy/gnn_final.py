"""Core GraphSAGE configuration ranker, extracted from the main-branch model.

Dependencies: NumPy and PyTorch.
A batch contains different configurations of ONE computation graph.
Lower scores indicate configurations predicted to run faster.

Example (data is an already-loaded mapping of NumPy arrays):
    inputs = prepare_inputs(data, config_indices=[0, 1, 2])
    model = GraphSAGERanker(
        node_dim=inputs["base_node"].shape[-1],
        config_dim=inputs["config_node"].shape[-1],
    )
    scores = model(**inputs)
    runtimes = torch.as_tensor(data["config_runtime"][[0, 1, 2]], dtype=torch.float32)
    loss = pairwise_margin_ranking_loss(scores, runtimes)
    loss.backward()

For prediction, call model.eval() and use torch.inference_mode(), then sort
scores ascending. The caller controls training, batching, and device placement.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_adjacency(edge_index, node_count):
    neighbors = [set() for _ in range(node_count)]
    for src, dst in np.asarray(edge_index, dtype=np.int64):
        if 0 <= src < node_count and 0 <= dst < node_count:
            neighbors[int(src)].add(int(dst))
            neighbors[int(dst)].add(int(src))
    return neighbors


def khop_subgraph_nodes(edge_index, node_count, seed_nodes, hops=2, max_nodes=512):
    neighbors = build_adjacency(edge_index, node_count)
    seed_nodes = [int(n) for n in seed_nodes if 0 <= int(n) < node_count]
    if not seed_nodes:
        return np.arange(min(node_count, max_nodes), dtype=np.int64)
    reached = set(seed_nodes)
    frontier = set(seed_nodes)
    ordered = list(dict.fromkeys(seed_nodes))
    for _ in range(hops):
        next_frontier = set()
        for node in sorted(frontier):
            for nbr in sorted(neighbors[node]):
                if nbr not in reached:
                    reached.add(nbr)
                    next_frontier.add(nbr)
                    ordered.append(nbr)
        frontier = next_frontier
        if not frontier:
            break
    if len(ordered) > max_nodes:
        seed_set = list(dict.fromkeys(seed_nodes))
        remaining = [node for node in ordered if node not in set(seed_set)]
        ordered = seed_set + remaining[:max(0, max_nodes - len(seed_set))]
    return np.array(sorted(ordered), dtype=np.int64)


def induced_edges(edge_index, kept_nodes):
    kept_nodes = np.asarray(kept_nodes, dtype=np.int64)
    old_to_new = {int(old): i for i, old in enumerate(kept_nodes)}
    src_list = []
    dst_list = []
    for src, dst in np.asarray(edge_index, dtype=np.int64):
        if int(src) in old_to_new and int(dst) in old_to_new:
            src_list.append(old_to_new[int(src)])
            dst_list.append(old_to_new[int(dst)])
            src_list.append(old_to_new[int(dst)])
            dst_list.append(old_to_new[int(src)])
    for i in range(len(kept_nodes)):
        src_list.append(i)
        dst_list.append(i)
    return (np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64), old_to_new)


def prepare_inputs(data, config_indices, hops=2, max_nodes=512, device=None):
    """Build the keyword arguments for GraphSAGERanker.forward().

    Required arrays:
      node_feat:        [all_nodes, node_feature_count]
      node_opcode:      [all_nodes]
      edge_index:       [edges, 2], with source/destination node IDs
      node_config_ids:  [configurable_nodes]
      node_config_feat: [all_candidates, configurable_nodes, config_feature_count]

    Retain a two-hop neighbourhood around configurable nodes by default.
    All configurable seeds are preserved, even when their count exceeds max_nodes.
    device is optional and caller-supplied; there is no automatic device tuning.
    """
    node_count = len(data["node_feat"])
    seeds = np.asarray(data["node_config_ids"], dtype=np.int64)
    kept = khop_subgraph_nodes(data["edge_index"], node_count, seeds, hops, max_nodes)
    src, dst, old_to_new = induced_edges(data["edge_index"], kept)

    # Normalise each feature over retained nodes, matching the original model.
    base = np.asarray(data["node_feat"][kept], dtype=np.float32)
    base = np.log1p(np.maximum(base, 0.0))
    base = (base - base.mean(axis=0, keepdims=True)) / (
        base.std(axis=0, keepdims=True) + 1e-6
    )
    opcode = np.clip(np.asarray(data["node_opcode"][kept], dtype=np.int64), 0, 255)

    # Attach each candidate's settings to the corresponding retained nodes.
    indices = np.asarray(config_indices, dtype=np.int64)
    selected = np.asarray(data["node_config_feat"][indices], dtype=np.float32)
    selected = np.where(selected == -1, 0.0, selected)
    config = np.zeros((len(indices), len(kept), selected.shape[-1]), dtype=np.float32)
    mask = np.zeros(len(kept), dtype=np.float32)
    for position, original_node in enumerate(seeds):
        if int(original_node) in old_to_new:
            local_node = old_to_new[int(original_node)]
            config[:, local_node, :] = selected[:, position, :]
            mask[local_node] = 1.0

    # Incoming degree includes the reverse edges and self-loops added above.
    degree = np.bincount(dst, minlength=len(kept)).astype(np.float32)
    return {
        "base_node": torch.as_tensor(base, dtype=torch.float32, device=device),
        "opcode": torch.as_tensor(opcode, dtype=torch.long, device=device),
        "config_node": torch.as_tensor(config, dtype=torch.float32, device=device),
        "edge_src": torch.as_tensor(src, dtype=torch.long, device=device),
        "edge_dst": torch.as_tensor(dst, dtype=torch.long, device=device),
        "degree": torch.as_tensor(degree, dtype=torch.float32, device=device),
        "config_mask": torch.as_tensor(mask, dtype=torch.float32, device=device),
    }



class GraphSAGERanker(nn.Module):
    """Two residual mean-aggregation layers, four pooling summaries, one score."""
    def __init__(self, node_dim, config_dim, hidden_dim=32, opcode_vocab_size=256, opcode_emb_dim=16):
        super().__init__()
        self.opcode_embedding = nn.Embedding(opcode_vocab_size, opcode_emb_dim)
        self.input_proj = nn.Linear(node_dim + config_dim + opcode_emb_dim, hidden_dim)
        self.sage1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def sage_step(self, h, edge_src, edge_dst, degree, layer):
        neigh = torch.zeros_like(h)
        neigh.index_add_(1, edge_dst, h[:, edge_src, :])
        neigh = neigh / degree.view(1, -1, 1).clamp_min(1.0)
        return F.relu(layer(torch.cat([h, neigh], dim=-1))) + h

    def forward(self, base_node, opcode, config_node, edge_src, edge_dst, degree, config_mask):
        # Reuse static graph information for every candidate in the batch.
        batch_size = config_node.shape[0]
        base = base_node.unsqueeze(0).expand(batch_size, -1, -1)
        opcode_emb = self.opcode_embedding(opcode).unsqueeze(0).expand(batch_size, -1, -1)
        # Join the feature lists for each node, then encode them as hidden features.
        h = F.relu(self.input_proj(torch.cat([base, config_node, opcode_emb], dim=-1)))
        h = self.sage_step(h, edge_src, edge_dst, degree, self.sage1)
        h = self.sage_step(h, edge_src, edge_dst, degree, self.sage2)

        # Summarise all nodes and configurable nodes separately (mean and max).
        global_pool = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        mask = config_mask.view(1, -1, 1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        config_mean = (h * mask).sum(dim=1) / denom
        config_max = h.masked_fill(mask == 0, -1e9).max(dim=1).values
        config_max = torch.where((mask.sum(dim=1) > 0), config_max, torch.zeros_like(config_max))
        config_pool = torch.cat([config_mean, config_max], dim=-1)
        return self.head(torch.cat([global_pool, config_pool], dim=-1)).squeeze(-1)


def pairwise_margin_ranking_loss(scores, runtimes, margin=0.1, min_gap=0.02, max_pairs=512):
    """Encourage faster candidates to score lower, comparing within one graph.

    Exclude pairs closer than min_gap in log-runtime; sample up to max_pairs.
    With no eligible pairs, regress toward median-centred log runtimes instead.
    """
    log_runtime = torch.log1p(runtimes)
    diff = log_runtime.view(-1, 1) - log_runtime.view(1, -1)
    pairs = torch.nonzero(diff < -min_gap, as_tuple=False)
    if pairs.numel() == 0:
        return F.smooth_l1_loss(scores, log_runtime - log_runtime.median())
    if len(pairs) > max_pairs:
        idx = torch.randperm(len(pairs), device=scores.device)[:max_pairs]
        pairs = pairs[idx]
    fast = pairs[:, 0]
    slow = pairs[:, 1]
    return F.relu(margin + scores[fast] - scores[slow]).mean()
