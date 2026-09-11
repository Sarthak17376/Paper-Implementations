# paste models.py contents here

"""
Own from-scratch GATv2 Actor-Critic for the Transit Route Network Design Problem.

Architecture (matches paper's stated design):
  - 4 GATv2 blocks, pre-LayerNorm, residual connections, head-averaging
    (not concatenation) for stable multi-head aggregation across depth.
  - Edge features (length, free_flow_speed) injected at every GAT layer.
  - Pointer-style actor head: scores every node against the current
    "frontier" node embedding (the end of the route being built).
  - Pooled critic head: mean-pooled graph embedding + route-progress vector.

This is intentionally written independently of rl/models.py in the
AlphaTransit repo -- it only *consumes* the TransitEnv observation
contract (rl/env.py), not their network code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


@dataclass
class ModelConfig:
    node_feat_dim: int = 16
    edge_feat_dim: int = 2
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    num_routes: int = 16
    dropout: float = 0.0


class GATv2Block(nn.Module):
    """
    Pre-LN residual GATv2 block with head-averaging.

    h_{l+1} = h_l + GATv2( LN(h_l), edge_index, edge_attr )

    GATv2Conv is configured with concat=False so multi-head outputs are
    averaged (not concatenated), keeping the hidden dimension constant
    across all 4 stacked blocks -- this is what "head-averaging" refers
    to in the paper's architecture description.
    """

    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,          # head-averaging, not concatenation
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=True,
        )
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        out = self.conv(h_norm, edge_index, edge_attr=edge_attr)
        out = self.act(out)
        return h + out  # residual


class GATv2Encoder(nn.Module):
    """Stack of GATv2Blocks producing per-node contextual embeddings."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.node_feat_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GATv2Block(cfg.hidden_dim, cfg.num_heads, cfg.edge_feat_dim, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_features)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        return self.final_norm(h)


class PointerActorHead(nn.Module):
    """
    Pointer-style actor: for a given frontier node embedding h_f and every
    candidate node embedding h_i, compute a compatibility score

        logit_i = w_a^T tanh(W1 h_f + W2 h_i + W3 p)

    where p is a small projection of the route-progress vector, broadcast
    to every node. This lets the policy condition next-node choice on how
    far along route construction currently is.
    """

    def __init__(self, hidden_dim: int, num_routes: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.progress_proj = nn.Linear(num_routes, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, node_emb: torch.Tensor, frontier_emb: torch.Tensor, route_progress: torch.Tensor) -> torch.Tensor:
        # node_emb: (N, H)  frontier_emb: (H,)  route_progress: (R,)
        p = self.progress_proj(route_progress)  # (H,)
        combined = torch.tanh(self.w1(frontier_emb) + self.w2(node_emb) + p)  # (N, H)
        logits = self.score(combined).squeeze(-1)  # (N,)
        return logits


class PooledCriticHead(nn.Module):
    """V(s) = MLP( mean_pool(node embeddings) || route_progress )."""

    def __init__(self, hidden_dim: int, num_routes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_routes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_emb: torch.Tensor, route_progress: torch.Tensor) -> torch.Tensor:
        graph_emb = node_emb.mean(dim=0)  # (H,)
        x = torch.cat([graph_emb, route_progress], dim=-1)
        return self.mlp(x).squeeze(-1)  # scalar


class GATv2ActorCritic(nn.Module):
    """
    Full actor-critic: shared GATv2 encoder, pointer actor head, pooled
    critic head. A "no-frontier" (NO_VALID_ACTION) node embedding is
    handled by falling back to a learned dummy embedding when
    frontier_index == n_nodes (per TransitEnv's action_space semantics).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GATv2Encoder(cfg)
        self.actor_head = PointerActorHead(cfg.hidden_dim, cfg.num_routes)
        self.critic_head = PooledCriticHead(cfg.hidden_dim, cfg.num_routes)
        self.no_frontier_embedding = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)

    def forward(
        self,
        node_features: torch.Tensor,   # (N, F)
        edge_index: torch.Tensor,      # (2, E) long
        edge_attr: torch.Tensor,       # (E, 2)
        route_progress: torch.Tensor,  # (R,)
        frontier_index: int,           # scalar int, may equal N (no frontier)
        valid_mask: torch.Tensor,      # (N,) bool -- True where action is legal
    ):
        """
        Returns:
            logits: (N+1,) -- last entry is the NO_VALID_ACTION logit,
                     which is forced to -inf unless valid_mask is all-False.
            value:  scalar V(s)
        """
        node_emb = self.encoder(node_features, edge_index, edge_attr)  # (N, H)
        n_nodes = node_emb.shape[0]

        if frontier_index >= n_nodes:
            frontier_emb = self.no_frontier_embedding
        else:
            frontier_emb = node_emb[frontier_index]

        node_logits = self.actor_head(node_emb, frontier_emb, route_progress)  # (N,)

        # Mask invalid actions with -inf (paper's Option-3 masking still
        # keeps is_valid_next as a *feature*; we additionally hard-mask at
        # the policy's output so the agent's action distribution only
        # spans currently-legal moves during rollout/PPO update).
        masked_logits = node_logits.masked_fill(~valid_mask, float("-inf"))

        no_valid_action_available = (~valid_mask).all()
        no_valid_logit = torch.tensor(
            0.0 if no_valid_action_available else float("-inf"),
            device=node_logits.device,
        )

        logits = torch.cat([masked_logits, no_valid_logit.unsqueeze(0)], dim=0)  # (N+1,)
        value = self.critic_head(node_emb, route_progress)
        return logits, value