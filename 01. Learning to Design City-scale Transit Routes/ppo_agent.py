# paste ppo_agent.py contents here

"""
Own from-scratch PPO implementation (clipped surrogate + GAE) for the
TransitEnv from the AlphaTransit repo (rl/env.py).

This does not reuse rl/ppo_agent.py -- only TransitEnv itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models import GATv2ActorCritic, ModelConfig

# Node feature indices, per rl/env.py's observation_space docstring:
# 0 x, 1 y, 2 degree, 3 d_out, 4 d_in,
# 5 d_out_cur_local, 6 d_in_cur_local, 7 d_out_comp_local, 8 d_in_comp_local,
# 9 d_out_cur_global, 10 d_in_cur_global, 11 d_out_comp_global, 12 d_in_comp_global,
# 13 in_current_route_flag, 14 in_completed_routes, 15 is_valid_next_flag
IS_VALID_NEXT_FEATURE_IDX = 15


@dataclass
class PPOConfig:
    gamma: float = 0.999           # matches paper's --gamma=0.999 (long-horizon route credit assignment)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs_per_update: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    rollout_steps: int = 512
    device: str = "cpu"


@dataclass
class Transition:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    route_progress: np.ndarray
    frontier_index: int
    valid_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


def obs_to_valid_mask(obs: dict, n_nodes: int) -> np.ndarray:
    """True where the node's is_valid_next_flag feature == 1."""
    return obs["node_features"][:, IS_VALID_NEXT_FEATURE_IDX] > 0.5


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
):
    """
    delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t       = sum_l (gamma*lam)^l * delta_{t+l}
    return_t  = A_t + V(s_t)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    def __init__(self, model_cfg: ModelConfig, ppo_cfg: PPOConfig):
        self.ppo_cfg = ppo_cfg
        self.device = torch.device(ppo_cfg.device)
        self.model = GATv2ActorCritic(model_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ppo_cfg.lr)

    def _obs_to_tensors(self, obs: dict):
        node_features = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=self.device)
        edge_features = torch.as_tensor(obs["edge_features"], dtype=torch.float32, device=self.device)
        route_progress = torch.as_tensor(obs["route_progress"], dtype=torch.float32, device=self.device)
        return node_features, edge_index, edge_features, route_progress

    @torch.no_grad()
    def act(self, obs: dict):
        """Sample an action from the current policy for a single env step."""
        node_features, edge_index, edge_features, route_progress = self._obs_to_tensors(obs)
        frontier_index = int(obs["frontier_index"])
        n_nodes = node_features.shape[0]

        valid_mask_np = obs_to_valid_mask(obs, n_nodes)
        valid_mask = torch.as_tensor(valid_mask_np, dtype=torch.bool, device=self.device)

        logits, value = self.model(
            node_features, edge_index, edge_features, route_progress, frontier_index, valid_mask
        )
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), valid_mask_np

    def evaluate_actions(
        self,
        node_features_list,
        edge_index_list,
        edge_features_list,
        route_progress_list,
        frontier_indices,
        valid_masks,
        actions,
    ):
        """
        Batched re-evaluation for the PPO update. TransitEnv graphs are all
        the same fixed 143-node Bloomington network, so we process the
        minibatch as a simple Python loop over variable-size per-sample
        graphs (still correct, just not vectorized across a PyG Batch --
        acceptable at this network scale of 143 nodes / 486 edges).
        """
        log_probs, values, entropies = [], [], []
        for i in range(len(actions)):
            logits, value = self.model(
                node_features_list[i],
                edge_index_list[i],
                edge_features_list[i],
                route_progress_list[i],
                frontier_indices[i],
                valid_masks[i],
            )
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(torch.tensor(actions[i], device=self.device)))
            entropies.append(dist.entropy())
            values.append(value)
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    def update(self, buffer: RolloutBuffer, last_value: float):
        cfg = self.ppo_cfg
        transitions = buffer.transitions
        T = len(transitions)

        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)

        # Pre-convert all per-step tensors once (small network -> fine to keep in memory)
        node_features_t, edge_index_t, edge_features_t, route_progress_t = [], [], [], []
        frontier_t, valid_mask_t = [], []
        for t in transitions:
            nf, ei, ef, rp = self._obs_to_tensors(
                {
                    "node_features": t.node_features,
                    "edge_index": t.edge_index,
                    "edge_features": t.edge_features,
                    "route_progress": t.route_progress,
                }
            )
            node_features_t.append(nf)
            edge_index_t.append(ei)
            edge_features_t.append(ef)
            route_progress_t.append(rp)
            frontier_t.append(t.frontier_index)
            valid_mask_t.append(torch.as_tensor(t.valid_mask, dtype=torch.bool, device=self.device))

        indices = np.arange(T)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(cfg.epochs_per_update):
            np.random.shuffle(indices)
            for start in range(0, T, cfg.minibatch_size):
                mb_idx = indices[start : start + cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_actions = actions[mb_idx]
                mb_old_log_probs = torch.as_tensor(old_log_probs[mb_idx], device=self.device)
                mb_advantages = torch.as_tensor(advantages[mb_idx], device=self.device)
                mb_returns = torch.as_tensor(returns[mb_idx], device=self.device)

                new_log_probs, new_values, entropy = self.evaluate_actions(
                    [node_features_t[i] for i in mb_idx],
                    [edge_index_t[i] for i in mb_idx],
                    [edge_features_t[i] for i in mb_idx],
                    [route_progress_t[i] for i in mb_idx],
                    [frontier_t[i] for i in mb_idx],
                    [valid_mask_t[i] for i in mb_idx],
                    mb_actions,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_bonus = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_bonus.item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats
