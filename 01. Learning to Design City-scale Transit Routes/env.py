# paste env.py contents here

"""
Own from-scratch Gymnasium environment for the TRNDP, assembling:
  network_data.py, transit_assignment.py, passenger_demand.py,
  bus_dispatcher.py, reward.py

State features (16-dim per node) follow the paper's own equations
(2)-(8) plus flags -- NOT reverse-engineered from rl/env.py:

  0,1   : normalized (x, y)
  2     : normalized node degree
  3,4   : d_out(i), d_in(i)                          -- Eq. 2 (OD marginals, whole network)
  5,6   : a_cand_{i->cur}, a_cand_{i<-cur}            -- Eq. 3,4 (gated: nonzero only if i in Ct)
  7,8   : a_core_{i->core}, a_core_{i<-core}          -- Eq. 5,6 (gated: nonzero only if i in Vcore = Vcur u Vcmp)
  9,10  : a_all_{i->cur}, a_all_{i<-cur}              -- Eq. 7 (ungated, w.r.t. Vcur)
  11,12 : a_all_{i->cmp}, a_all_{i<-cmp}              -- Eq. 8 (ungated, w.r.t. Vcmp)
  13    : 1{i in Vcur}
  14    : fraction of COMPLETED routes so far that contain i, in [0,1]
  15    : 1{i in Ct} (valid next node)

DESIGN CHOICES made where the paper's text is silent (flagged honestly):
  - Vcore in Eq. 5/6 is read as Vcur u Vcmp ("any designed route" so far,
    including the in-progress one), since Eq. 7/8 already separately
    distinguish Vcur vs Vcmp -- Vcore reads as the union.
  - Action index `n_nodes` (NO_VALID_ACTION) doubles as a voluntary
    "terminate route early" action once the route has >=2 nodes, and is
    the forced action when the candidate set Ct is empty.
  - Since our own policy (models.py) hard-masks logits to -inf for
    invalid actions, this env does not need an "agent picked an illegal
    action" recovery path in normal training -- it trusts the action is
    always in the current valid mask, matching how models.py samples.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_data import NetworkData, load_network_data, load_network_data_from_hf, build_world, inject_car_demand
from transit_assignment import (
    build_route_geometries, assign_transit_demand, compute_frequencies, compute_passenger_itineraries,
)
from passenger_demand import generate_passengers
from bus_dispatcher import BusDispatcher
from reward import compute_psi, compute_omega, compute_partial_reward, compute_final_reward

N_NODE_FEATURES = 16
NO_VALID_ACTION_LABEL = "__terminate__"


class TransitEnv(gym.Env):
    def __init__(
        self,
        net: NetworkData,
        num_routes: int = 16,
        max_route_length: int = 14,
        alpha: float = 0.3,
        t_sim: float = 10000.0,
        seed: int = 0,
    ):
        super().__init__()
        self.net = net
        self.num_routes = num_routes
        self.max_route_length = max_route_length
        self.alpha = alpha
        self.t_sim = t_sim
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.n_nodes = len(self.net.node_ids)
        self.node_idx = {nid: i for i, nid in enumerate(self.net.node_ids)}
        self._build_static_graph()

        self.action_space = spaces.Discrete(self.n_nodes + 1)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_nodes, N_NODE_FEATURES), dtype=np.float32),
                "edge_index": spaces.Box(low=0, high=self.n_nodes, shape=(2, self.edge_index.shape[1]), dtype=np.int64),
                "edge_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_features.shape[0], 2), dtype=np.float32),
                "route_progress": spaces.Box(low=0, high=1, shape=(self.num_routes,), dtype=np.float32),
                "frontier_index": spaces.Discrete(self.n_nodes + 1),
            }
        )

        # Precompute the whole-network OD marginals (Eq. 2) -- static across the episode
        self._d_out_all = self.net.demand.sum(axis=1)
        self._d_in_all = self.net.demand.sum(axis=0)

        self._reset_episode_state()

    @classmethod
    def from_csv(cls, nodes_csv: str, links_csv: str, demand_csv: str, routes_json: str, **kwargs) -> "TransitEnv":
        net = load_network_data(nodes_csv, links_csv, demand_csv, routes_json)
        return cls(net, **kwargs)

    @classmethod
    def from_huggingface(cls, nodes_ds, links_ds, demand_ds, routes_ds, **kwargs) -> "TransitEnv":
        """
        Example:
            from datasets import load_dataset
            nodes  = load_dataset("matrix-multiply/bloomington-tndp", "nodes", split="benchmark")
            links  = load_dataset("matrix-multiply/bloomington-tndp", "links", split="benchmark")
            demand = load_dataset("matrix-multiply/bloomington-tndp", "demand", split="benchmark")
            routes = load_dataset("matrix-multiply/bloomington-tndp", "existing_routes", split="benchmark")
            env = TransitEnv.from_huggingface(nodes, links, demand, routes, num_routes=16, max_route_length=14)
        """
        net = load_network_data_from_hf(nodes_ds, links_ds, demand_ds, routes_ds)
        return cls(net, **kwargs)

    # ---------------------------------------------------------------- graph
    def _build_static_graph(self):
        out_neighbors: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        edges_u, edges_v, lengths, speeds = [], [], [], []
        for (u, v, length, speed) in self.net.edges:
            ui, vi = self.node_idx[u], self.node_idx[v]
            for a, b in [(ui, vi), (vi, ui)]:
                out_neighbors[a].append(b)
                edges_u.append(a)
                edges_v.append(b)
                lengths.append(length)
                speeds.append(speed)

        self.out_neighbors = out_neighbors
        self.edge_index = np.array([edges_u, edges_v], dtype=np.int64)
        lengths = np.array(lengths, dtype=np.float32)
        speeds = np.array(speeds, dtype=np.float32)
        self.edge_features = np.stack(
            [lengths / max(lengths.max(), 1e-6), speeds / max(speeds.max(), 1e-6)], axis=1
        ).astype(np.float32)

        xs = np.array([self.net.node_xy[nid][0] for nid in self.net.node_ids], dtype=np.float32)
        ys = np.array([self.net.node_xy[nid][1] for nid in self.net.node_ids], dtype=np.float32)
        self._x_norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-6)
        self._y_norm = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-6)
        degrees = np.array([len(out_neighbors[i]) for i in range(self.n_nodes)], dtype=np.float32)
        self._degree_norm = degrees / max(degrees.max(), 1e-6)

    # ------------------------------------------------------------- episode
    def _reset_episode_state(self):
        self.completed_routes: Dict[str, List[str]] = {}       # name -> node id list
        self.completed_route_sets: List[Set[str]] = []          # one set per completed route
        self.current_route: List[str] = []
        self.frontier_idx: int = self.n_nodes  # "no frontier" sentinel until first route starts
        self.route_number = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self._reset_episode_state()
        start_node = self.rng.choice(self.net.node_ids)
        self.current_route = [start_node]
        self.frontier_idx = self.node_idx[start_node]
        obs = self._get_obs()
        return obs, {}

    def _candidates(self) -> List[int]:
        if self.frontier_idx == self.n_nodes:
            return []
        visited = set(self.node_idx[n] for n in self.current_route)
        return [nb for nb in self.out_neighbors[self.frontier_idx] if nb not in visited]

    # ------------------------------------------------------------ features
    def _get_obs(self) -> dict:
        Vcur = set(self.node_idx[n] for n in self.current_route)
        Vcmp = set()
        for s in self.completed_route_sets:
            Vcmp |= set(self.node_idx[n] for n in s)
        Vcore = Vcur | Vcmp
        Ct = set(self._candidates())

        n = self.n_nodes
        feats = np.zeros((n, N_NODE_FEATURES), dtype=np.float32)
        feats[:, 0] = self._x_norm
        feats[:, 1] = self._y_norm
        feats[:, 2] = self._degree_norm
        feats[:, 3] = self._d_out_all / max(self._d_out_all.max(), 1e-6)
        feats[:, 4] = self._d_in_all / max(self._d_in_all.max(), 1e-6)

        D = self.net.demand
        Vcur_list = list(Vcur)
        Vcore_list = list(Vcore)

        if Vcur_list:
            d_out_cur = D[:, Vcur_list].sum(axis=1)  # sum_j in Vcur D[i,j]
            d_in_cur = D[Vcur_list, :].sum(axis=0)   # sum_j in Vcur D[j,i]
        else:
            d_out_cur = np.zeros(n)
            d_in_cur = np.zeros(n)

        if Vcore_list:
            d_out_core = D[:, Vcore_list].sum(axis=1)
            d_in_core = D[Vcore_list, :].sum(axis=0)
        else:
            d_out_core = np.zeros(n)
            d_in_core = np.zeros(n)

        Vcmp_list = list(Vcmp)
        if Vcmp_list:
            d_out_cmp = D[:, Vcmp_list].sum(axis=1)
            d_in_cmp = D[Vcmp_list, :].sum(axis=0)
        else:
            d_out_cmp = np.zeros(n)
            d_in_cmp = np.zeros(n)

        denom = max(D.max(), 1e-6) * max(n, 1)  # generic normalizer for demand-sum features

        for i in range(n):
            in_Ct = i in Ct
            in_Vcore = i in Vcore
            feats[i, 5] = (d_out_cur[i] / denom) if in_Ct else 0.0
            feats[i, 6] = (d_in_cur[i] / denom) if in_Ct else 0.0
            feats[i, 7] = (d_out_core[i] / denom) if in_Vcore else 0.0
            feats[i, 8] = (d_in_core[i] / denom) if in_Vcore else 0.0
            feats[i, 9] = d_out_cur[i] / denom
            feats[i, 10] = d_in_cur[i] / denom
            feats[i, 11] = d_out_cmp[i] / denom
            feats[i, 12] = d_in_cmp[i] / denom
            feats[i, 13] = 1.0 if i in Vcur else 0.0
            feats[i, 15] = 1.0 if in_Ct else 0.0

        n_completed = max(len(self.completed_route_sets), 1)
        frac_completed = np.zeros(n, dtype=np.float32)
        for s in self.completed_route_sets:
            for nid in s:
                frac_completed[self.node_idx[nid]] += 1.0
        feats[:, 14] = frac_completed / n_completed

        route_progress = np.zeros(self.num_routes, dtype=np.float32)
        for i in range(min(self.route_number, self.num_routes)):
            route_progress[i] = 1.0
        if self.route_number < self.num_routes:
            route_progress[self.route_number] = len(self.current_route) / self.max_route_length

        return {
            "node_features": feats,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "route_progress": route_progress,
            "frontier_index": self.frontier_idx,
        }

    # ---------------------------------------------------------------- step
    def _current_geometries(self):
        """All routes (completed so far + current in-progress) as RouteGeometry, for Psi/omega."""
        all_routes = dict(self.completed_routes)
        if len(self.current_route) >= 2:
            all_routes[f"__current_{self.route_number}__"] = self.current_route
        return build_route_geometries(self.net, all_routes)

    def step(self, action: int):
        candidates = self._candidates()
        terminate = action == self.n_nodes or (candidates and action not in candidates)
        # Note: per module docstring, we trust action in-support from our own masked
        # policy; the `action not in candidates` branch is a defensive fallback only.

        if not terminate:
            next_node = self.net.node_ids[action]
            self.current_route.append(next_node)
            self.frontier_idx = action

        route_len = len(self.current_route)
        route_finished = terminate or route_len >= self.max_route_length or not self._candidates()

        if not route_finished:
            geometries = self._current_geometries()
            psi = compute_psi(self.net, geometries)
            omega = compute_omega(geometries, self.num_routes)
            reward = compute_partial_reward(psi, omega)
            obs = self._get_obs()
            return obs, reward, False, False, {"psi": psi, "omega": omega}

        # --- route finished: apply early-termination penalty if applicable, then full sim ---
        early = route_len < self.max_route_length
        geometries = self._current_geometries()
        psi = compute_psi(self.net, geometries)
        omega = compute_omega(geometries, self.num_routes)
        partial_component = compute_partial_reward(
            psi, omega, route_terminated_early=early, route_len=route_len, max_len=self.max_route_length
        )

        # Full traffic simulation over ALL routes completed so far, including this one
        route_name = f"route_{self.route_number}"
        self.completed_routes[route_name] = list(self.current_route)
        self.completed_route_sets.append(set(self.current_route))

        sim_geometries = build_route_geometries(self.net, self.completed_routes)
        segment_load = assign_transit_demand(self.net, sim_geometries, self.alpha)
        frequencies = compute_frequencies(sim_geometries, segment_load)
        itineraries = compute_passenger_itineraries(self.net, sim_geometries, self.alpha)
        passengers = generate_passengers(self.net, itineraries, self.alpha, self.t_sim, self.np_rng)

        W = build_world(self.net, tmax=self.t_sim)
        inject_car_demand(W, self.net, self.alpha, t_end=self.t_sim)
        dispatcher = BusDispatcher(W, sim_geometries, frequencies)
        dispatcher.set_passengers(passengers)
        dispatcher.schedule_all_initial_departures(self.t_sim)
        W.exec_simulation()
        dispatcher.finalize_incomplete()

        final_info = compute_final_reward(psi, omega, dispatcher.stats)
        reward = partial_component + final_info["reward"]

        self.route_number += 1
        episode_done = self.route_number >= self.num_routes

        if not episode_done:
            start_node = self.rng.choice(self.net.node_ids)
            self.current_route = [start_node]
            self.frontier_idx = self.node_idx[start_node]
        else:
            self.current_route = []
            self.frontier_idx = self.n_nodes

        obs = self._get_obs()
        info = {"psi": psi, "omega": omega, **final_info}
        return obs, reward, episode_done, False, info