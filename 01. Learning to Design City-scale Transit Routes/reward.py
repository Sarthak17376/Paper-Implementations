# paste reward.py contents here

"""
Own from-scratch reward computation, matching the paper's Eq. 11-15 exactly.

    Psi   (Eq. 11): coverage potential = demand on OD pairs reachable via
                     the currently-built network, over total demand.
    omega          : route overlap -- average, over every network edge
                     covered by >=1 route, of (count_covering - 1)/(K-1).
    R_partial (Eq. 12): 40*Psi - 20*omega, minus an under-length penalty
                     15*(1 - |r_k|/L_max) if a route terminates early.
    sigma (Eq. 13): N_boarded / N_want
    tau   (Eq. 14): min( mean(travel_time_p)/3600 over boarded p, 1 )
    R_final (Eq. 15): 30*Psi + 15*sigma - 15*tau - 10*omega
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from network_data import NetworkData
from transit_assignment import RouteGeometry, _build_assignment_graph, _dijkstra
from bus_dispatcher import SimStats

BETA0, BETA1, BETA2 = 40.0, 20.0, 15.0   # R_partial coefficients (Eq. 12)
BETA3, BETA4, BETA5, BETA6 = 30.0, 15.0, 15.0, 10.0  # R_final coefficients (Eq. 15)


def compute_psi(net: NetworkData, geometries: Dict[str, RouteGeometry]) -> float:
    """
    Eq. 11: fraction of TOTAL demand (not modal-split-scaled) whose OD
    pair is reachable via some path through the currently-built network
    (with transfers allowed, same layered graph as transit_assignment).
    """
    total_demand = net.demand.sum()
    if total_demand <= 0 or not geometries:
        return 0.0

    adjacency, stops_to_routes = _build_assignment_graph(geometries)
    served_nodes = set(stops_to_routes.keys())
    idx = {nid: i for i, nid in enumerate(net.node_ids)}

    reachable_demand = 0.0
    n = len(net.node_ids)
    origins_with_demand = [i for i in range(n) if net.demand[i, :].sum() > 0 and net.node_ids[i] in served_nodes]

    for i in origins_with_demand:
        orig = net.node_ids[i]
        dist, _ = _dijkstra(adjacency, (orig, "__street__"))
        for j in range(n):
            if net.demand[i, j] <= 0:
                continue
            dest = net.node_ids[j]
            if orig == dest:
                continue
            if (dest, "__street__") in dist:
                reachable_demand += net.demand[i, j]

    return float(reachable_demand / total_demand)


def compute_omega(geometries: Dict[str, RouteGeometry], num_routes_total: int) -> float:
    """
    Route overlap: for every physical edge covered by >=1 route, depth =
    (count_covering - 1) / (K - 1), K = num_routes_total (the configured
    total route budget, e.g. 16 -- NOTE: the paper's text is ambiguous on
    whether K here means the fixed total design capacity or the number
    of routes built so far; we use the fixed total capacity as the more
    natural reading of "shared by all routes", and flag this explicitly
    as our own interpretation of an underspecified detail.)
    """
    if num_routes_total <= 1:
        return 0.0

    edge_route_count: Dict[tuple, int] = {}
    for k, geo in geometries.items():
        seen = set()
        for a, b in zip(geo.nodes[:-1], geo.nodes[1:]):
            key = tuple(sorted((a, b)))
            if key not in seen:
                edge_route_count[key] = edge_route_count.get(key, 0) + 1
                seen.add(key)

    if not edge_route_count:
        return 0.0

    depths = [(count - 1) / (num_routes_total - 1) for count in edge_route_count.values()]
    return float(np.mean(depths))


def compute_partial_reward(
    psi: float, omega: float, route_terminated_early: bool = False, route_len: int = 0, max_len: int = 14
) -> float:
    r = BETA0 * psi - BETA1 * omega
    if route_terminated_early:
        r -= BETA2 * (1 - route_len / max_len)
    return r


def compute_final_reward(psi: float, omega: float, stats: SimStats) -> Dict[str, float]:
    sigma = stats.n_boarded / stats.n_want if stats.n_want > 0 else 0.0
    if stats.n_boarded > 0 and stats.travel_times:
        mean_travel_hours = float(np.mean(stats.travel_times)) / 3600.0
        tau = min(mean_travel_hours, 1.0)
    else:
        tau = 1.0  # no one boarded -> worst-case travel time signal
    reward = BETA3 * psi + BETA4 * sigma - BETA5 * tau - BETA6 * omega
    return {"reward": reward, "psi": psi, "omega": omega, "sigma": sigma, "tau": tau}
