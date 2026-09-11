# paste transit_assignment.py contents here

"""
Own from-scratch transit demand assignment + Frequency-of-Service (FOS)
computation, matching Definition 4 and Equation 19 of the paper.

ASSUMPTION FLAGGED HONESTLY: the paper does not fully specify the exact
demand-assignment algorithm used to estimate segment loads before FOS
sizing -- it only cites "max-load principle [7],[68]" and states loads
are normalized by the number of overlapping routes per segment. This
module implements a standard min-transfers-then-min-time shortest path
assignment over a route-layered graph, which is a reasonable, well-known
approach for this kind of static transit assignment, but it is our own
design choice for the unspecified part, not a byte-exact reproduction of
unpublished internal code.

Stop spacing s_k = 1 for all routes (confirmed in paper's supplementary
hyperparameter table), so every route node is a stop -- Definition 4
reduces to S(r_k) = all nodes in r_k.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from network_data import NetworkData

TRANSFER_PENALTY_SECONDS = 5 * 60.0  # heuristic transfer disutility, see module docstring
DELTA_MAX = 0.8                       # comfort threshold (Eq. 19)
BUS_CAPACITY = 40                     # C_k (Sec. IV-A)


@dataclass
class RouteGeometry:
    name: str
    nodes: List[str]                       # stop sequence (s_k=1 -> == route node sequence)
    segment_time_s: List[float]            # travel time for each consecutive stop pair (both directions equal, undirected edge reused)


def build_route_geometries(net: NetworkData, routes: Dict[str, List[str]]) -> Dict[str, RouteGeometry]:
    """Computes per-segment free-flow travel time along each route's exact edges."""
    edge_time = {}
    for (u, v, length, speed) in net.edges:
        t = length / max(speed, 1e-6)
        edge_time[(u, v)] = t
        edge_time[(v, u)] = t  # bidirectional (Def. 2)

    geometries = {}
    for name, nodes in routes.items():
        seg_times = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            if (a, b) not in edge_time:
                raise ValueError(f"Route {name}: no edge between consecutive stops {a}->{b}; route is not a valid simple path on G.")
            seg_times.append(edge_time[(a, b)])
        geometries[name] = RouteGeometry(name=name, nodes=nodes, segment_time_s=seg_times)
    return geometries


def _build_assignment_graph(geometries: Dict[str, RouteGeometry]):
    """
    Layered graph over states (node, route_or_None):
      - "route" state (p, k): currently riding route k, physically at stop p
      - "street" state (p, None): not on any vehicle, physically at stop p
    Edges:
      - board:    (p, None) -> (p, k)      weight 0,               for every route k stopping at p
      - alight:   (p, k)    -> (p, None)   weight 0
      - transfer: (p, k1)   -> (p, k2)     weight TRANSFER_PENALTY, k1 != k2, both stop at p
      - ride:     (p, k)    -> (q, k)      weight = segment travel time, for consecutive stops p,q on route k (both directions)
    Cost is a single scalar (transfer penalty dominates, so shortest path naturally minimizes transfers first in practice).
    """
    adjacency: Dict[Tuple[str, str], List[Tuple[Tuple[str, str], float]]] = {}

    def add_edge(u_state, v_state, w):
        adjacency.setdefault(u_state, []).append((v_state, w))

    stops_to_routes: Dict[str, List[str]] = {}
    for k, geo in geometries.items():
        for p in geo.nodes:
            stops_to_routes.setdefault(p, []).append(k)

    for p, route_list in stops_to_routes.items():
        street_state = (p, "__street__")
        for k in route_list:
            ride_state = (p, k)
            add_edge(street_state, ride_state, 0.0)   # board
            add_edge(ride_state, street_state, 0.0)   # alight
        for k1 in route_list:
            for k2 in route_list:
                if k1 != k2:
                    add_edge((p, k1), (p, k2), TRANSFER_PENALTY_SECONDS)  # transfer

    for k, geo in geometries.items():
        for i in range(len(geo.nodes) - 1):
            p, q = geo.nodes[i], geo.nodes[i + 1]
            t = geo.segment_time_s[i]
            add_edge((p, k), (q, k), t)  # ride forward
            add_edge((q, k), (p, k), t)  # ride backward (bidirectional operation, Def. 2)

    return adjacency, stops_to_routes


def _dijkstra(adjacency, source_state):
    dist = {source_state: 0.0}
    prev = {}
    pq = [(0.0, source_state)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adjacency.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _reconstruct_route_segments_used(prev, source_state, target_state) -> List[Tuple[str, str, str]]:
    """Walk back the shortest path, returning (route, from_node, to_node) for every 'ride' edge used."""
    path_states = [target_state]
    u = target_state
    while u != source_state:
        u = prev[u]
        path_states.append(u)
    path_states.reverse()

    segments = []
    for a, b in zip(path_states[:-1], path_states[1:]):
        (p_a, k_a), (p_b, k_b) = a, b
        if k_a == k_b and k_a != "__street__" and p_a != p_b:
            segments.append((k_a, p_a, p_b))
    return segments


def assign_transit_demand(
    net: NetworkData, geometries: Dict[str, RouteGeometry], alpha: float
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Assigns D^transit = alpha * D onto route segments via min-cost
    (transfer-penalized) shortest path per OD pair.

    Returns: {route_name: {(from_node, to_node): assigned_load_trips_per_hour}}
    """
    adjacency, stops_to_routes = _build_assignment_graph(geometries)
    served_nodes = set(stops_to_routes.keys())

    segment_load: Dict[str, Dict[Tuple[str, str], float]] = {k: {} for k in geometries}

    n = len(net.node_ids)
    # Group by unique origins actually served, to avoid O(n) Dijkstra calls when demand is sparse
    origins_with_demand = [i for i in range(n) if net.demand[i, :].sum() > 0 and net.node_ids[i] in served_nodes]

    for i in origins_with_demand:
        orig = net.node_ids[i]
        source_state = (orig, "__street__")
        dist, prev = _dijkstra(adjacency, source_state)

        for j in range(n):
            demand_ij = net.demand[i, j] * alpha
            if demand_ij <= 0:
                continue
            dest = net.node_ids[j]
            target_state = (dest, "__street__")
            if target_state not in dist:
                continue  # unreachable via current transit network -- unserved demand

            segments = _reconstruct_route_segments_used(prev, source_state, target_state)
            for (route_name, a, b) in segments:
                key = (a, b) if (a, b) in segment_load[route_name] or (b, a) not in segment_load[route_name] else (b, a)
                segment_load[route_name][key] = segment_load[route_name].get(key, 0.0) + demand_ij

    return segment_load


def _collapse_into_legs(segments: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Collapses a sequence of consecutive (route, from, to) hops into
    (route, board_stop, alight_stop) legs, merging consecutive hops on the
    same route into one ride, so a transfer is only counted where the
    route actually changes.
    """
    if not segments:
        return []
    legs = []
    cur_route, board_stop, _ = segments[0]
    last_to = segments[0][2]
    for route, a, b in segments[1:]:
        if route == cur_route:
            last_to = b
        else:
            legs.append((cur_route, board_stop, last_to))
            cur_route, board_stop, last_to = route, a, b
    legs.append((cur_route, board_stop, last_to))
    return legs


def compute_passenger_itineraries(
    net: NetworkData, geometries: Dict[str, RouteGeometry], alpha: float
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """
    Returns {(orig_node, dest_node): [(route, board_stop, alight_stop), ...]}
    for every OD pair with positive transit demand and a feasible transit
    path. Reuses the same shortest-path assignment as
    `assign_transit_demand` (kept separate to avoid recomputation coupling
    -- this is intentionally a second, independent Dijkstra pass per OD
    pair since itinerary reconstruction needs the raw segment list, not
    just aggregated loads).
    """
    adjacency, stops_to_routes = _build_assignment_graph(geometries)
    served_nodes = set(stops_to_routes.keys())
    n = len(net.node_ids)

    itineraries: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    origins_with_demand = [i for i in range(n) if net.demand[i, :].sum() > 0 and net.node_ids[i] in served_nodes]

    for i in origins_with_demand:
        orig = net.node_ids[i]
        source_state = (orig, "__street__")
        dist, prev = _dijkstra(adjacency, source_state)
        for j in range(n):
            if net.demand[i, j] * alpha <= 0:
                continue
            dest = net.node_ids[j]
            if orig == dest or dest not in served_nodes:
                continue
            target_state = (dest, "__street__")
            if target_state not in dist:
                continue
            segments = _reconstruct_route_segments_used(prev, source_state, target_state)
            legs = _collapse_into_legs(segments)
            if legs:
                itineraries[(orig, dest)] = legs

    return itineraries


def compute_frequencies(
    geometries: Dict[str, RouteGeometry],
    segment_load: Dict[str, Dict[Tuple[str, str], float]],
) -> Dict[str, int]:
    """
    Equation 19: F_k = ceil( Q_k,max^norm / (delta_max * C_k) ).

    Overlap normalization: for each segment, divide its load by the number
    of routes that also serve that same physical edge, before taking the
    max over route k's own segments (see module docstring re: this being
    our own reasonable interpretation of the paper's stated normalization).
    """
    # Count how many routes cover each undirected physical edge
    edge_route_count: Dict[Tuple[str, str], int] = {}
    for k, geo in geometries.items():
        seen_edges_this_route = set()
        for a, b in zip(geo.nodes[:-1], geo.nodes[1:]):
            edge_key = tuple(sorted((a, b)))
            if edge_key not in seen_edges_this_route:
                edge_route_count[edge_key] = edge_route_count.get(edge_key, 0) + 1
                seen_edges_this_route.add(edge_key)

    frequencies = {}
    for k, geo in geometries.items():
        loads = segment_load.get(k, {})
        max_norm_load = 0.0
        for (a, b), load in loads.items():
            edge_key = tuple(sorted((a, b)))
            overlap = max(edge_route_count.get(edge_key, 1), 1)
            norm_load = load / overlap
            max_norm_load = max(max_norm_load, norm_load)
        f_k = int(np.ceil(max_norm_load / (DELTA_MAX * BUS_CAPACITY))) if max_norm_load > 0 else 1
        frequencies[k] = max(f_k, 1)  # at least 1 bus/hour even for zero-demand routes
    return frequencies