# paste network_data.py contents here

"""
Own from-scratch data loading + UXsim World construction for the
Bloomington TRNDP dataset.

Uses ONLY vanilla UXsim (pip package) primitives -- no code from
AlphaTransit's rl/env.py or vendored uxsim/BusHandler/ is imported here.
The raw CSV/JSON files themselves are just data (same files distributed
on HuggingFace as matrix-multiply/bloomington-tndp).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from uxsim import World


@dataclass
class NetworkData:
    node_ids: List[str]                 # ordered list, index position == node index used elsewhere
    node_xy: Dict[str, Tuple[float, float]]
    edges: List[Tuple[str, str, float, float]]  # (u, v, length_m, free_flow_speed_mps)
    demand: np.ndarray                  # (n, n) trips/hour, Dij
    existing_routes: Dict[str, List[str]]  # route name -> ordered list of node ids


def _parse_from_frames(nodes_df: pd.DataFrame, links_df: pd.DataFrame, demand_df: pd.DataFrame, routes_records: list) -> NetworkData:
    """
    Shared parsing logic once nodes/links/demand are pandas DataFrames and
    routes_records is a list of dicts with keys "name" and "nodes"
    (a list of node ids) -- works whether the frames came from local CSVs
    or from `datasets.Dataset.to_pandas()` / row iteration over a
    HuggingFace dataset split.
    """
    node_ids = nodes_df["name"].astype(str).tolist()
    node_xy = {
        str(name): (float(x), float(y))
        for name, x, y in zip(nodes_df["name"], nodes_df["x"], nodes_df["y"])
    }

    edges = []
    for u, v, length, speed in zip(links_df["start"], links_df["end"], links_df["length"], links_df["free_flow_speed"]):
        edges.append((str(u), str(v), float(length), float(speed)))

    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    D = np.zeros((n, n), dtype=np.float64)
    for o, d, volume in zip(demand_df["orig"], demand_df["dest"], demand_df["volume"]):
        o, d = str(o), str(d)
        if o in idx and d in idx:
            D[idx[o], idx[d]] += float(volume)

    existing_routes = {}
    for route in routes_records:
        existing_routes[str(route["name"])] = [str(v) for v in route["nodes"]]

    return NetworkData(node_ids=node_ids, node_xy=node_xy, edges=edges, demand=D, existing_routes=existing_routes)


def load_network_data(nodes_csv: str, links_csv: str, demand_csv: str, routes_json: str) -> NetworkData:
    """
    Loads from local CSV/JSON files.
    Actual schema (verified against bloomington_*_standard.csv):
      nodes:  name, x, y
      links:  name, start, end, length (meters), free_flow_speed (m/s)
      demand: orig, dest, volume (trips/hour)
      routes: JSON list of {"name", "short_name", "nodes": [int, ...]}
    """
    nodes_df = pd.read_csv(nodes_csv)
    links_df = pd.read_csv(links_csv)
    demand_df = pd.read_csv(demand_csv)

    import json as _json
    with open(routes_json) as f:
        routes_records = _json.load(f)

    return _parse_from_frames(nodes_df, links_df, demand_df, routes_records)


def load_network_data_from_hf(nodes_ds, links_ds, demand_ds, routes_ds) -> NetworkData:
    """
    Loads directly from HuggingFace `datasets.Dataset` splits, e.g.:

        from datasets import load_dataset
        nodes  = load_dataset("matrix-multiply/bloomington-tndp", "nodes", split="benchmark")
        links  = load_dataset("matrix-multiply/bloomington-tndp", "links", split="benchmark")
        demand = load_dataset("matrix-multiply/bloomington-tndp", "demand", split="benchmark")
        routes = load_dataset("matrix-multiply/bloomington-tndp", "existing_routes", split="benchmark")
        net = load_network_data_from_hf(nodes, links, demand, routes)

    NOTE: this could not be tested against the live dataset from this
    build environment (huggingface.co is not reachable from the sandbox
    used to build/verify the rest of this project). Column names are
    assumed identical to the CSV/JSON schema (same underlying files per
    the AlphaTransit repo's data table), but if HF packaging renamed any
    column, this will raise a KeyError -- run the inspection snippet
    below first to confirm before relying on this in a long training run.

        print(nodes.column_names, nodes[0])
        print(links.column_names, links[0])
        print(demand.column_names, demand[0])
        print(routes.column_names, routes[0])
    """
    nodes_df = nodes_ds.to_pandas()
    links_df = links_ds.to_pandas()
    demand_df = demand_ds.to_pandas()
    routes_records = list(routes_ds)  # list of dicts, one per route

    return _parse_from_frames(nodes_df, links_df, demand_df, routes_records)


def build_world(net: NetworkData, tmax: float = 10000.0, deltan: int = 5) -> World:
    """
    Builds a vanilla UXsim World from the raw network -- no bus-specific
    logic here, just nodes/links matching the paper's Sec. IV-A setup
    (DELTAN=5 platoons, DELTAT=1s reaction time, Tmax=10,000s).
    """
    W = World(
        name="bloomington_trndp",
        deltan=deltan,
        tmax=tmax,
        print_mode=0,
        save_mode=0,
        show_mode=0,
        random_seed=0,
    )
    for nid in net.node_ids:
        x, y = net.node_xy[nid]
        W.addNode(nid, x, y)

    for (u, v, length, speed) in net.edges:
        W.addLink(f"{u}_{v}", u, v, length=length, free_flow_speed=speed)
        W.addLink(f"{v}_{u}", v, u, length=length, free_flow_speed=speed)  # bidirectional per Def. 2

    return W


def inject_car_demand(W: World, net: NetworkData, alpha: float, t_end: float = 3600.0):
    """
    Injects the (1 - alpha) share of OD demand as private-car trips, so
    buses experience real congestion from competing car traffic -- this
    is the piece that a free-flow-only approximation would skip.
    """
    n = len(net.node_ids)
    car_share = 1.0 - alpha
    for i in range(n):
        for j in range(n):
            trips_per_hour = net.demand[i, j] * car_share
            if trips_per_hour <= 0:
                continue
            W.adddemand(net.node_ids[i], net.node_ids[j], t_start=0, t_end=t_end, flow=trips_per_hour / 3600.0)