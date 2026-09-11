# paste passenger_demand.py contents here

"""
Own from-scratch passenger generation: turns OD itineraries + hourly
demand rates into individual synthetic passengers with Poisson arrival
times over the simulation horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from network_data import NetworkData


@dataclass
class Passenger:
    name: int
    orig: str
    dest: str
    depart_time: float                       # time they arrive at the first stop, wanting to travel
    itinerary: List[Tuple[str, str, str]]     # [(route, board_stop, alight_stop), ...]
    leg_idx: int = 0
    wait_start_time: float = None             # set when they start waiting for the *current* leg's bus
    board_time: float = None                  # most recent boarding time (reset per leg)
    n_transfers: int = 0
    completed: bool = False
    aborted: bool = False                     # could not board within sim horizon
    total_wait_time: float = 0.0
    total_invehicle_time: float = 0.0

    def current_leg(self):
        if self.leg_idx < len(self.itinerary):
            return self.itinerary[self.leg_idx]
        return None


def generate_passengers(
    net: NetworkData,
    itineraries: Dict[Tuple[str, str], List[Tuple[str, str, str]]],
    alpha: float,
    t_horizon_s: float,
    rng: np.random.Generator,
) -> List[Passenger]:
    """
    For each OD pair with a known itinerary, generates a Poisson process
    of passenger arrivals at rate (alpha * D_ij) trips/hour over
    t_horizon_s seconds.
    """
    idx = {nid: i for i, nid in enumerate(net.node_ids)}
    passengers: List[Passenger] = []
    pid = 0
    horizon_hours = t_horizon_s / 3600.0

    for (orig, dest), legs in itineraries.items():
        i, j = idx[orig], idx[dest]
        rate_per_hour = net.demand[i, j] * alpha
        if rate_per_hour <= 0:
            continue
        expected_count = rate_per_hour * horizon_hours
        n_passengers = rng.poisson(expected_count)
        if n_passengers == 0:
            continue
        depart_times = np.sort(rng.uniform(0, t_horizon_s, size=n_passengers))
        for t in depart_times:
            passengers.append(
                Passenger(name=pid, orig=orig, dest=dest, depart_time=float(t), itinerary=list(legs))
            )
            pid += 1

    passengers.sort(key=lambda p: p.depart_time)
    return passengers
