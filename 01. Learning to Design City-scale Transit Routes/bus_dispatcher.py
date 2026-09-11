# paste bus_dispatcher.py contents here

"""
Own from-scratch bus simulation layer on top of VANILLA UXsim.

Core technique ("leg-chaining"): each bus trip is broken into one UXsim
Vehicle per route segment (mode="single_trip", route forced via
Vehicle.enforce_route so it follows the RL-designed edges exactly, not
UXsim's own route choice). When a leg's vehicle reaches its destination
node, a `node_event` callback (the same mechanism vanilla UXsim's
TaxiHandler uses for pickup/dropoff) runs our boarding/alighting logic
and then dynamically spawns the *next* leg's vehicle with
departure_time = arrival_time + DWELL_TIME_S, producing a bus that
dwells at every stop and is subject to real congestion from the
car population sharing the same links.

None of this imports rl/env.py or the vendored uxsim/BusHandler module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from uxsim import World

from transit_assignment import RouteGeometry, BUS_CAPACITY
from passenger_demand import Passenger

DWELL_TIME_S = 60.0  # Sec. IV-A


@dataclass
class SimStats:
    n_boarded: int = 0
    n_want: int = 0
    n_completed: int = 0
    n_transfer_trips: int = 0
    wait_times: List[float] = field(default_factory=list)
    travel_times: List[float] = field(default_factory=list)   # wait + in-vehicle, per boarded passenger (so-far if incomplete)
    invehicle_times: List[float] = field(default_factory=list)


class BusDispatcher:
    def __init__(self, W: World, geometries: Dict[str, RouteGeometry], frequencies: Dict[str, int], capacity: int = BUS_CAPACITY):
        self.W = W
        self.geometries = geometries
        self.frequencies = frequencies
        self.capacity = capacity

        self.waiting_queues: Dict[Tuple[str, str], List[Passenger]] = {}   # (stop, route) -> [Passenger]
        self.bus_load: Dict[str, int] = {}
        self.bus_passengers: Dict[str, List[Passenger]] = {}

        self.pending_passengers: List[Passenger] = []  # sorted by depart_time, consumed via pointer
        self._pending_ptr = 0

        self.stats = SimStats()
        self._bus_counter = 0

    def set_passengers(self, passengers: List[Passenger]):
        self.pending_passengers = sorted(passengers, key=lambda p: p.depart_time)
        self._pending_ptr = 0
        self.stats.n_want = len(passengers)

    def _release_ready(self, now: float):
        """Move passengers whose depart_time has arrived into their first-leg waiting queue."""
        n = len(self.pending_passengers)
        while self._pending_ptr < n and self.pending_passengers[self._pending_ptr].depart_time <= now:
            p = self.pending_passengers[self._pending_ptr]
            leg = p.current_leg()
            if leg is not None:
                route, board_stop, _ = leg
                p.wait_start_time = now
                self.waiting_queues.setdefault((board_stop, route), []).append(p)
            self._pending_ptr += 1

    def schedule_all_initial_departures(self, t_horizon_s: float):
        """Kicks off the very first leg of every scheduled bus trip, both directions, for every route."""
        for route_name, geo in self.geometries.items():
            freq = self.frequencies.get(route_name, 1)
            headway_s = 3600.0 / max(freq, 1)

            for direction_nodes in (geo.nodes, list(reversed(geo.nodes))):
                if len(direction_nodes) < 2:
                    continue
                t = 0.0
                while t < t_horizon_s:
                    self._bus_counter += 1
                    bus_id = f"bus_{route_name}_{self._bus_counter}"
                    self.bus_load[bus_id] = 0
                    self.bus_passengers[bus_id] = []
                    self._dispatch_leg(route_name, direction_nodes, next_stop_idx=1, departure_time=t, bus_id=bus_id)
                    t += headway_s

    def _dispatch_leg(self, route_name: str, direction_nodes: List[str], next_stop_idx: int, departure_time: float, bus_id: str):
        """Creates the UXsim vehicle for the leg ending at direction_nodes[next_stop_idx]."""
        orig = direction_nodes[next_stop_idx - 1]
        dest = direction_nodes[next_stop_idx]
        link_name = f"{orig}_{dest}"

        veh = self.W.addVehicle(
            orig, dest, departure_time,
            name=f"{bus_id}_leg{next_stop_idx}",
            mode="single_trip",
        )
        veh.enforce_route([link_name], set_avoid=True)
        veh.node_event[self.W.get_node(dest)] = lambda: self._on_arrival(
            route_name, direction_nodes, next_stop_idx, bus_id
        )

    def _on_arrival(self, route_name: str, direction_nodes: List[str], stop_idx: int, bus_id: str):
        now = self.W.TIME * self.W.DELTAT
        stop = direction_nodes[stop_idx]
        self._release_ready(now)

        # --- Alighting ---
        for p in self.bus_passengers[bus_id][:]:
            leg = p.current_leg()
            if leg is not None and leg[2] == stop:
                self.bus_passengers[bus_id].remove(p)
                self.bus_load[bus_id] -= 1
                p.total_invehicle_time += now - p.board_time
                p.leg_idx += 1
                next_leg = p.current_leg()
                if next_leg is None:
                    p.completed = True
                    self.stats.n_completed += 1
                    total_time = p.total_wait_time + p.total_invehicle_time
                    self.stats.travel_times.append(total_time)
                else:
                    p.n_transfers += 1
                    self.stats.n_transfer_trips += 1
                    p.wait_start_time = now
                    next_route, next_board_stop, _ = next_leg
                    self.waiting_queues.setdefault((next_board_stop, next_route), []).append(p)

        # --- Boarding ---
        queue = self.waiting_queues.get((stop, route_name), [])
        capacity_left = self.capacity - self.bus_load[bus_id]
        boarding = queue[:max(capacity_left, 0)]
        remaining = queue[max(capacity_left, 0):]
        self.waiting_queues[(stop, route_name)] = remaining

        for p in boarding:
            wait = now - (p.wait_start_time if p.wait_start_time is not None else p.depart_time)
            p.total_wait_time += wait
            self.stats.wait_times.append(wait)
            p.board_time = now
            if not getattr(p, "_ever_boarded", False):
                p._ever_boarded = True
                self.stats.n_boarded += 1
            self.bus_passengers[bus_id].append(p)
            self.bus_load[bus_id] += 1

        # --- Chain next leg, or end of line ---
        if stop_idx < len(direction_nodes) - 1:
            self._dispatch_leg(route_name, direction_nodes, stop_idx + 1, now + DWELL_TIME_S, bus_id)
        # else: terminus reached; any remaining onboard passengers indicate an
        # itinerary/route mismatch (shouldn't happen if itineraries were built
        # from this same route's geometry).

    def finalize_incomplete(self):
        """
        At simulation end, passengers still riding or still waiting count
        toward Nboarded's average travel time using their so-far
        accumulated time (matches paper's 'mid-journey at simulation end'
        handling for the Travel Time metric, Sec. IV-C).

        Uses the ACTUAL final simulated clock time (W.TIME * W.DELTAT)
        rather than the nominal horizon passed to schedule_all_initial_departures,
        since UXsim continues simulating already-dispatched vehicles until
        they complete their trips, which can run past the nominal horizon.
        """
        t_end = self.W.TIME * self.W.DELTAT
        for bus_id, riders in self.bus_passengers.items():
            for p in riders:
                so_far_invehicle = max(t_end - p.board_time, 0.0)
                total_time = p.total_wait_time + p.total_invehicle_time + so_far_invehicle
                self.stats.travel_times.append(total_time)
        # Passengers still waiting (never boarded) at sim end are correctly
        # excluded from Nboarded-based averages (wait_times/travel_times)
        # by construction -- they were never appended to those lists.
