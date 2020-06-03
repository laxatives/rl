import collections
from typing import Any, Dict, List, Set, Tuple

from grid import Grid


HEX_GRID = Grid()


class Driver:
    def __init__(self, od: Dict[str, Any]):
        self.driver_id = od['driver_id']  # type: str
        self.coord = od['driver_location']  # type: Tuple[float, float]
        self.location = loc_to_grid(od['driver_location'])

    def __repr__(self):
        return f'Driver:{self.driver_id}@{self.location}'


class Request:
    def __init__(self, od: Dict[str, Any]):
        self.request_id = od['order_id']  # type: str
        self.start_coord = od['order_start_location']  # type: Tuple[float, float]
        self.start_loc = loc_to_grid(od['order_start_location'])
        self.end_coord = od['order_finish_location']  # type: Tuple[float, float]
        self.end_loc = loc_to_grid(od['order_finish_location'])
        self.request_ts = od['timestamp']  # type: int
        self.finish_ts = od['order_finish_timestamp']  # type: int
        self.day_of_week = od['day_of_week']  # type: int
        self.reward = od['reward_units']  # type: float

    def __repr__(self):
        return f'Request:{self.request_id},{self.start_loc}-{self.end_loc}:{self.reward}'


class DispatchCandidate:
    def __init__(self, od: Dict[str, Any]):
        self.driver_id = od['driver_id']  # type: str
        self.request_id = od['order_id']  # type: str
        self.distance = od['order_driver_distance']  # type: float
        self.eta = od['pick_up_eta']  # type: float

    def __repr__(self):
        return f'Candidate:{self.driver_id},{self.request_id}:{self.distance},{self.eta}'


class RepositionData:
    def __init__(self, r: Dict[str, Any]):
        self.timestamp = r['timestamp']  # type: int
        self.drivers = []  # type: List[Tuple[str, str]]
        for d in r['driver_info']:
            self.drivers.append((d['driver_id'], d['grid_id']))
        self.day_of_week = r['day_of_week']  # type: int


def parse_dispatch(dispatch_input: List[Dict[str, Any]]) -> (Dict[str, Driver], Dict[str, Request], Dict[str, Set[DispatchCandidate]]):
    drivers = dict()  # type: Dict[str, Driver]
    requests = dict()  # type: Dict[str, Request]
    candidates = collections.defaultdict(set)  # type: Dict[str, Set[DispatchCandidate]]
    for candidate in dispatch_input:
        driver = Driver(candidate)
        drivers[driver.driver_id] = driver
        request = Request(candidate)
        requests[request.request_id] = request
        candidates[request.request_id].add(DispatchCandidate(candidate))
    return drivers, requests, candidates


def loc_to_grid(location: Tuple[float, float]) -> str:
    # This is actually not bad
    #return f'{location[1]:0.2f},{location[0]:0.2f}'
    return HEX_GRID.lookup(location[0], location[1])