import collections
from typing import Any, Dict, Set, Tuple


# Apparently these aren't correct...
LAT_RANGE = (30.652828, 30.727818)
LNG_RANGE = (104.042102, 104.129591)


class Driver:
    def __init__(self, od):
        self.driver_id = od['driver_id']
        self.location = loc_to_grid(od['driver_location'])

    def __repr__(self):
        return f'Driver:{self.driver_id}@{self.location}'


class Request:
    def __init__(self, od):
        self.request_id = od['order_id']
        self.start_loc = loc_to_grid(od['order_start_location'])
        self.end_loc = loc_to_grid(od['order_finish_location'])
        self.request_ts = od['timestamp']
        self.finish_ts = od['order_finish_timestamp']
        self.day_of_week = od['day_of_week']
        self.reward = od['reward_units']

    def __repr__(self):
        return f'Request:{self.request_id},{self.start_loc}-{self.end_loc}:{self.reward}'


class DispatchCandidate:
    def __init__(self, od):
        self.driver_id = od['driver_id']
        self.request_id = od['order_id']
        self.distance = od['order_driver_distance']
        self.eta = od['pick_up_eta']

    def __repr__(self):
        return f'Candidate:{self.driver_id},{self.request_id}:{self.distance},{self.eta}'


class RepositionData:
    def __init__(self, r):
        self.timestamp = r['timestamp']
        self.drivers = []
        for d in r['driver_info']:
            self.drivers.append((d['driver_id'], d['grid_id']))
        self.day_of_week = r['day_of_week']

def parse_dispatch(dispatch_input: Dict[str, Any]) -> (Dict[str, Driver], Dict[str, Request], Dict[str, Set[DispatchCandidate]]):
    drivers = dict()
    requests = dict()
    candidates = collections.defaultdict(set)
    for candidate in dispatch_input:
        driver = Driver(candidate)
        drivers[driver.driver_id] = driver
        request = Request(candidate)
        requests[request.request_id] = request
        candidates[request.request_id].add(DispatchCandidate(candidate))
    return drivers, requests, candidates


def loc_to_grid(location: Tuple[float, float]) -> str:
    # TODO: Convert to Didi grid
    # Apparently we can't use libraries...
    #h3.geo_to_h3(location[1], location[0], h3_resolution)

    # This is actually not bad
    return f'{location[1]:0.2f},{location[0]:0.2f}'
