import collections
import csv
import math
import os
from abc import abstractmethod
from typing import Dict, List, Set, Tuple

from parse import DispatchCandidate, Driver, Request


CANCEL_DISTANCE_FIT = lambda x: 0.02880619 * math.exp(0.00075371 * x)
STEP_SECONDS = 2
LAT_OFFSET = 0.01135 / 4
LNG_OFFSET = 0.01234 / 4


class Dispatcher:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
    @staticmethod
    def _init_state_values() -> Dict[str, float]:
        state_values = collections.defaultdict(float)
        value_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_values.csv')
        with open(value_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                grid_id, value = row
                state_values[grid_id] = float(value)
        return state_values

    @abstractmethod
    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        ...

    @abstractmethod
    def get_grid_ids(self) -> Set[str]:
        ...

    @abstractmethod
    def state_value(self, coord: Tuple[float, float], grid_id: str) -> float:
        ...

    @abstractmethod
    def state_value_grid(self, grid_id: str) -> float:
        ...

    @abstractmethod
    def update_state_value(self, coord: Tuple[float, float], grid_id: str, delta: float) -> None:
        ...


class ScoredCandidate:
    def __init__(self, candidate: DispatchCandidate, score: float):
        self.candidate = candidate
        self.score = score

    def __repr__(self):
        return f'{self.candidate}|{self.score}'


class Sarsa(Dispatcher):
    def __init__(self, alpha, gamma):
        super().__init__(alpha, gamma)
        # Expected gain from each driver in (location)
        self.state_values_grid = Dispatcher._init_state_values()
        self.offsets = [[0, 0], [0, 1], [1, 0], [1, 1], [3, 1], [1, 3]]
        self.state_values_tiled = [collections.defaultdict(float) for _ in self.offsets]
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        # Rank candidates based on incremental driver value improvement
        ranking = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            self.timestamp = max(request.request_ts, self.timestamp)

            v0 = self.state_value(driver.coord, driver.location)  # Value of the driver current position
            v1 = self.state_value(request.end_coord, request.end_loc)  # Value of the proposed new position
            likelihood = completion_rate(candidate.distance)
            if likelihood > 0:
                # Best incremental improvement (get the ride AND improve driver position)
                #complete_update = request.reward + self.gamma * v1 - v0
                #cancel_update = self.gamma * v0 - v0  # no reward, no transition
                expected_update = likelihood * request.reward + self.gamma * v1 - v0
                ranking.append(ScoredCandidate(candidate, expected_update))

        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]
        for scored in sorted(ranking, key=lambda x: x.score, reverse=True):  # type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)
            request = requests[candidate.request_id]
            dispatch[request.request_id] = candidate

            likelihood = completion_rate(candidate.distance)
            gain = likelihood * request.reward + self.gamma * self.state_value(request.end_coord, request.end_loc)

            # Update value at driver location
            driver = drivers[candidate.driver_id]
            self.update_state_value(driver.coord, driver.location, gain)

        # Reward (zero) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            gain = self.gamma * self.state_value(driver.coord, driver.location)
            self.update_state_value(driver.coord, driver.location, gain)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set(self.state_values_grid.keys())

    def state_value_grid(self, grid_id: str) -> float:
        return self.state_values_grid[grid_id]

    @staticmethod
    def _tiled_hash(lng: float, lat: float) -> str:
        return f'{lng:0.2f},{lat:0.2f}'

    def state_value(self, coord: Tuple[float, float], grid_id: str) -> float:
        value = self.state_values_grid[grid_id]
        for i, offset in enumerate(self.offsets):
            grid_id = Sarsa._tiled_hash(coord[0] + offset[0] * LNG_OFFSET, coord[1] + offset[1] * LAT_OFFSET)
            value += self.state_values_tiled[i][grid_id]
        return value / (1 + len(self.offsets))

    def update_state_value(self, coord: Tuple[float, float], grid_id: str, gain: float) -> None:
        self.state_values_grid[grid_id] = self.alpha * gain - self.state_values_grid[grid_id]
        for i, offset in enumerate(self.offsets):
            grid_id = Sarsa._tiled_hash(coord[0] + offset[0] * LNG_OFFSET, coord[1] + offset[1] * LAT_OFFSET)
            self.state_values_tiled[i][grid_id] += self.alpha * gain - self.state_values_tiled[i][grid_id]

def completion_rate(distance_meters: float) -> float:
    return 1 - max(min(CANCEL_DISTANCE_FIT(distance_meters), 1), 0)