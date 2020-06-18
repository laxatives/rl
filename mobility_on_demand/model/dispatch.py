import collections
import csv
import math
import os
from abc import abstractmethod
from typing import Dict, List, Set, Tuple

from parse import DispatchCandidate, Driver, HEX_GRID, Request


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
    def state_value(self, coord: Tuple[float, float]) -> float:
        ...

    @abstractmethod
    def state_value_grid(self, grid_id: str) -> float:
        ...

    @abstractmethod
    def update_state_value(self, coord: Tuple[float, float], delta: float) -> None:
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
        self.offsets = [[0, 0], [1, 3], [2, 6], [3, 9]]
        self.state_values_tiled = [Dispatcher._init_state_values() for _ in self.offsets]
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        # Rank candidates based on incremental driver value improvement
        ranking = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            self.timestamp = max(request.request_ts, self.timestamp)

            v0 = self.state_value(driver.coord)  # Value of the driver current position
            v1 = self.state_value(request.end_coord)  # Value of the proposed new position
            expected_reward = completion_rate(candidate.distance) * request.reward
            if expected_reward > 0:
                # Best incremental improvement (get the ride AND improve driver position)
                update = expected_reward + self.gamma * v1 - v0
                ranking.append(ScoredCandidate(candidate, update))

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

            # Update value at driver location
            driver = drivers[candidate.driver_id]
            self.update_state_value(driver.coord, self.alpha * scored.score)

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_value(driver.coord)
            v1 = v0  # no transition
            update = self.gamma * v1 - v0  # no reward
            self.update_state_value(driver.coord, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set(self.state_values_tiled[0].keys())

    def state_value_grid(self, grid_id: str) -> float:
        return self.state_values_tiled[0][grid_id]

    def state_value(self, coord: Tuple[float, float]) -> float:
        value = 0
        for i, offset in enumerate(self.offsets):
            grid_id = HEX_GRID.lookup(coord[0] + offset[0] * LNG_OFFSET, coord[1] + offset[1] * LAT_OFFSET)
            value += self.state_values_tiled[i][grid_id]
        return value / len(self.offsets)

    def update_state_value(self, coord: Tuple[float, float], delta: float) -> None:
        for i, offset in enumerate(self.offsets):
            grid_id = HEX_GRID.lookup(coord[0] + offset[0] * LNG_OFFSET, coord[1] + offset[1] * LAT_OFFSET)
            self.state_values_tiled[i][grid_id] += delta / 4

def completion_rate(distance_meters: float) -> float:
    return 1 - max(min(CANCEL_DISTANCE_FIT(distance_meters), 1), 0)