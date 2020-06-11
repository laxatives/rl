import collections
import csv
import math
import os
import random
from abc import abstractmethod
from typing import Dict, List, Set, Tuple

from parse import DispatchCandidate, Driver, HEX_GRID, Request


CANCEL_DISTANCE_FIT = lambda x: 0.02880619 * math.exp(0.00075371 * x)
STEP_SECONDS = 2


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
    def state_value(self, grid_id: str) -> float:
        ...

    @abstractmethod
    def update_state_value(self, grid_id: str, delta: float) -> None:
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
        self.state_values = Dispatcher._init_state_values()
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        # Rank candidates based on incremental driver value improvement
        ranking = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            self.timestamp = max(request.request_ts, self.timestamp)

            v0 = self.state_value(driver.location)  # Value of the driver current position
            v1 = self.state_value(request.end_loc)  # Value of the proposed new position
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
            self.update_state_value(driver.location, self.alpha * scored.score)

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_value(driver.location)
            # Expected Sarsa
            v1 = 0
            for destination, probability in HEX_GRID.idle_transitions(self.timestamp, driver.location).items():
                v1 += probability * self.state_value(destination)
            update = self.gamma * v1 - v0
            self.update_state_value(driver.location, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set(self.state_values.keys())

    def state_value(self, grid_id: str) -> float:
        return self.state_values[grid_id]

    def update_state_value(self, grid_id: str, delta: float) -> None:
        self.state_values[grid_id] += delta


class Dql(Dispatcher):
    def __init__(self, alpha, gamma):
        super().__init__(alpha, gamma)
        self.student = Dispatcher._init_state_values()
        self.teacher = Dispatcher._init_state_values()
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        #  Flip a coin
        if random.random() < 0.5:
            self.student, self.teacher = self.teacher, self.student

        # Rank candidates
        updates = dict()  # type: Dict[Tuple[str, str], ScoredCandidate]
        ranking = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            # Teacher provides the destination position value
            request = requests[candidate.request_id]
            v1 = self.teacher[request.end_loc]
            self.timestamp = max(request.request_ts, self.timestamp)

            # Compute student update
            driver = drivers[candidate.driver_id]
            v0 = self.student[driver.location]
            expected_reward = completion_rate(candidate.distance) * request.reward
            update = expected_reward + self.gamma * v1 - v0
            updates[(candidate.request_id, candidate.driver_id)] = ScoredCandidate(candidate, update)

            # Joint Ranking for actual driver assignment
            v1 = self.state_value(request.end_loc)
            expected_gain = expected_reward + self.gamma * v1
            ranking.append(ScoredCandidate(candidate, expected_gain))

        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]
        for scored in sorted(ranking, key=lambda x: x.score, reverse=True):  # type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)

            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            dispatch[request.request_id] = candidate

            # Update student for selected candidate
            v0 = self.state_value(driver.location)
            gain = updates[(candidate.request_id, candidate.driver_id)].score
            self.update_state_value(driver.location, self.alpha * (gain - v0))

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_value(driver.location)
            # Expected Sarsa
            v1 = 0
            for destination, probability in HEX_GRID.idle_transitions(self.timestamp, driver.location).items():
                v1 += probability * self.state_value(destination)
            update = self.gamma * v1 - v0
            self.update_state_value(driver.location, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set(self.student.keys()).union(set(self.teacher.keys()))

    def state_value(self, grid_id: str) -> float:
        return self.student[grid_id] + self.teacher[grid_id]

    def update_state_value(self, grid_id: str, delta: float) -> None:
        self.student[grid_id] += delta


def completion_rate(distance_meters: float) -> float:
    return 1 - max(min(CANCEL_DISTANCE_FIT(distance_meters), 1), 0)