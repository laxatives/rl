import collections
import csv
import math
import os
import random
import time
from abc import abstractmethod
from typing import Dict, List, Set, Tuple

from parse import DispatchCandidate, Driver, HEX_GRID, Request


CANCEL_DISTANCE_FIT = lambda x: 0.02880619 * math.exp(0.00075371 * x)
STEP_SECONDS = 2


class Dispatcher:
    def __init__(self, alpha, gamma, idle_reward, open_reward):
        self.alpha = alpha
        self.gamma = gamma
        self.idle_reward = idle_reward
        self.open_reward = open_reward

    @staticmethod
    def _init_state_values() -> Dict[Tuple[str, int], float]:
        state_values = collections.defaultdict(float)
        value_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_values.csv')
        with open(value_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                grid_id, t, value = row
                state_values[(grid_id, t)] = float(value)
        return state_values

    @staticmethod
    def _fallback_state_values() -> Dict[str, float]:
        state_values = collections.defaultdict(float)
        value_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_values_181486.csv')
        with open(value_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                grid_id, value = row
                state_values[grid_id] = float(value)
        return state_values

    @staticmethod
    def _get_state(grid_id: str, t: float) -> Tuple[str, int]:
        t = time.gmtime(t)
        return grid_id, 24 * t.tm_wday + t.tm_hour

    @abstractmethod
    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        ...

    @abstractmethod
    def get_grid_ids(self) -> Set[str]:
        ...

    @abstractmethod
    def state_value(self, grid_id: str, t: float) -> float:
        ...

    @abstractmethod
    def update_state_value(self, grid_id: str, t: float, delta: float) -> None:
        ...


class ScoredCandidate:
    def __init__(self, candidate: DispatchCandidate, score: float):
        self.candidate = candidate
        self.score = score

    def __repr__(self):
        return f'{self.candidate}|{self.score}'


class Sarsa(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward, open_reward):
        super().__init__(alpha, gamma, idle_reward, open_reward)

        # Expected gain from each driver in (location)
        self.state_values = Dispatcher._init_state_values()
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, str]:
        # Rank candidates based on incremental driver value improvement
        ranked = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            self.timestamp = max(request.request_ts, self.timestamp)

            v0 = self.state_value(driver.location, self.timestamp)  # Value of the driver current position
            end_ts = self.timestamp + candidate.eta + request.finish_ts - request.request_ts
            v1 = self.state_value(request.end_loc, end_ts)  # Value of the proposed new position
            expected_reward = completion_rate(candidate.distance) * request.reward
            if expected_reward > 0:
                # Best incremental improvement (get the ride AND improve driver position)
                discount = math.pow(self.gamma, (request.finish_ts - request.request_ts + candidate.eta) / STEP_SECONDS)
                update = expected_reward + discount * v1 - v0
                if update > 0:
                    ranked.append(ScoredCandidate(candidate, update))

        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, str]
        for scored in sorted(ranked, key=lambda x: x.score, reverse=True):  # type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)
            request = requests[candidate.request_id]
            dispatch[request.request_id] = candidate.driver_id

            driver = drivers[candidate.driver_id]
            self.update_state_value(driver.location, self.timestamp, self.alpha * scored.score)

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue

            v0 = self.state_value(driver.location, self.timestamp)
            v1 = 0
            for destination, probability in HEX_GRID.idle_transitions(self.timestamp, driver.location).items():
                v1 += probability * self.state_value(destination, self.timestamp + STEP_SECONDS)
            update = self.idle_reward + self.gamma * v1 - v0
            self.update_state_value(driver.location, self.timestamp, self.alpha * update)

        # Update value (positive) for open requests
        for request in requests.values():
            if request.request_id in dispatch:
                continue
            v0 = self.state_value(request.start_loc, self.timestamp)
            end_ts = self.timestamp + request.finish_ts - request.request_ts
            v1 = self.state_value(request.end_loc, end_ts)
            discount = math.pow(self.gamma, request.finish_ts - request.request_ts)
            update = self.open_reward * (request.reward + discount * v1 - v0)
            self.update_state_value(request.start_loc, self.timestamp, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set([grid_id for grid_id, _ in self.state_values.keys()])

    def state_value(self, grid_id: str, t: float) -> float:
        u = (t % 3600) / 3600
        return (1 - u) * self.state_values[self._get_state(grid_id, t)] +\
                u * self.state_values[self._get_state(grid_id, t + 3600)]

    def update_state_value(self, grid_id: str, t: float, delta: float) -> None:
        u = (t % 3600) / 3600
        self.state_values[self._get_state(grid_id, t)] += (1 - u) * delta
        self.state_values[self._get_state(grid_id, t + 3600)] += u * delta


class Dql(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward, open_reward):
        super().__init__(alpha, gamma, idle_reward, open_reward)
        self.student = Dispatcher._init_state_values()
        self.teacher = Dispatcher._init_state_values()
        self.timestamp = 0

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, str]:
        #  Flip a coin
        if random.random() < 0.5:
            self.student, self.teacher = self.teacher, self.student

        # Score candidates based on incremental driver value improvement
        ranked = []  # type: List[ScoredCandidate]
        for candidate in set(c for cs in candidates.values() for c in cs):  # type: DispatchCandidate
            request = requests[candidate.request_id]
            driver = drivers[candidate.driver_id]
            self.timestamp = max(request.request_ts, self.timestamp)

            v0 = self.state_value(driver.location, self.timestamp)  # Value of the driver current position
            end_ts = self.timestamp + candidate.eta + request.finish_ts - request.request_ts
            v1 = self.state_value(request.end_loc, end_ts)  # Value of the proposed new position
            expected_reward = completion_rate(candidate.distance) * request.reward
            if expected_reward > 0:
                # Best incremental improvement (get the ride AND improve driver position)
                discount = math.pow(self.gamma, (request.finish_ts - request.request_ts + candidate.eta) / STEP_SECONDS)
                update = expected_reward + discount * v1 - v0
                if update > 0:
                    ranked.append(ScoredCandidate(candidate, update))

        # Assign drivers
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, str]
        for scored in sorted(ranked, key=lambda x: x.score, reverse=True):  # type: ScoredCandidate
            candidate = scored.candidate
            if candidate.request_id in dispatch or candidate.driver_id in assigned_driver_ids:
                continue
            assigned_driver_ids.add(candidate.driver_id)

            request = requests[candidate.request_id]
            dispatch[request.request_id] = candidate.driver_id

            # Teacher provides the destination position value
            v1 = self._get_teacher_value(request.end_loc,
                                         self.timestamp + candidate.eta + request.finish_ts - request.request_ts)

            # Compute student update
            driver = drivers[candidate.driver_id]
            v0 = self._get_student_value(request.start_loc, self.timestamp)
            expected_reward = completion_rate(candidate.distance) * request.reward
            discount = math.pow(self.gamma, (request.finish_ts - request.request_ts + candidate.eta) / STEP_SECONDS)
            update = expected_reward + discount * v1 - v0
            self.update_state_value(driver.location, self.timestamp, self.alpha * update)

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self._get_student_value(driver.location, self.timestamp)
            v1 = 0
            for destination, probability in HEX_GRID.idle_transitions(self.timestamp, driver.location).items():
                v1 += probability * self._get_teacher_value(destination, self.timestamp + STEP_SECONDS)
            update = self.idle_reward + self.gamma * v1 - v0
            self.update_state_value(driver.location, self.timestamp, self.alpha * update)

        # Update value (positive) for open requests
        for request in requests.values():
            if request.request_id in dispatch:
                continue
            v0 = self._get_student_value(request.start_loc, self.timestamp)
            v1 = self._get_teacher_value(request.end_loc, self.timestamp + request.finish_ts - request.request_ts)
            discount = math.pow(self.gamma, request.finish_ts - request.request_ts)
            update = self.open_reward * (request.reward + discount * v1 - v0)
            self.update_state_value(request.start_loc, self.timestamp, self.alpha * update)

        return dispatch

    def get_grid_ids(self) -> Set[str]:
        return set([grid_id for grid_id, _ in self.student.keys()]).\
            union(set([grid_id for grid_id, _ in self.teacher.keys()]))

    def _get_student_value(self, grid_id: str, t: float) -> float:
        return self.student[self._get_state(grid_id, t)]

    def _get_teacher_value(self, grid_id: str, t: float) -> float:
        return self.teacher[self._get_state(grid_id, t)]

    def state_value(self, grid_id: str, t: float) -> float:
        state = self._get_state(grid_id, t)
        return 0.5 * (self.student[state] + self.teacher[state])

    def update_state_value(self, grid_id: str, t: float, delta: float) -> None:
        self.student[self._get_state(grid_id, t)] += delta


def completion_rate(distance_meters: float) -> float:
    return 1 - max(min(CANCEL_DISTANCE_FIT(distance_meters), 1), 0)