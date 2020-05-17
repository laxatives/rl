import collections
import random
from abc import abstractmethod
from typing import Dict, Set

from utils import DispatchCandidate, Driver, Request

class Dispatcher:
    def __init__(self, alpha, gamma, idle_reward):
        self.alpha = alpha
        self.gamma = gamma
        self.idle_reward = idle_reward

    @abstractmethod
    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        ...


class Sarsa(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward):
        super().__init__(alpha, gamma, idle_reward)
        self.state_values = collections.defaultdict(int)  # Expected reward for a driver in each geohash

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
              candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]

        # TODO: Iterate over candidates instead of requests
        # Greedily match highest reward requests first
        for request in sorted(requests.values(), key=lambda x: x.reward, reverse=True):  # type: Request
            best_candidate: DispatchCandidate = None
            best_update: int = 0  # Wait if value is negative

            for candidate in sorted(candidates[request.request_id], key=lambda x: x.eta,
                                    reverse=True):  # type: DispatchCandidate
                if candidate.driver_id in assigned_driver_ids:
                    continue

                driver = drivers[candidate.driver_id]
                v0 = self.state_values[driver.location]  # Value of the driver current position
                v1 = self.state_values[request.end_loc]  # Value of the proposed new position
                reward = request.reward

                # TODO: penalize cancellation rate
                # Best incremental improvement (get the ride AND improve driver position)
                update = reward + self.gamma * v1 - v0
                if update > best_update:
                    best_update, best_candidate = update, candidate

            # Assign driver
            if best_candidate:
                assigned_driver_ids.add(best_candidate.driver_id)
                dispatch[request.request_id] = best_candidate
                driver = drivers[best_candidate.driver_id]
                self.state_values[driver.location] += self.alpha * best_update

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_values[driver.location]
            # TODO: use idle transition probabilities
            v1 = self.state_values[driver.location]  # Driver hasn't moved if idle
            self.state_values[driver.location] += self.alpha * (self.idle_reward + self.gamma * v1 - v0)

        return dispatch


class Dql(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward):
        super().__init__(alpha, gamma, idle_reward)
        self.values_left = collections.defaultdict(int)
        self.values_right = collections.defaultdict(int)


    def _joint_policy(self, request, candidates, drivers, assigned_driver_ids):
        best_candidate: DispatchCandidate = None
        best_update: int = 0  # Wait if value is negative

        for candidate in sorted(candidates[request.request_id], key=lambda x: x.eta,
                                reverse=True):  # type: DispatchCandidate
            if candidate.driver_id in assigned_driver_ids:
                continue

            driver = drivers[candidate.driver_id]
            v0 = self.values_left[driver.location] + self.values_right[driver.location]
            v1 = self.values_left[request.end_loc] + self.values_right[request.end_loc]
            reward = request.reward

            # TODO: penalize cancellation rate
            # Best incremental improvement (get the ride AND improve driver position)
            update = reward + self.gamma * v1 - v0
            if update > best_update:
                best_update, best_candidate = update, candidate
        return best_candidate


    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
                 candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]

        # TODO: Iterate over candidates instead of requests
        # Greedily match highest reward requests first
        for request in sorted(requests.values(), key=lambda x: x.reward, reverse=True):  # type: Request
            for candidate in sorted(candidates[request.request_id], key=lambda x: x.eta,
                                    reverse=True):  # type: DispatchCandidate
                if candidate.driver_id in assigned_driver_ids:
                    continue

                # Flip a coin
                if random.random() < 0.5:
                    student, teacher = self.values_left, self.values_right
                else:
                    student, teacher = self.values_right, self.values_left

                # Teacher provides the value estimate
                q1 = teacher[request.end_loc]

                # Update the student
                driver = drivers[candidate.driver_id]
                q0 = student[driver.location]
                student[driver.location] += self.alpha * (request.reward + self.gamma * q1 - q0)

            # Assign driver
            selected = self._joint_policy(request, candidates, drivers, assigned_driver_ids)
            if selected:
                assigned_driver_ids.add(selected.driver_id)
                dispatch[request.request_id] = selected

        # Reward (negative) for idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            for values in (self.values_left, self.values_right):
                v0 = values[driver.location]
                # TODO: use idle transition probabilities
                v1 = values[driver.location]  # Driver hasn't moved if idle
                values[driver.location] += self.alpha * (self.idle_reward + self.gamma * v1 - v0)
        return dispatch
