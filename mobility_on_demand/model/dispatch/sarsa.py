import collections
from typing import Dict, Set

from dispatch.base import Dispatcher
from utils import DispatchCandidate, Driver, Request


class Sarsa(Dispatcher):
    def __init__(self, alpha, gamma, idle_reward):
        super().__init__(alpha, gamma, idle_reward)
        self.state_values = collections.defaultdict(int)  # Expected reward for a driver in each geohash

    def dispatch(self, drivers: Dict[str, Driver], requests: Dict[str, Request],
              candidates: Dict[str, Set[DispatchCandidate]]) -> Dict[str, DispatchCandidate]:
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]

        # Greedily match highest reward requests first
        for request in sorted(requests.values(), key=lambda x: x.reward, reverse=True):  # type: Request
            best_candidate: DispatchCandidate = None
            best_update: int = 0  # Wait if value is negative

            for candidate in sorted(candidates[request.request_id], key=lambda x: x.distance,
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
            self.state_values[driver.location] += self.alpha + (self.idle_reward + self.gamma * v1 - v0)

        return dispatch