import math
from abc import abstractmethod
from typing import Dict, List

from dispatch import Dispatcher
from parse import HEX_GRID, RepositionData


SPEED = 6 # 3 m/s @ 2 second interval


class Repositioner:
    def __init__(self, dispatcher: Dispatcher, gamma: float):
        self.dispatcher = dispatcher
        self.gamma = gamma

    @abstractmethod
    def reposition(self, data: RepositionData) -> List[Dict[str, str]]:
        ...


class ScoredCandidate:
    def __init__(self, grid_id: str, score: float):
        self.grid_id = grid_id
        self.score = score

    def __repr__(self):
        return f'{self.grid_id}|{self.score}'


class StateValueGreedy(Repositioner):
    def reposition(self, data: RepositionData) -> List[Dict[str, str]]:
        # Rank candidates using Dispatcher state values
        candidate_grid_ids = []  # type: List[ScoredCandidate]
        for grid_id in self.dispatcher.get_grid_ids():
            value = self.dispatcher.state_value(grid_id)
            candidate_grid_ids.append(ScoredCandidate(grid_id, value))

        # Rank discounted incremental gain
        reposition = []  # type: List[Dict[str, str]]
        for driver_id, current_grid_id in data.drivers:
            current_value = self.dispatcher.state_value(current_grid_id)
            best_grid_id, best_value = current_grid_id, 0  # don't move for lower gain
            for grid_candidate in candidate_grid_ids:
                time = HEX_GRID.distance(current_grid_id, grid_candidate.grid_id) / SPEED
                discount = math.pow(self.gamma, time)
                proposed_value = self.dispatcher.state_value(grid_candidate.grid_id)
                incremental_value = discount * proposed_value - current_value
                if incremental_value > best_value:
                    best_grid_id, best_value = grid_candidate.grid_id, incremental_value
            reposition.append(dict(driver_id=driver_id, destination=best_grid_id))
        return reposition