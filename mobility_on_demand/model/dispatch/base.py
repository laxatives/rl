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

