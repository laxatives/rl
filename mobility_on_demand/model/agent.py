# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05
import collections
import os
from importlib import util as import_util
from typing import Dict, Set

if import_util.find_spec('model'):
    # Use  this for submission
    from model import utils
    from model.utils import DispatchCandidate, Request, RepositionData
elif import_util.find_spec('mobility_on_demand'):
    # Use this for development
    from mobility_on_demand.model import utils
    from mobility_on_demand.model.utils import DispatchCandidate, Request, RepositionData
else:
    raise RuntimeError()



MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TODO')


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """

    def __init__(self, alpha=2/(5*60), gamma=0.9, h3_resolution=8, idle_cost=-2/(60*60)):
        self.alpha = alpha
        self.gamma = gamma
        self.h3_resolution = h3_resolution
        self.idle_cost = idle_cost

        self.state_values = collections.defaultdict(int)  # Expected reward for a driver in each geohash
        self.last_dispatch = {}
        self.last_reposition = {}

    def dispatch(self, dispatch_input):
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = utils.parse_dispatch(dispatch_input, self.h3_resolution)
        assigned_driver_ids = set()  # type: Set[str]
        dispatch = dict()  # type: Dict[str, DispatchCandidate]

        # Greedily match highest reward requests first
        for request in sorted(requests.values(), key=lambda x: x.reward, reverse=True):  # type: Request
            best_candidate: DispatchCandidate = None
            best_update: int = 0  # Wait if value is negative

            for candidate in sorted(candidates[request.request_id], key=lambda x: x.distance, reverse=True):  # type: DispatchCandidate
                if candidate.driver_id in assigned_driver_ids:
                    continue

                driver = drivers[candidate.driver_id]
                request = requests[candidate.request_id]
                v0 = self.state_values[driver.location]  # Value of the driver current position
                v1 = self.state_values[request.end_loc]  # Value of the proposed new position
                reward = request.reward

                # TODO: penalize cancellation rate
                # Best incremental improvement (get the ride AND improve driver position)
                value = reward + self.gamma * v1 - v0
                if value > best_update:
                    best_update, best_candidate = value, candidate

            # Assign driver
            if best_candidate:
                assigned_driver_ids.add(best_candidate.driver_id)
                dispatch[request.request_id] = best_candidate
                driver = drivers[best_candidate.driver_id]
                self.state_values[driver.location] += self.alpha * best_update

        # Decay idle driver positions
        for driver in drivers.values():
            if driver.driver_id in assigned_driver_ids:
                continue
            v0 = self.state_values[driver.location]
            # TODO: use idle transition probabilities
            v1 = self.state_values[driver.location]  # Driver hasn't moved if idle
            self.state_values[driver.location] += self.alpha + (self.idle_cost + self.gamma * v1 - v0)

        self.last_dispatch = dispatch
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input):
        """ Compute the reposition action for the given drivers
    :param reposition_input: a dict, the key in the dict includes:
        timestamp: int
        driver_info: a list of dict, the key in the dict includes:
            driver_id: driver_id of the idle driver in the treatment group, int
            grid_id: id of the grid the driver is located at, str
        day_of_week: int

    :return: a list of dict, the key in the dict includes:
        driver_id: corresponding to the driver_id in the od_list
        destination: id of the grid the driver is repositioned to, str
    """
        repo_action = []
        data = RepositionData(reposition_input)
        for driver in reposition_input['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append(dict(driver_id=driver['driver_id'], destination=driver['grid_id']))
        return repo_action
