# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05
import collections
import os
from typing import Dict, Set

import utils
from utils import DispatchCandidate, Request, RepositionData


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TODO')


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """

    def __init__(self, alpha=2/(5*60), gamma=0.9, h3_resolution=8, idle_cost=-2/(60*60)):
        self.alpha = alpha
        self.gamma = gamma
        self.h3_resolution = h3_resolution
        self.idle_cost = idle_cost

        self.state_values = collections.defaultdict(int)  # Expected reward for a driver in each geohash
        self.drivers = {}
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
                update = reward + self.gamma * v1 - v0
                if update > best_update:
                    best_update, best_candidate = update, candidate

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

        self.drivers = drivers
        self.last_dispatch = dispatch
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input):
        """ Return target new positions for the given idle drivers """
        reposition = []
        data = RepositionData(reposition_input)

        driver_ids = {}
        for driver_id, position_id in data.drivers:
            if driver_id not in self.drivers:
                reposition.append(dict(driver_id=driver_id, destination=position_id))
            else:
                #driver_ids.add(driver_id)
                reposition.append(dict(driver_id=driver_id, destination=position_id))

        for driver_id in driver_ids:
            pass

        return reposition
