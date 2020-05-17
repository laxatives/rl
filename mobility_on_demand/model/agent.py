# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05
import os

import utils
from dispatch.dql import Dql
from utils import RepositionData


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TODO')


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, alpha=2/(5*60), gamma=0.9, h3_resolution=8, idle_reward=-2/(60*60)):
        self.h3_resolution = h3_resolution

        self.dispatcher = Dql(alpha, gamma, idle_reward)
        self.drivers = {}
        self.last_dispatch = {}
        self.last_reposition = {}

    def dispatch(self, dispatch_input):
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = utils.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)

        self.drivers = drivers
        self.last_dispatch = dispatch
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input):
        """ Return target new positions for the given idle drivers """
        reposition = []
        data = RepositionData(reposition_input)

        driver_ids = set()
        for driver_id, position_id in data.drivers:
            if driver_id not in self.drivers:
                reposition.append(dict(driver_id=driver_id, destination=position_id))
            else:
                driver_ids.add(driver_id)

        for driver_id in driver_ids:
            driver = self.drivers[driver_id]

            # TODO: consider distant neighbors, penalize using gamma and movement speed
            best_position, best_value = driver.location, self.state_values[driver.location]
            for position in utils.get_neighbors(driver.location):
                value = self.state_values[position]
                if value > best_value:
                    best_value, best_position = value, position
            # TODO: need to map these back to Didi grid_id's
            #reposition.append(dict(driver_id=driver_id, destination=best_position))
            reposition.append(dict(driver_id=driver_id, destination=driver.location))

        return reposition
