# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05
import os

import utils
from dispatch import Dql
from utils import RepositionData


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TODO')


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, alpha=2/(5*60), gamma=0.9, idle_reward=-2/(60*60)):
        self.dispatcher = Dql(alpha, gamma, idle_reward)
        self.last_dispatch = {}
        self.last_reposition = {}

    def dispatch(self, dispatch_input):
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = utils.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)

        self.last_dispatch = dispatch
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input):
        """ Return target new positions for the given idle drivers """
        reposition = []
        data = RepositionData(reposition_input)
        for driver_id, position_id in data.drivers:
            reposition.append(dict(driver_id=driver_id, destination=position_id))

        # TODO: Brute-force expected reward, discounted by ETA

        return reposition
