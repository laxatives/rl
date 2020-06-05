# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05-16
from typing import Any, List, Dict

import approximation
import parse
import reposition as repositioner


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, alpha=2/(5*60), dispatch_gamma=0.9, reposition_gamma=0.9997):
        self.dispatcher = approximation.LinearApprox(alpha, dispatch_gamma)
        self.repositioner = repositioner.StateValueGreedy(self.dispatcher, reposition_gamma)

    def dispatch(self, dispatch_input: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = parse.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)
        return [dict(order_id=order_id, driver_id=driver_id) for order_id, driver_id in dispatch.items()]

    def reposition(self, reposition_input: Dict[str, Any]) -> List[Dict[str, str]]:
        """ Return target new positions for the given idle drivers """
        data = parse.RepositionData(reposition_input)
        return [dict(driver_id=driver_id, destination=grid_id) for driver_id, grid_id in data.drivers]
        #return self.repositioner.reposition(data)