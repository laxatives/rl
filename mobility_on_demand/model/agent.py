# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05-16
from typing import Any, List, Dict

import dispatch as dispatcher
import parse


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, alpha=0.04, gamma=0.999, idle_reward=-2/(60*60)):
        self.dispatcher = dispatcher.Sarsa(alpha, gamma, idle_reward)

    def dispatch(self, dispatch_input: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = parse.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input: Dict[str, Any]) -> List[Dict[str, str]]:
        """ Return target new positions for the given idle drivers """
        reposition = []  # type: List[Dict[str, str]]
        data = parse.RepositionData(reposition_input)
        for driver_id, position_id in data.drivers:
            reposition.append(dict(driver_id=driver_id, destination=position_id))
        return reposition