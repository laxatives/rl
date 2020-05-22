# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05-16
from typing import List, Dict

import dispatch as dispatcher
import parse
import reposition


class Agent:
    """ Agent for dispatching and repositioning drivers for the 2020 ACM SIGKDD Cup Competition """
    def __init__(self, d=False, a=0.0067, g=0.9999, ir=0):
        self.dispatcher = dispatcher.Dql(a, g, ir) if d \
            else dispatcher.Sarsa(a, g, ir)
        self.repositioner = reposition.StateValueGreedy(self.dispatcher, g)

    def dispatch(self, dispatch_input) -> List[Dict[str, str]]:
        """ Compute the assignment between drivers and passengers at each time step """
        drivers, requests, candidates = parse.parse_dispatch(dispatch_input)
        dispatch = self.dispatcher.dispatch(drivers, requests, candidates)
        return [dict(order_id=order_id, driver_id=d.driver_id) for order_id, d in dispatch.items()]

    def reposition(self, reposition_input) -> List[Dict[str, str]]:
        """ Return target new positions for the given idle drivers """
        data = parse.RepositionData(reposition_input)
        if not data.drivers:
            return []
        return self.repositioner.reposition(data)
