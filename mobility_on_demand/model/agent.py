# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Benjamin Han (template provided by Xiaocheng Tang)
# @Date:   2020-05
import collections
import os
import random


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TODO')


class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
        """ Load your trained model and initialize the parameters """
        pass

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
    :param dispatch_observ: a list of dict, the key in the dict includes:
        order_id, int
        driver_id, int
        order_driver_distance, float
        order_start_location, a list as [lng, lat], float
        order_finish_location, a list as [lng, lat], float
        driver_location, a list as [lng, lat], float
        timestamp, int
        order_finish_timestamp, int
        day_of_week, int
        reward_units, float
        pick_up_eta, float

    :return: a list of dict, the key in the dict includes:
        order_id and driver_id, the pair indicating the assignment
    """
        dispatch_observ.sort(key=lambda od_info: od_info['reward_units'], reverse=True)

        orders = collections.OrderedDict()
        for od in dispatch_observ:
            order_id = od['order_id']
            if order_id not in orders:
                orders[order_id] = []
            orders[order_id].append(od)

        assigned_drivers = set()
        dispatch_action = []

        # Iterate highest reward first
        while orders:
            order_id, ods = orders.popitem(last=False)

            # Greedily match low pick up ETA
            sorted_candidates = sorted(ods, key=lambda x: x['pick_up_eta'])
            for candidate in sorted_candidates:
                if candidate['pick_up_eta'] > 600:
                    break

                if candidate['driver_id'] not in assigned_drivers:
                    assigned_drivers.add(candidate['driver_id'])
                    dispatch_action.append(dict(order_id=order_id, driver_id=candidate['driver_id']))
                    break

        return dispatch_action

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
    :param repo_observ: a dict, the key in the dict includes:
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
        for driver in repo_observ['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append(dict(driver_id=driver['driver_id'], destination=driver['grid_id']))
        return repo_action
