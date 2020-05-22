import json
import os
import unittest

import parse
from dispatch import Sarsa, completion_rate, MEAN_CANCEL_RATES


SAMPLE_DIR = os.path.abspath('../samples')


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.alpha = 2 / (5 * 60)
        self.gamma = 0.9
        self.idle_reward = -2 / (60 * 60)

        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            self.dispatch_observ = json.load(f)

    def test_sarsa(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = Sarsa(self.alpha, self.gamma, self.idle_reward)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    def test_dql(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = Sarsa(self.alpha, self.gamma, self.idle_reward)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    def test_cancel_rate(self):
        rate = completion_rate(0)
        assert rate < 0.03
        for i in range(10):
            distance = 200 + i * 200
            rate = completion_rate(distance)
            print(distance, rate)
            assert abs(rate - MEAN_CANCEL_RATES[i]) < 0.01, rate

        assert completion_rate(1e5) <= 1