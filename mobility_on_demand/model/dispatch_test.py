import json
import os
import unittest

import dispatch
import parse


SAMPLE_DIR = os.path.abspath('../samples')


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.alpha = 2 / (5 * 60)
        cls.gamma = 0.9
        cls.idle_reward = -2 / (60 * 60)
        cls.missed_request = 0.001

        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            cls.dispatch_observ = json.load(f)

    def test_sarsa(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = dispatch.Sarsa(self.alpha, self.gamma, self.idle_reward, self.missed_request)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    def test_dql(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = dispatch.Dql(self.alpha, self.gamma, self.idle_reward, self.missed_request)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    @staticmethod
    def test_cancel_rate():
        rate = dispatch.completion_rate(0)
        assert 0.97 < rate < 1.0, rate
        for i in range(10):
            distance = 200 + i * 200
            rate = 1 - dispatch.completion_rate(distance)
            assert abs(rate - dispatch.MEAN_CANCEL_RATES[i]) < 0.01, rate

        rate = dispatch.completion_rate(5000)
        assert 0 <= rate < 1e-6, rate
