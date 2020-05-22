import json
import os
import unittest

import dispatch
import parse


SAMPLE_DIR = os.path.abspath('../samples')


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.alpha = 2 / (5 * 60)
        self.gamma = 0.9
        self.idle_reward = -2 / (60 * 60)
        self.open_request_reward = 2 / (5 * 60)

        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            self.dispatch_observ = json.load(f)

    def test_sarsa(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = dispatch.Sarsa(self.alpha, self.gamma, self.idle_reward, self.open_request_reward)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    def test_dql(self):
        drivers, requests, candidates = parse.parse_dispatch(self.dispatch_observ)
        dispatcher = dispatch.Dql(self.alpha, self.gamma, self.idle_reward, self.open_request_reward)
        for _ in range(3):
            d = dispatcher.dispatch(drivers, requests, candidates)
            assert d

    def test_cancel_rate(self):
        rate = dispatch.completion_rate(0)
        assert 0.97 < rate < 1.0, rate
        for i in range(10):
            distance = 200 + i * 200
            rate = 1 - dispatch.completion_rate(distance)
            assert abs(rate - dispatch.MEAN_CANCEL_RATES[i]) < 0.01, rate

        rate = dispatch.completion_rate(5000)
        assert 0 <= rate < 1e-6, rate
