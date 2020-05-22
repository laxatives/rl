import json
import random
import os
import unittest

from agent import Agent
from parse import HEX_GRID


SAMPLE_DIR = os.path.abspath('../samples')


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            cls.dispatch_observ = json.load(f)
        with open(os.path.join(SAMPLE_DIR, 'repo_observ'), 'r') as f:
            cls.repo_observ = json.load(f)

    def test_agent_init(self):
        agent = Agent()
        for _ in range(3):
            d = agent.dispatch(self.dispatch_observ)
            assert d
            r = agent.reposition(self.repo_observ)
            assert r

    def test_agent_steady(self):
        agent = Agent()
        for grid_id in HEX_GRID.ids:
            agent.dispatcher.state_values[grid_id] = random.random()

        for _ in range(3):
            d = agent.dispatch(self.dispatch_observ)
            assert d
            r = agent.reposition(self.repo_observ)
            assert r
