import json
import os
import unittest

from agent import Agent


SAMPLE_DIR = os.path.abspath('../samples')


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            cls.dispatch_observ = json.load(f)
        with open(os.path.join(SAMPLE_DIR, 'repo_observ'), 'r') as f:
            cls.repo_observ = json.load(f)

    def test_agent(self):
        agent = Agent()

        for _ in range(3):
            d = agent.dispatch(self.dispatch_observ)
            assert d
            r = agent.reposition(self.repo_observ)
            assert r