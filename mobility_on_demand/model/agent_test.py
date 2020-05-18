import json
import os
import unittest

from agent import Agent


SAMPLE_DIR = os.path.abspath('../samples')


class AgentTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            self.dispatch_observ = json.load(f)
        with open(os.path.join(SAMPLE_DIR, 'repo_observ'), 'r') as f:
            self.repo_observ = json.load(f)


    def test_agent(self):
        for _ in range(10):
            d = Agent().dispatch(self.dispatch_observ)
            assert d
            r = Agent().reposition(self.repo_observ)
            assert r
