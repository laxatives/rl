import json
import os
import unittest
from pprint import pformat

from agent import Agent


SAMPLE_DIR = os.path.abspath('../samples')


class AgentTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            self.dispatch_observ = json.load(f)
        with open(os.path.join(SAMPLE_DIR, 'repo_observ'), 'r') as f:
            self.repo_observ = json.load(f)


    def test_dispatch(self):
        for _ in range(3):
            d = Agent().dispatch(self.dispatch_observ)
            assert d
            print("Agent dispatch action:\n{}".format(pformat(d)))


    def test_reposition(self):
        for _ in range(3):
            d = Agent().dispatch(self.dispatch_observ)
            assert d
            r = Agent().reposition(self.repo_observ)
            assert r
            print("Agent reposition action:\n{}".format(pformat(r)))


class UtilsTest(unittest.TestCase):
    def test_h3(self):
        #neighbors = utils.get_neighbors('8840e3cca9fffff')
        #assert len(neighbors) == 6
        pass
