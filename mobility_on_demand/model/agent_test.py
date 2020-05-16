import json
import os
import unittest
from importlib import util as import_util
from pprint import pformat


if import_util.find_spec('mobility_on_demand'):
    # Use this for development
    from mobility_on_demand.model.agent import Agent
elif import_util.find_spec('model'):
    # Use this for submission
    from model.agent import Agent
else:
    raise RuntimeError()


SAMPLE_DIR = os.path.abspath('../samples')


class AgentTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(SAMPLE_DIR, 'dispatch_observ'), 'r') as f:
            self.dispatch_observ = json.load(f)
        with open(os.path.join(SAMPLE_DIR, 'repo_observ'), 'r') as f:
            self.repo_observ = json.load(f)


    def test_dispatch(self):
        d = Agent().dispatch(self.dispatch_observ)
        assert d
        print("Agent dispatch action:\n{}".format(pformat(d)))


    def test_reposition(self):
        r = Agent().reposition(self.repo_observ)
        assert r
