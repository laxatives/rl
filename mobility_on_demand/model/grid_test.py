import unittest

from grid import Grid


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = Grid()

    def test_hex(self):
        grid_id = self.grid.lookup(104.50, 30.71)
        assert grid_id == '5973d1e3fdf1f878', grid_id
        grid_id = self.grid.lookup(103.99, 30.40)
        assert grid_id == '27471aff3df268a1', grid_id
        grid_id = self.grid.lookup(104.52, 30.37)
        assert grid_id == '15948343c6223064', grid_id

    def test_distance_fast(self):
        # SE: (30.65924666666667, 104.12614), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('386c78bc3c226d88', '926d27c14e84f5d0')
        expected = 11126
        error = abs(distance - expected)
        assert error < 2000, distance

        # NE: (30.725296666666665, 104.13031000000001), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('111297464a0c9cc8', '926d27c14e84f5d0')
        expected = 8247
        error = abs(distance - expected)
        assert error < 2000, distance

        # NE: (30.725296666666665, 104.13031000000001), SE: (30.65924666666667, 104.12614)
        distance = self.grid.distance('111297464a0c9cc8', '386c78bc3c226d88')
        expected = 7333
        error = abs(distance - expected)
        assert error < 2000, distance

    def test_distance_haversine(self):
        # SE: (30.65924666666667, 104.12614), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('386c78bc3c226d88', '926d27c14e84f5d0', fast=False)
        expected = 11126
        error = abs(distance - expected)
        assert error < 100, distance

        # NE: (30.725296666666665, 104.13031000000001), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('111297464a0c9cc8', '926d27c14e84f5d0', fast=False)
        expected = 8247
        error = abs(distance - expected)
        assert error < 100, distance

        # NE: (30.725296666666665, 104.13031000000001), SE: (30.65924666666667, 104.12614)
        distance = self.grid.distance('111297464a0c9cc8', '386c78bc3c226d88', fast=False)
        expected = 7333
        error = abs(distance - expected)
        assert error < 100, distance

    def test_idle_transition(self):
        transitions = self.grid.idle_transitions(148865000, '79365a623250931c')
        assert abs(transitions['d5798236d9cf3f65'] - 0.043478260869565216) < 1e-9, transitions['d5798236d9cf3f65']
        assert abs(transitions['45b05a52ebc86721'] - 0.043478260869565216) < 1e-9, transitions['45b05a52ebc86721']
        assert abs(sum(transitions.values()) - 1.0) < 1e-9