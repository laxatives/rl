import unittest

from grid import Grid


class GridTest(unittest.TestCase):
    def setUp(self):
        self.grid = Grid()

    def test_hex(self):
        grid_id = self.grid.lookup(104.50, 30.71)
        assert grid_id == '5973d1e3fdf1f878', grid_id
        grid_id = self.grid.lookup(103.99, 30.40)
        assert grid_id == '27471aff3df268a1', grid_id
        grid_id = self.grid.lookup(104.52, 30.37)
        assert grid_id == '15948343c6223064', grid_id

    def test_distance(self):
        # SE: (30.65924666666667, 104.12614), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('386c78bc3c226d88', '926d27c14e84f5d0')
        expected = 11126
        error = abs(distance - expected)
        assert error < 100, distance

        # NE: (30.725296666666665, 104.13031000000001), NW: (30.73054666666667, 104.04442)
        distance = self.grid.distance('111297464a0c9cc8', '926d27c14e84f5d0')
        expected = 8247
        error = abs(distance - expected)
        assert error < 100, distance

        # NE: (30.725296666666665, 104.13031000000001), SE: (30.65924666666667, 104.12614)
        distance = self.grid.distance('111297464a0c9cc8', '386c78bc3c226d88')
        expected = 7333
        error = abs(distance - expected)
        assert error < 100, distance
