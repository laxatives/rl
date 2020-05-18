import csv
import math
import os
from typing import Dict, List, Tuple


class Grid:
    def __init__(self):
        self.ids = []  # type: List[str]
        self.coords = dict()  # type: Dict[str, Tuple[float, float]]
        coords_list = []

        grid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/hexagon_grid_table.csv')
        with open(grid_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                if len(row) != 13:
                    continue
                grid_id = row[0]
                self.ids.append(grid_id)
                lng = sum([float(row[i]) for i in range(1, 13, 2)]) / 6
                lat = sum([float(row[i]) for i in range(2, 13, 2)]) / 6
                coords_list.append((lng, lat))
                self.coords[grid_id] = (lng, lat)

        assert len(coords_list) == 8518, f'Failed to initialize hex grid: found {len(coords_list)} of 8518 expected ids'

    def lookup(self, lng: float, lat: float) -> str:
        best_id, best_distance = None, 1000
        for grid_id, (grid_lng, grid_lat) in self.coords.items():
            dist = abs(lng - grid_lng) + abs(lat - grid_lat)
            if dist < best_distance:
                best_id, best_distance = grid_id, dist

        return best_id

    def distance(self, x: str, y: str) -> float:
        """ Return haversine distance in meters """
        lng_x, lat_x = self.coords[x]
        lng_y, lat_y = self.coords[y]

        # Haversine
        lng_x, lng_y, lat_x, lat_y = map(math.radians, [lng_x, lng_y, lat_x, lat_y])
        lng_delta, lat_delta = abs(lng_x - lng_y), abs(lat_x - lat_y)
        a = math.sin(lat_delta / 2) ** 2 + math.cos(lat_x) * math.cos(lat_y) * math.sin(lng_delta / 2) ** 2
        return 6371000 * 2 * math.asin(a ** 0.5)

    def cancel_prob(self, id: str) -> float:
        # TODO
        return 0
