import csv
import math
import os
import time
from typing import Dict, List, Tuple


LNG_FACTOR = 0.685  # Assume latitude ~30.6


class Grid:
    def __init__(self):
        self.ids = []  # type: List[str]
        self.coords = dict()  # type: Dict[str, Tuple[float, float]]
        self.transitions = dict()  # type: Dict[int, Dict[start_grid_id, Dict[str, float]]

        grid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hexagon_grid_table.csv')
        with open(grid_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                if len(row) != 13:
                    continue
                grid_id = row[0]
                self.ids.append(grid_id)

                # Use centroid for simplicity
                lng = sum([float(row[i]) for i in range(1, 13, 2)]) / 6
                lat = sum([float(row[i]) for i in range(2, 13, 2)]) / 6
                self.coords[grid_id] = (lng, lat)
        assert len(self.coords) == 8518

        transitions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'idle_transition_probability.csv')
        with open(transitions_path, 'r') as csvfile:
            for row in csv.reader(csvfile):
                # TODO: verify hour in GMT
                hour, start_grid_id, end_grid_id, probability = row
                hour = int(hour)
                if hour not in self.transitions:
                    self.transitions[hour] = dict()

                hour_dict = self.transitions[hour]
                if start_grid_id not in hour_dict:
                    hour_dict[start_grid_id] = dict()

                start_dict = hour_dict[start_grid_id]
                if end_grid_id not in start_dict:
                    start_dict[end_grid_id] = float(probability)
        assert len(self.transitions) == 24


    def lookup(self, lng: float, lat: float) -> str:
        best_id, best_distance = None, 1e12
        for grid_id, (grid_lng, grid_lat) in self.coords.items():
            dist = LNG_FACTOR * abs(lng - grid_lng) + abs(lat - grid_lat)
            if dist < best_distance:
                best_id, best_distance = grid_id, dist

        return best_id

    def distance(self, x: str, y: str) -> float:
        """ Return haversine distance in meters """
        if x not in self.coords or y not in self.coords:
            return 1e12

        lng_x, lat_x = self.coords[x]
        lng_y, lat_y = self.coords[y]

        # Haversine
        lng_x, lng_y, lat_x, lat_y = map(math.radians, [lng_x, lng_y, lat_x, lat_y])
        lng_delta, lat_delta = abs(lng_x - lng_y), abs(lat_x - lat_y)
        a = math.sin(lat_delta / 2) ** 2 + math.cos(lat_x) * math.cos(lat_y) * math.sin(lng_delta / 2) ** 2
        return 6371000 * 2 * math.asin(a ** 0.5)

    def idle_transitions(self, timestamp: int, start_grid_id: str) -> Dict[str, float]:
        hour = time.gmtime(timestamp).tm_hour
        if hour in self.transitions and start_grid_id in self.transitions[hour]:
            return self.transitions[hour][start_grid_id]
        return {start_grid_id: 1.}
