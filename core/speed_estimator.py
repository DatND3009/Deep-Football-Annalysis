import math
from collections import defaultdict
from config.settings import SPEED_HISTORY_LENGTH

class SpeedEstimator:
    def __init__(self, fps: float):
        self.fps = fps
        self.coordinates = defaultdict(list)
        self.history_length = SPEED_HISTORY_LENGTH

    def estimate_speed(self, players_detections, pitch_players_xy):
        speed_labels = []
        pitch_players_id = players_detections.tracker_id

        for player_id, player_coordinate in zip(pitch_players_id, pitch_players_xy):
            if len(self.coordinates[player_id]) < self.history_length:
                self.coordinates[player_id].append(player_coordinate)
                speed_labels.append("")
            else:
                x_start, y_start = self.coordinates[player_id][0]
                x_end, y_end = self.coordinates[player_id][-1]
                
                dist = math.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                time = len(self.coordinates[player_id]) / self.fps
                
                speed = dist / time * 0.036 
                speed_labels.append(f"{int(speed)} km/h")
                
                self.coordinates[player_id] = self.coordinates[player_id][1:] + [player_coordinate]

        return speed_labels