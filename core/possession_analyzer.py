import math
from collections import defaultdict
from config.settings import POSSESSION_MIN_DIST

class PossessionAnalyzer:
    def __init__(self):
        self.stats = defaultdict(lambda: 0)

    def update(self, pitch_ball_xy, pitch_players_xy, players_detections):
        if len(pitch_ball_xy) != 1:
            return None

        ball_carrying_id = None
        ball_carrying_class_id = None
        cur_min_dist = float('inf')
        
        valid_players = players_detections[players_detections.class_id != 2]
        
        for coordinate, track_id, class_id in zip(pitch_players_xy, 
                                                  valid_players.tracker_id,
                                                  valid_players.class_id):
            player_x, player_y = coordinate
            ball_x, ball_y = pitch_ball_xy[0]
            
            dist = math.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
            if dist < cur_min_dist:
                cur_min_dist = dist
                ball_carrying_id = track_id
                ball_carrying_class_id = class_id

        if ball_carrying_class_id is not None and cur_min_dist < POSSESSION_MIN_DIST:
            self.stats[ball_carrying_class_id] += 1
            return ball_carrying_id
            
        return None