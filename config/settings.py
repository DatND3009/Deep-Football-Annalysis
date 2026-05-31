import supervision as sv

VIDEO_PATH = "assets/football.mp4"
OUTPUT_VIDEO_PATH = "assets/demo_out.mp4"

PLAYERS_MODEL_PATH = "models/best_players.pt"
BALL_MODEL_PATH = "models/best_ball.pt"
PITCH_MODEL_PATH = "models/best_pitch.pt"

COLORS = {
    "team_1": '#00BFFF',
    "team_2": '#FFD700',
    "goalkeeper": '#FF1493',
    "ball": "#FF0000",
    "text": "#000000",
    "white": "#FFFFFF"
}

NMS_THRESHOLD = 0.5
POSSESSION_MIN_DIST = 300
SPEED_HISTORY_LENGTH = 10