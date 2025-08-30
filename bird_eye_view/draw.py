from bird_eye_view.config import FootballPitchConfig
import supervision as sv
import numpy as np
import cv2 as cv

def draw_pitch(
    pitch_config: FootballPitchConfig,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
):
    scaled_width = int(pitch_config.width * scale)
    scaled_length = int(pitch_config.length * scale)
    scaled_penalty_spot_distance = int(pitch_config.penalty_spot_distance * scale)
    scaled_centre_circle_radius = int(pitch_config.centre_circle_radius * scale)

    pitch = np.ones(
        shape = (scaled_width + padding * 2, scaled_length + padding * 2, 3),
        dtype = np.uint8
    ) * np.array(background_color.as_bgr(), dtype = np.uint8)

    vertices = pitch_config.get_vertices()
    edges = pitch_config.get_edges()

    for edge in edges:
        start, end = edge
        start_x, start_y = int(vertices[start - 1][0] * scale) + padding, int(vertices[start - 1][1] * scale) + padding
        end_x, end_y = int(vertices[end - 1][0] * scale) + padding, int(vertices[end - 1][1] * scale) + padding
        cv.line(pitch, (int(start_x), int(start_y)), (int(end_x), int(end_y)), line_color.as_bgr(), line_thickness)

    cv.circle(pitch, (scaled_length // 2 + padding, scaled_width // 2 + padding),
                        scaled_centre_circle_radius, line_color.as_bgr(), line_thickness)
    cv.circle(pitch, (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
                        point_radius, line_color.as_bgr(), -1)
    cv.circle(pitch, (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
                        point_radius, line_color.as_bgr(), -1)

    return pitch


def project_objects_on_pitch(
    coordinates: np.array,
    color: sv.Color,
    pitch: np.array,
    padding: int = 50,
    scale: float = 0.1,
    radius: int = 10
):
  for coordinate in coordinates:
    x, y = int(coordinate[0] * scale) + padding, int(coordinate[1] * scale) + padding
    cv.circle(pitch, (x, y), radius, color.as_bgr(), thickness = -1)
    cv.circle(pitch, (x, y), radius, (0, 0, 0), thickness = 2)

  return pitch


def draw_statistic_board(
    scene,
    ball_controlling,
    color: sv.Color = sv.Color.from_hex("FFFFFF")
):
    frame = scene.copy()
    h, w = frame.shape[:2] 
    resized_w, resized_h = w // 4, h // 4
    board = np.ones((resized_h, resized_w, 3), 
                    dtype=np.uint8) * np.array(color.as_bgr(), dtype=np.uint8)

    total_controlled_frame = sum(ball_controlling.values())
    team_0_controlling = int(ball_controlling[0] / total_controlled_frame * 100)
    team_1_controlling = 100 - team_0_controlling

    cv.putText(board, "Ball Controlling", (70, 70), cv.FONT_HERSHEY_SIMPLEX,
               1.5, (0, 0, 0), 3)
    cv.putText(board, "Blue team", (30, 130), cv.FONT_HERSHEY_SIMPLEX,
               1.3, sv.Color.from_hex("#00BFFF").as_bgr(), 3)
    cv.putText(board, "Red team", (250, 130), cv.FONT_HERSHEY_SIMPLEX,
               1.3, sv.Color.from_hex("#FF1493").as_bgr(), 3)

    bar_x, bar_y = 30, 180
    bar_width = resized_w - 60
    bar_height = 30

    team_0_width = int(bar_width * (team_0_controlling / 100))
    team_1_width = bar_width - team_0_width

    cv.rectangle(board,
                 (bar_x, bar_y),
                 (bar_x + team_0_width, bar_y + bar_height),
                 sv.Color.from_hex("#00BFFF").as_bgr(), -1)
    cv.rectangle(board,
                 (bar_x + team_0_width, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 sv.Color.from_hex("#FF1493").as_bgr(), -1)
    cv.rectangle(board,
                 (bar_x, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 (0, 0, 0), 2)

    cv.putText(board, f"{team_0_controlling}%", (bar_x + 10, bar_y + bar_height - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv.putText(board, f"{team_1_controlling}%", (bar_width - 30, bar_y + bar_height - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    rect = sv.Rect(
        x=w - resized_w,
        y=0,
        width=resized_w,
        height=resized_h
    )
    frame_with_board = sv.draw_image(scene=frame, image=board, opacity=0.7, rect=rect)
    return frame_with_board