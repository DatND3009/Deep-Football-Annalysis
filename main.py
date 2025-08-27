from ultralytics import YOLO
import supervision as sv
import cv2 as cv
from utils.crop import extract_crop
from utils.team import TeamClassifier, classify_goalkeepers
from bird_eye_view.draw import draw_pitch, project_objects_on_pitch
from bird_eye_view.config import FootballPitchConfig
from bird_eye_view.view import ViewTransformer
import numpy as np
from collections import defaultdict
import math

def main():
    PLAYERS_MODEL = YOLO("models/best_players.pt")
    BALL_MODEL = YOLO("models/best_ball.pt")
    PITCH_MODEL = YOLO("models/best_pitch.pt")

    generator = sv.get_video_frames_generator("football.mp4")
    vid_info = sv.VideoInfo.from_video_path("football.mp4")
    fps = vid_info.fps

    crops = extract_crop(PLAYERS_MODEL, "football.mp4", 2)
    team_classifier = TeamClassifier()
    team_classifier.fit(crops)
    human_tracker = sv.ByteTrack()
    human_tracker.reset()

    thickness = sv.calculate_optimal_line_thickness((vid_info.width, vid_info.height))
    text_scale = sv.calculate_optimal_text_scale((vid_info.width, vid_info.height))

    # human and ball detection
    human_annotator = sv.EllipseAnnotator(
        color = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness = thickness
    )
    ball_annotator = sv.CircleAnnotator(
        color = sv.Color.from_hex("FF0000"),
        thickness = thickness + 1
    )
    ball_carrying_annotator = sv.TriangleAnnotator(
        color = sv.Color.from_hex("FF0000"),
        base = 20,
        height = 15
    )
    id_label_annotator = sv.LabelAnnotator(
        color = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color = sv.Color.from_hex('#000000'),
        text_scale = text_scale / 3,
        text_thickness = thickness // 3,
        text_position = sv.Position.TOP_CENTER
    )

    players_distance = defaultdict(lambda: [])

    with sv.VideoSink("demo_out.mp4", vid_info) as sink:
        for idx, frame in enumerate(generator):
            if idx == 200:
                break
            res = PLAYERS_MODEL(frame, verbose = False)[0]
            detections = sv.Detections.from_ultralytics(res)
            human_detections = detections[detections.class_id != 0]
            human_detections = human_detections.with_nms(threshold = 0.5, class_agnostic = True)
            
            res_ball = BALL_MODEL(frame, verbose = False)[0]
            ball_detections = sv.Detections.from_ultralytics(res_ball)
            
            players_detections = detections[detections.class_id == 2]
            goalkeepers_detections = detections[detections.class_id == 1]
            referees_detections = detections[detections.class_id == 3]
            
            crops = [sv.crop_image(frame, xyxy = xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(crops)
            
            goalkeepers_detections.class_id = classify_goalkeepers(goalkeepers_detections, players_detections)
            referees_detections.class_id -= 1
            
            all_detections = sv.Detections.merge(
                [
                    players_detections, goalkeepers_detections, referees_detections
                ]
            )
            all_detections = human_tracker.update_with_detections(all_detections)
            
            labels = [
                f"#{track_id}" for track_id in all_detections.tracker_id
            ]
            
            annotated_frame = frame.copy()
            annotated_frame = human_annotator.annotate(scene = annotated_frame, detections = all_detections)
            annotated_frame = ball_annotator.annotate(scene = annotated_frame, detections = ball_detections)
            annotated_frame = id_label_annotator.annotate(scene = annotated_frame, detections = all_detections, labels = labels)
            
            # draw pitch and project objects onto it
            pitch_config = FootballPitchConfig()
            pitch = draw_pitch(pitch_config)
            
            pitch_res = PITCH_MODEL(frame, verbose = False)[0]
            frame_keypoints_detections = sv.KeyPoints.from_ultralytics(pitch_res)
            filter = frame_keypoints_detections.confidence[0] > 0.5
            frame_keypoints_filtered_coordinates = frame_keypoints_detections.xy[0][filter]
            pitch_keypoints_filtered_coordinates = np.array(pitch_config.get_vertices())[filter]
            
            view_transformer = ViewTransformer(
                source = frame_keypoints_filtered_coordinates,
                target = pitch_keypoints_filtered_coordinates
            )
            
            frame_ball_xy = ball_detections.get_anchors_coordinates(anchor = sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = view_transformer.transform(frame_ball_xy)
            
            frame_all_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_all_xy = view_transformer.transform(frame_all_xy)
            
            pitch = project_objects_on_pitch(
                    coordinates = pitch_ball_xy,
                    color = sv.Color.from_hex("#FFFFFF"),
                    pitch = pitch
            )
            pitch = project_objects_on_pitch(
                    coordinates = pitch_all_xy[all_detections.class_id == 0],
                    color = sv.Color.from_hex("#00BFFF"),
                    pitch = pitch
            )
            pitch = project_objects_on_pitch(
                    coordinates = pitch_all_xy[all_detections.class_id == 1],
                    color = sv.Color.from_hex("#FF1493"),
                    pitch = pitch
            )
            pitch = project_objects_on_pitch(
                    coordinates = pitch_all_xy[all_detections.class_id == 2],
                    color = sv.Color.from_hex("#FFD700"),
                    pitch = pitch
            )
            
            if len(pitch_ball_xy) == 1:
                ball_carrying_id = None
                cur_min_dist = 1000000
                players_detections = all_detections[all_detections.class_id != 2]
                for coordinate, track_id in zip(pitch_all_xy[all_detections.class_id != 2], players_detections.tracker_id):
                    player_x, player_y = coordinate
                    ball_x, ball_y = pitch_ball_xy[0]
                    dist = math.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                    players_distance[track_id] = dist
                    if dist < cur_min_dist:
                        cur_min_dist = dist
                        if dist < 300:
                            ball_carrying_id = track_id
            
                if ball_carrying_id is not None:
                    carrying_detections = players_detections[players_detections.tracker_id == ball_carrying_id]
                    annotated_frame = ball_carrying_annotator.annotate(
                        scene = annotated_frame,
                        detections = carrying_detections
                    )
            
            h, w = annotated_frame.shape[:2]
            resized_pitch = sv.resize_image(pitch, (w // 4, h // 4))
            resized_pitch_h, resized_pitch_w = resized_pitch.shape[:2]
            rect = sv.Rect(
                x = 0,
                y = 0,
                width = resized_pitch_w,
                height = resized_pitch_h
            )
            demo = sv.draw_image(scene = annotated_frame, image = pitch, opacity = 0.7, rect = rect)
            # sv.plot_image(demo, (16, 16))
            sink.write_frame(demo)

if __name__ == "__main__":
    main()