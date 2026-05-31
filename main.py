import cv2 as cv
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch

from config.settings import *
from core.speed_estimator import SpeedEstimator
from core.possession_analyzer import PossessionAnalyzer
from visualizers.annotators import VideoAnnotator

from utils.crop import extract_crop
from utils.team import TeamClassifier, classify_goalkeepers
from bird_eye_view.draw import draw_pitch, project_objects_on_pitch, draw_statistic_board
from bird_eye_view.config import FootballPitchConfig
from bird_eye_view.view import ViewTransformer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    players_model = YOLO(PLAYERS_MODEL_PATH).to(device)
    ball_model = YOLO(BALL_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_MODEL_PATH).to(device)

    vid_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    generator = sv.get_video_frames_generator(VIDEO_PATH)
    
    crops = extract_crop(players_model, VIDEO_PATH, 2)
    team_classifier = TeamClassifier()
    team_classifier.fit(crops)
    
    human_tracker = sv.ByteTrack()
    human_tracker.reset()

    annotators = VideoAnnotator(resolution=(vid_info.width, vid_info.height))
    speed_estimator = SpeedEstimator(fps=vid_info.fps)
    possession_analyzer = PossessionAnalyzer()
    pitch_config = FootballPitchConfig()

    with sv.VideoSink(OUTPUT_VIDEO_PATH, vid_info) as sink:
        for frame in generator:
            res_players = players_model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(res_players)
            
            res_ball = ball_model(frame, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(res_ball)
            
            players_detections = detections[detections.class_id == 2]
            goalkeepers_detections = detections[detections.class_id == 1]
            referees_detections = detections[detections.class_id == 3]
            
            crops = [sv.crop_image(frame, xyxy=xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(crops)
            goalkeepers_detections.class_id = classify_goalkeepers(goalkeepers_detections, players_detections)
            referees_detections.class_id -= 1
            
            all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
            all_detections = human_tracker.update_with_detections(all_detections)
            
            annotated_frame = frame.copy()
            annotated_frame = annotators.human.annotate(scene=annotated_frame, detections=all_detections)
            annotated_frame = annotators.ball.annotate(scene=annotated_frame, detections=ball_detections)
            
            pitch = draw_pitch(pitch_config)
            pitch_res = pitch_model(frame, verbose=False)[0]
            frame_keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            
            filter_kp = frame_keypoints.confidence[0] > 0.5
            frame_pts = frame_keypoints.xy[0][filter_kp]
            pitch_pts = np.array(pitch_config.get_vertices())[filter_kp]
            
            view_transformer = ViewTransformer(source=frame_pts, target=pitch_pts)
            
            frame_ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = view_transformer.transform(frame_ball_xy)
            
            frame_all_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_all_xy = view_transformer.transform(frame_all_xy)
            
            pitch = project_objects_on_pitch(coordinates=pitch_ball_xy, color=sv.Color.from_hex(COLORS["white"]), pitch=pitch)
            pitch = project_objects_on_pitch(coordinates=pitch_all_xy[all_detections.class_id == 0], color=sv.Color.from_hex(COLORS["team_1"]), pitch=pitch)
            pitch = project_objects_on_pitch(coordinates=pitch_all_xy[all_detections.class_id == 1], color=sv.Color.from_hex(COLORS["goalkeeper"]), pitch=pitch)
            pitch = project_objects_on_pitch(coordinates=pitch_all_xy[all_detections.class_id == 2], color=sv.Color.from_hex(COLORS["team_2"]), pitch=pitch)
            
            h, w = annotated_frame.shape[:2]
            resized_pitch = sv.resize_image(pitch, (w // 4, h // 4))
            rect = sv.Rect(x=0, y=0, width=resized_pitch.shape[1], height=resized_pitch.shape[0])
            demo = sv.draw_image(scene=annotated_frame, image=pitch, opacity=0.7, rect=rect)
        
            players_only = all_detections[all_detections.data["class_name"] == "player"] if "class_name" in all_detections.data else all_detections
            pitch_players_xy = pitch_all_xy[all_detections.data.get("class_name", "") == "player"] if "class_name" in all_detections.data else pitch_all_xy
            
            speed_labels = speed_estimator.estimate_speed(players_only, pitch_players_xy)
            demo = annotators.speed_label.annotate(scene=demo, detections=players_only, labels=speed_labels)
            
            ball_carrier_id = possession_analyzer.update(pitch_ball_xy, pitch_players_xy, all_detections)
            if ball_carrier_id is not None:
                carrier_detections = all_detections[all_detections.tracker_id == ball_carrier_id]
                demo = annotators.ball_carrying.annotate(scene=demo, detections=carrier_detections)
                
            demo_with_board = draw_statistic_board(scene=demo, ball_controlling=possession_analyzer.stats)
            
            sink.write_frame(demo_with_board)

if __name__ == "__main__":
    main()