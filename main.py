from ultralytics import YOLO
import supervision as sv 
import cv2 as cv 
from utils.crop import extract_crop
from utils.team import TeamClassifier, classify_goalkeepers
from bird_eye_view.draw import draw_pitch, project_objects_on_pitch
from bird_eye_view.config import FootballPitchConfig
from bird_eye_view.view import ViewTransformer
import numpy as np

def main():
    MODEL = YOLO("models/best_players.pt")
    PITCH_MODEL = YOLO("models/best_pitch.pt")

    generator = sv.get_video_frames_generator("football.mp4")
    vid_info = sv.VideoInfo.from_video_path("football.mp4")

    crops = extract_crop(MODEL, "football.mp4", 2) # id: ball - 0, goalkeepers - 1, players - 2, referees - 3
    team_classifier = TeamClassifier()
    team_classifier.fit(crops)

    thickness = sv.calculate_optimal_line_thickness((vid_info.width, vid_info.height))
    text_scale = sv.calculate_optimal_text_scale((vid_info.width, vid_info.height))

    frame = next(iter(generator))

    # human and ball detection
    human_annotator = sv.EllipseAnnotator(
        color = sv.ColorPalette.from_hex(['#90ee90', '#ffffff', '#00BFFF', '#FF1493', '#FFD700']),
        thickness = thickness
    )
    ball_annotator = sv.TriangleAnnotator(
        color = sv.ColorPalette.from_hex(["#FF8C00"]),
        height = 30,
        base = 30
    )
    label_annotator = sv.LabelAnnotator(
        color = sv.ColorPalette.from_hex(['#90ee90', '#ffffff', '#00BFFF', '#FF1493', '#FFD700']),
        text_color = sv.Color.from_hex('#000000'),
        text_scale = text_scale / 2,
        text_thickness = int(thickness / 3),
        text_position = sv.Position.BOTTOM_CENTER
    )

    res = MODEL(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(res)

    human_detections = detections[detections.class_id != 0]
    ball_detections = detections[detections.class_id == 0]

    human_detections = human_detections.with_nms(threshold = 0.5, class_agnostic = True)
    ball_detections.xyxy = sv.pad_boxes(xyxy = ball_detections.xyxy, px = 10, py = 10)

    players_detections = detections[detections.class_id == 2]
    goalkeepers_detections = detections[detections.class_id == 1]
    referees_detections = detections[detections.class_id == 3]

    crops = [sv.crop_image(frame, xyxy = xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(crops)

    goalkeepers_detections.class_id = classify_goalkeepers(goalkeepers_detections, players_detections)
    referees_detections.class_id += 1

    all_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections, referees_detections
    ])

    labels = [
        f"{class_name}" for class_name in all_detections.data["class_name"]
    ]

    annotated_frame = frame.copy()
    annotated_frame = human_annotator.annotate(scene = annotated_frame, detections = all_detections)
    annotated_frame = ball_annotator.annotate(scene = annotated_frame, detections = ball_detections)
    annotated_frame = label_annotator.annotate(scene = annotated_frame, detections = all_detections, labels = labels)

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

    frame_players_0_xy = players_detections[players_detections.class_id == 0].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_1_xy = players_detections[players_detections.class_id == 1].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_0_xy = view_transformer.transform(frame_players_0_xy)
    pitch_players_1_xy = view_transformer.transform(frame_players_1_xy)

    frame_goalkeepers_0_xy = goalkeepers_detections[goalkeepers_detections.class_id == 2].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_goalkeepers_1_xy = goalkeepers_detections[goalkeepers_detections.class_id == 3].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_goalkeepers_0_xy = view_transformer.transform(frame_goalkeepers_0_xy)
    pitch_goalkeepers_1_xy = view_transformer.transform(frame_goalkeepers_1_xy)

    frame_referees_xy = referees_detections.get_anchors_coordinates(anchor = sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = view_transformer.transform(frame_referees_xy)

    colors = ["#FF8C00", '#90ee90', '#ffffff', '#00BFFF', '#FF1493', '#FFD700']
    labels = ["ball", "players_0", "players_1", "goalkeepers_0", "goalkeepers_2", "referees"]
    coordinates = [pitch_ball_xy, pitch_players_0_xy, pitch_players_1_xy, pitch_goalkeepers_0_xy, pitch_goalkeepers_1_xy, pitch_referees_xy]

    for color, coordinate in zip(colors, coordinates):
        color = sv.Color.from_hex(color)
        pitch = project_objects_on_pitch(
            coordinates = coordinate,
            color = color,
            pitch = pitch
        )

    h, w = annotated_frame.shape[:2]
    resized_pitch = sv.resize_image(pitch, (w // 3, h // 3))
    resized_pitch_h, resized_pitch_w = resized_pitch.shape[:2]
    rect = sv.Rect(
        x = w // 2 - resized_pitch_w // 2,
        y = h - resized_pitch_h,
        width = resized_pitch_w,
        height = resized_pitch_h
    )
    demo = sv.draw_image(scene = annotated_frame, image = pitch, opacity = 0.5, rect = rect)
    cv.imshow("f", demo)
    cv.waitKey(0)

if __name__ == "__main__":
    main()