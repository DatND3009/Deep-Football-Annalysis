import supervision as sv
from tqdm import tqdm
import math

def extract_crop(model, vid_path, class_id, stride: int = 50):
    generator = sv.get_video_frames_generator(vid_path, stride=stride)
    vid_info = sv.VideoInfo.from_video_path(vid_path)
    
    crops = []
    
    expected_frames = math.ceil(vid_info.total_frames / stride)
    
    for frame in tqdm(generator, total=expected_frames, desc="Extracting player crops"):
        res = model(frame, device="cuda", verbose=False)[0]
        
        detections = sv.Detections.from_ultralytics(res)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == class_id]
        
        crops += [
            sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
        ]

    return crops