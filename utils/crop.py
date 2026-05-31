import supervision as sv

def extract_crop(model, vid_path, class_id, stride: int = 50):
  generator = sv.get_video_frames_generator(vid_path, stride = stride)
  vid_info = sv.VideoInfo.from_video_path(video_path)
  crops = []
  for i, frame in tqdm(enumerate(generator), total=vid_info.total_frames, desc="📸 Extracting player crops"):
    if max_frames and i >= max_frames:
      break
    res = model(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(res)
    detections = detections.with_nms(threshold = 0.5, class_agnostic = True)
    detections = detections[detections.class_id == class_id]
    crops += [
        sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
    ]

  return crops