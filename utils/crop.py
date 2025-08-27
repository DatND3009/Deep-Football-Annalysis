import supervision as sv

def extract_crop(model, vid_path, class_id, stride: int = 50):
  generator = sv.get_video_frames_generator(vid_path, stride = stride)
  crops = []
  for frame in generator:
    res = model(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(res)
    detections = detections.with_nms(threshold = 0.5, class_agnostic = True)
    detections = detections[detections.class_id == class_id]
    crops += [
        sv.crop_image(frame, xyxy) for xyxy in detections.xyxy
    ]

  return crops