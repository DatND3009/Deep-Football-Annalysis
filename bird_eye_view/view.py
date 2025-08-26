import numpy as np 
import cv2 as cv

class ViewTransformer():
  def __init__(self, source, target):
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    self.matrix, _ = cv.findHomography(source, target)

  def transform(self, points):
    if points.size == 0:
      return points
    points = points.astype(np.float32).reshape(-1, 1, 2)
    transformed_points = cv.perspectiveTransform(points, self.matrix)
    return transformed_points.reshape(-1, 2).astype(np.int64)