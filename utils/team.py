import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
import numpy as np
from more_itertools import chunked
from tqdm.auto import tqdm
import supervision as sv

def classify_goalkeepers(goalkeepers_detections, players_detections):
  goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
  players_0_xy = players_detections[players_detections.class_id == 0].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
  players_1_xy = players_detections[players_detections.class_id == 1].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
  players_0_centroid = players_0_xy.mean(axis = 0)
  players_1_centroid = players_1_xy.mean(axis = 0)
  team_id = []
  for goalkeeper_xy in goalkeepers_xy:
    dis0 = np.linalg.norm(goalkeeper_xy - players_0_centroid)
    dis1 = np.linalg.norm(goalkeeper_xy - players_1_centroid)
    team_id.append(2 if dis0 < dis1 else 3)

  return np.array(team_id)

class TeamClassifier():
  def __init__(self):
    model_path = "google/siglip-base-patch16-224"
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.embedding_processor = AutoProcessor.from_pretrained(model_path)
    self.embedding_model = SiglipVisionModel.from_pretrained(model_path).to(self.device)
    self.reducer = umap.UMAP(n_components = 3)
    self.cluster = KMeans(n_clusters = 2)
    self.fitted = False

  def extract_embeddings(self, crops):
    crops = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops, 32)
    data = []
    with torch.inference_mode():
      for batch in batches:
        inputs = self.embedding_processor(images = batch, return_tensors = "pt").to(self.device)
        outputs = self.embedding_model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim = 1).cpu().numpy()
        data.append(embeddings)

    return np.concatenate(data)

  def fit(self, crops):
    data = self.extract_embeddings(crops)
    projections = self.reducer.fit_transform(data)
    self.cluster.fit(projections)
    self.fitted = True

  def predict(self, crops):
    if not self.fitted:
        raise RuntimeError("You must call fit() before predict().")

    data = self.extract_embeddings(crops)
    projections = self.reducer.transform(data)
    return self.cluster.predict(projections)