import torch
# from transformers import AutoProcessor, SiglipVisionModel
# import umap
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
    team_id.append(0 if dis0 < dis1 else 1)

  return np.array(team_id)

# class TeamClassifier: # Siglip + UMAP + KMeans (reliable but slower)
#   def __init__(self):
#     model_path = "google/siglip-base-patch16-224"
#     self.device = "cuda" if torch.cuda.is_available() else "cpu"
#     self.embedding_processor = AutoProcessor.from_pretrained(model_path)
#     self.embedding_model = SiglipVisionModel.from_pretrained(model_path).to(self.device)
#     self.reducer = umap.UMAP(n_components = 3)
#     self.cluster = KMeans(n_clusters = 2)
#     self.fitted = False

#   def extract_embeddings(self, crops):
#     crops = [sv.cv2_to_pillow(crop) for crop in crops]
#     batches = chunked(crops, 32)
#     data = []
#     with torch.inference_mode():
#       for batch in batches:
#         inputs = self.embedding_processor(images = batch, return_tensors = "pt").to(self.device)
#         outputs = self.embedding_model(**inputs)
#         embeddings = torch.mean(outputs.last_hidden_state, dim = 1).cpu().numpy()
#         data.append(embeddings)

#     return np.concatenate(data)

#   def fit(self, crops):
#     data = self.extract_embeddings(crops)
#     projections = self.reducer.fit_transform(data)
#     self.cluster.fit(projections)
#     self.fitted = True

#   def predict(self, crops):
#     if not self.fitted:
#         raise RuntimeError("You must call fit() before predict().")

#     data = self.extract_embeddings(crops)
#     projections = self.reducer.transform(data)
#     return self.cluster.predict(projections)

class TeamClassifier: # KMeans only (faster + good even with CPU)
    def __init__(self):
        self.kmeans = KMeans(n_clusters = 2, n_init = 10)
        self.fitted = False

    def get_player_color(self, crop_image):
        crop = crop_image.copy()
        top_half = crop[0:(crop.shape[0] // 2), :]
        top_half_2d = top_half.reshape(-1, 3)

        kmeans = KMeans(n_clusters = 2, n_init = 1)
        kmeans.fit(top_half_2d)

        labels_2d = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corners = [
            labels_2d[0, 0], labels_2d[0, -1], labels_2d[-1, 0], labels_2d[-1, -1]
        ]
        non_player_value = max(set(corners), key = corners.count)
        player_value = 1 - non_player_value

        player_color = kmeans.cluster_centers_[player_value]
        return player_color

    def fit(self, crops):
        player_colors = []
        for crop in crops:
            player_color = self.get_player_color(crop)
            player_colors.append(player_color)

        self.kmeans.fit(player_colors)
        self.fitted = True

    def predict(self, crops):
        if not self.fitted:
            raise RuntimeError("You haven't fitted yet!")

        if isinstance(crops, np.ndarray):
            player_color = self.get_player_color(crops)
            return self.kmeans.predict(player_color.reshape(1, -1))[0]

        elif isinstance(crops, list):
            results = []
            for crop in crops:
                player_color = self.get_player_color(crop)
                results.append(self.kmeans.predict(player_color.reshape(1, -1))[0])
            return np.array(results)