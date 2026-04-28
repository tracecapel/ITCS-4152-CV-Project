import pickle
import torch
import numpy as np

class FaceRecognizer:
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.known_faces = pickle.load(f)

    def recognize(self, embedding, threshold):
        best_name = None
        best_score = -1

        # Ensure query is tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding)

        embedding = embedding.to("cpu")

        for name, known_embedding in self.known_faces.items():

            # ===== FIX HERE =====
            if isinstance(known_embedding, np.ndarray):
                known_embedding = torch.tensor(known_embedding)

            known_embedding = known_embedding.to("cpu")

            score = torch.cosine_similarity(
                embedding,
                known_embedding,
                dim=0
            ).item()

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return best_name, best_score

        return "Unknown", best_score