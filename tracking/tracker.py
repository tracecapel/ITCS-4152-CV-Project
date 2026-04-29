import torch
import cv2

class StudentTracker:
    def __init__(self, buffer=3):
        self.students = {}
        self.temp_tracks = {}
        self.temp_frames = {}
        self.frames_elapsed = 0
        self.buffer = buffer

    def update(self, id, embedding, crop):
        # Reset temp buffers periodically
        if self.frames_elapsed > 1 and self.frames_elapsed % self.buffer == 0:
            self.temp_tracks = {}
            self.temp_frames = {}

        self.frames_elapsed += 1

        if id in self.temp_tracks:
            self.temp_tracks[id] += embedding
            self.temp_frames[id] += 1

            if self.temp_frames[id] >= self.buffer:
                avg = self.temp_tracks[id] / self.temp_frames[id]
                self.students[id] = torch.nn.functional.normalize(avg, dim=0)
                cv2.imwrite(f"Students/student_{id}.jpg", crop)
        else:
            self.temp_tracks[id] = embedding
            self.temp_frames[id] = 1

    #now embeddings are treated in their own respective spaces
    def is_unique(self, embedding, threshold=0.9):
        apppearence_embedding = embedding[:512]
        face_embedding = embedding[512:]

        for student in self.students.values():
            sim_appearance = float(torch.cosine_similarity(student[:512], apppearence_embedding, dim=0))
            sim_face = float(torch.cosine_similarity(student[512:], face_embedding, dim=0))

            if sim_appearance > threshold and sim_face > threshold:
                return False
        return True