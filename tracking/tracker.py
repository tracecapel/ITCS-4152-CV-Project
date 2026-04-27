import torch
import cv2

class StudentTracker:
    def __init__(self, buffer=30):
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
                self.students[id] = self.temp_tracks[id] / self.temp_frames[id]
                cv2.imwrite(f"Students/student_{id}.jpg", crop)
        else:
            self.temp_tracks[id] = embedding
            self.temp_frames[id] = 1

    def is_unique(self, embedding, threshold=0.9):
        for student in self.students.values():
            sim = float(torch.cosine_similarity(student, embedding, dim=0))
            if sim > threshold:
                return False
        return True