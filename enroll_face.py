import os
import pickle
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ---------- settings ----------
PERSON_NAME = "Andy"          #change for new ppl
NUM_SAMPLES = 10             
SAVE_FILE = "known_faces.pkl"
# -----------------------------

device = "cpu"

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

cap = cv2.VideoCapture(0)

embeddings = []
print("Press SPACE to capture a face sample. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read webcam frame.")
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, _ = mtcnn.detect(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        display,
        f"Samples: {len(embeddings)}/{NUM_SAMPLES}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Enroll Face", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        face_tensor = mtcnn(pil_img)
        if face_tensor is None:
            print("No face detected. Try again.")
            continue

        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]

        embeddings.append(embedding)
        print(f"Captured sample {len(embeddings)}/{NUM_SAMPLES}")

        if len(embeddings) >= NUM_SAMPLES:
            break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(embeddings) == 0:
    print("No samples collected.")
    raise SystemExit

avg_embedding = np.mean(embeddings, axis=0)

if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

known_faces[PERSON_NAME] = avg_embedding

with open(SAVE_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print(f"Saved embedding for {PERSON_NAME} to {SAVE_FILE}")