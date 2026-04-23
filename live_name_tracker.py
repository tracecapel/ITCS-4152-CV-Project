import pickle
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ---------- settings ----------
MODE = "predict" #change to predict or track

KNOWN_FACES_FILE = "known_faces.pkl"
MATCH_THRESHOLD = 0.9  #change
RECOGNIZE_EVERY_N_FRAMES = 10
# -----------------------------

yolo_device = "mps"
face_device = "cpu"

mtcnn = MTCNN(image_size=160, margin=20, device=face_device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(face_device)

with open(KNOWN_FACES_FILE, "rb") as f:
    known_faces = pickle.load(f)

model = YOLO("yolov8s.pt")

track_to_name = {}

frame_count = 0

def recognize_face_from_crop(person_crop_bgr):
    if person_crop_bgr is None or person_crop_bgr.size == 0:
        return None

    rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None

    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(face_device)).cpu().numpy()[0]

    best_name = None
    best_dist = float("inf")

    for name, known_embedding in known_faces.items():
        dist = np.linalg.norm(embedding - known_embedding)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist < MATCH_THRESHOLD:
        return best_name

    return None


if MODE == "track":
    results = model.track(
        source=0,
        stream=True,
        classes=[0],
        conf=0.3,
        imgsz=640,
        device=yolo_device,
        persist=True,
        verbose=False
    )
else:
    results = model.predict(
        source=0,
        stream=True,
        classes=[0],
        conf=0.3,
        imgsz=640,
        device=yolo_device,
        verbose=False
    )


for r in results:
    frame_count += 1
    frame = r.orig_img.copy()

    if r.boxes is not None:

        if MODE == "track" and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()

            for box, track_id in zip(r.boxes, ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                person_crop = frame[y1:y2, x1:x2]

                if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0 or track_id not in track_to_name:
                    name = recognize_face_from_crop(person_crop)
                    if name is not None:
                        track_to_name[track_id] = name

                display_name = track_to_name.get(track_id, "person")
                label = f"id: {track_id}, {display_name} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            for i, box in enumerate(r.boxes, start=1):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                person_crop = frame[y1:y2, x1:x2]

                name = recognize_face_from_crop(person_crop)
                display_name = name if name is not None else "person"

                label = f"id: {i}, {display_name} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO + Face Name", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()