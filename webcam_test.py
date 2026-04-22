from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

results = model.track(
    source=0,
    stream=True,
    classes=[0],
    conf=0.5,
    imgsz=640,
    device="mps",
    persist=True,
    verbose=False
)

for r in results:
    frame = r.orig_img  

    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()

        for box, track_id in zip(r.boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            label = f"id: {track_id}, person ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Custom", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()