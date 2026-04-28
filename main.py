import argparse
from pathlib import Path
from typing import Optional

import os
import cv2
import torch
from ultralytics import YOLO

# ===================== OUR MODULES =====================
from tracking.tracker import StudentTracker
from detectors.pose_rule_based import PoseRuleBasedDetector
from recognition.face_recognizer import FaceRecognizer

# ===================== FACE MODEL =====================
from facenet_pytorch import MTCNN, InceptionResnetV1

# ===================== APPEARANCE MODEL =====================
from torchreid.reid.utils import feature_extractor


def parse_args():
    """
Parse
command - line
arguments
"""
    parser = argparse.ArgumentParser(
        description="Vision Based Attendance System - Student tracking with face recognition and hand detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to input video file or camera index (0, 1, etc.)"
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output video file (default: output/[source]_annotated.mp4)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video, only display"
    )

    # Display arguments
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Show live video window (cv2.imshow)"
    )

    parser.add_argument(
        "--window-name",
        type=str,
        default="Classroom Monitor",
        help="Name of the display window"
    )

    # Model paths
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo26n.pt",
        help="Path to YOLO detection model"
    )

    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo26n-pose.pt",
        help="Path to YOLO pose estimation model"
    )

    parser.add_argument(
        "--known-faces",
        type=str,
        default="models/known_faces.pkl",
        help="Path to known faces pickle file"
    )

    parser.add_argument(
        "--appearance-model",
        type=str,
        default="osnet_x0_75_imagenet.pth",
        help="Path to appearance feature extractor model"
    )

    # Recognition thresholds
    parser.add_argument(
        "--face-confidence", "-fc",
        type=float,
        default=0.6,
        help="Face recognition confidence threshold (0.0-1.0)"
    )

    # Hybrid embedding weights
    parser.add_argument(
        "--appearance-weight",
        type=float,
        default=0.6,
        help="Weight for appearance embedding in hybrid (0.0-1.0)"
    )

    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.4,
        help="Weight for face embedding in hybrid (0.0-1.0)"
    )

    # Video settings
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output video FPS (default: same as input)"
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use for inference (default: auto-detect)"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """
Main
execution
function
"""
    args = parse_args()

    # ===================== DEVICE SETUP =====================
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.verbose:
        print(f"[INFO] Using device: {device}")

    # ===================== VALIDATE INPUTS =====================
    source_path = Path(args.source)
    if not source_path.exists() and not args.source.isdigit():
        print(f"[ERROR] Source not found: {args.source}")
        return 1

    # ===================== OUTPUT SETUP =====================
    if args.output:
        output_video_path = Path(args.output)
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        if args.source.isdigit():
            output_video_path = output_dir / f"camera_{args.source}_annotated.mp4"
        else:
            output_video_path = output_dir / (source_path.stem + "_annotated.mp4")

    # ===================== LOAD MODELS =====================
    if args.verbose:
        print(f"[INFO] Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    if args.verbose:
        print(f"[INFO] Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)

    if args.verbose:
        print(f"[INFO] Loading face models...")
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    if args.verbose:
        print(f"[INFO] Loading appearance model: {args.appearance_model}")
    image_feature_extractor = feature_extractor.FeatureExtractor(
        model_name="osnet_x0_75",
        model_path=args.appearance_model,
        device="cpu"  # keep CPU because known_faces.pkl is numpy
    )

    # ===================== INITIALIZE SYSTEMS =====================
    if args.verbose:
        print(f"[INFO] Initializing tracker and recognizer...")
    tracker = StudentTracker()
    detector = PoseRuleBasedDetector(pose_model)
    recognizer = FaceRecognizer(args.known_faces)

    # ===================== GET VIDEO FPS =====================
    cap = cv2.VideoCapture(args.source)
    if args.fps:
        fps = args.fps
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0
    cap.release()

    if args.verbose:
        print(f"[INFO] Video FPS: {fps:.2f}")

    # ===================== VIDEO WRITER INITIALIZATION =====================
    writer: Optional[cv2.VideoWriter] = None

    # ===================== VALIDATE WEIGHTS =====================
    if args.appearance_weight < 0:
        print(f"[WARNING] Appearance weight is negative and will be est to 0 automatically")
        args.appearance_weight = 0
    if args.face_weight < 0:
        print(f"[WARNING] Face weight is negative and will be est to 0 automatically")
        args.face_weight = 0
    if abs(args.appearance_weight + args.face_weight - 1.0) > 0.01:
        print(f"[WARNING] Weights don't sum to 1.0: appearance={args.appearance_weight}, face={args.face_weight}")
        print(f"[WARNING] Results will be normalized automatically")

    # ===================== MAIN LOOP =====================
    print(f"[INFO] Starting processing...")
    print(f"[INFO] Source: {args.source}")
    if not args.no_save:
        print(f"[INFO] Output: {output_video_path}")
    if args.display:
        print(f"[INFO] Display enabled (press 'q' to quit)")

    frame_count = 0

    try:
        for r in model.track(source=args.source, stream=True, classes=args.classes):
            frame = r.orig_img
            frame_count += 1

            # Initialize writer once we know frame size
            if writer is None and not args.no_save:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    fps,
                    (w, h)
                )

                if args.verbose:
                    print(f"[INFO] Video writer initialized: {w}x{h} @ {fps:.2f} FPS")

            # ===================== PROCESS DETECTIONS =====================
            if r.boxes is not None:
                for box in r.boxes:
                    if box is None or box.id is None:
                        continue

                    track_id = int(box.id.item())

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    # ===================== FACE EMBEDDING =====================
                    face_tensor = mtcnn(crop)

                    if face_tensor is None:
                        face_embedding = torch.zeros(512, device=device)
                    else:
                        face_tensor = face_tensor.to(device)

                        with torch.no_grad():
                            face_embedding = resnet(face_tensor.unsqueeze(0)).squeeze(0)

                    # ===================== APPEARANCE EMBEDDING =====================
                    appearance_embedding = image_feature_extractor(crop).squeeze(0)

                    # ===================== MOVE TO SAME DEVICE =====================
                    appearance_embedding = appearance_embedding.to(device)
                    face_embedding = face_embedding.to(device)

                    # ===================== HYBRID EMBEDDING =====================
                    hybrid_embedding = torch.cat(
                        (appearance_embedding * args.appearance_weight,
                         face_embedding * args.face_weight),
                        dim=0
                    )
                    hybrid_embedding = torch.nn.functional.normalize(hybrid_embedding, dim=0)

                    # ===================== TRACKING =====================
                    if tracker.is_unique(hybrid_embedding):
                        tracker.update(track_id, hybrid_embedding, crop)

                    # ===================== RECOGNITION =====================
                    name, score = recognizer.recognize(face_embedding, args.face_confidence)

                    # ===================== HAND DETECTION =====================
                    result = detector.detect(crop)

                    # ===================== LABEL =====================
                    label = f"{name} ({score:.2f})"

                    if result["left"] or result["right"]:
                        label += " | HAND RAISED"

                    # ===================== DRAW =====================
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

            # ===================== SHOW + SAVE =====================
            if args.display:
                cv2.imshow(args.window_name, frame)

            if writer is not None:
                writer.write(frame)

            # Verbose progress
            if args.verbose and frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count} frames...")

            # Check for quit
            if args.display and cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User requested quit")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    finally:
        # ===================== CLEANUP =====================
        if args.display:
            cv2.destroyAllWindows()

        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved to: {output_video_path}")

        print(f"[INFO] Processed {frame_count} frames total")

    return 0


if __name__ == "__main__":
    exit(main())