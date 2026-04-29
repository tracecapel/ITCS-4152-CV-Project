import argparse
from pathlib import Path
from typing import Optional, Dict, List, Set
from collections import defaultdict

import os
import json
import time
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
    """Parse command-line arguments"""
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

    # Attendance tracking
    parser.add_argument(
        "--student-ids-dir",
        type=str,
        default="StudentIDs",
        help="Directory containing student ID subdirectories"
    )

    parser.add_argument(
        "--attendance-threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of checkpoints to mark as present (default: 0.8 = 80%%)"
    )

    parser.add_argument(
        "--camera-check-interval",
        type=int,
        default=10,
        help="Seconds between attendance checks for camera mode (default: 10)"
    )

    parser.add_argument(
        "--checkpoint-window",
        type=int,
        default=15,
        help="Number of frames before/after checkpoint to check for attendance (default: 15)"
    )

    parser.add_argument(
        "--hand-raise-cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between counting hand raises for same student (default: 3.0)"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_student_roster(student_ids_dir: str) -> List[str]:
    """Load list of registered students from directory structure"""
    roster = []
    student_dir = Path(student_ids_dir)

    if not student_dir.exists():
        print(f"[WARNING] StudentIDs directory not found: {student_ids_dir}")
        return roster

    for subdir in student_dir.iterdir():
        if subdir.is_dir():
            roster.append(subdir.name)

    return sorted(roster)


def calculate_checkpoints(total_frames: int, num_checks: int = 5) -> List[int]:
    """Calculate evenly spaced checkpoint frames for video mode"""
    if total_frames < num_checks:
        return list(range(total_frames))

    step = total_frames / num_checks
    checkpoints = [int(i * step) for i in range(num_checks)]
    return checkpoints


def is_in_checkpoint_window(frame_num: int, checkpoint_frame: int, window_size: int) -> bool:
    """Check if current frame is within the checkpoint window"""
    return checkpoint_frame - window_size <= frame_num <= checkpoint_frame + window_size


def save_attendance_metrics(
        output_path: Path,
        student_roster: List[str],
        attendance_checks: Dict[str, List[bool]],
        hand_raises: Dict[str, int],
        threshold: float
):
    """Save attendance metrics to file"""

    # Calculate attendance status
    metrics = {
        "attendance_summary": {
            "total_students": len(student_roster),
            "attendance_threshold": threshold,
        },
        "students": {}
    }

    present_count = 0

    for student in student_roster:
        checks = attendance_checks.get(student, [])
        total_checks = len(checks) if checks else 0
        present_checks = sum(checks) if checks else 0

        if total_checks > 0:
            attendance_rate = present_checks / total_checks
            is_present = attendance_rate >= threshold
        else:
            attendance_rate = 0.0
            is_present = False

        if is_present:
            present_count += 1

        hand_raise_count = hand_raises.get(student, 0)

        metrics["students"][student] = {
            "present": is_present,
            "attendance_rate": round(attendance_rate, 3),
            "checks_detected": present_checks,
            "total_checks": total_checks,
            "hand_raises": hand_raise_count
        }

    metrics["attendance_summary"]["students_present"] = present_count
    metrics["attendance_summary"]["students_absent"] = len(student_roster) - present_count

    # Save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save as human-readable text
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ATTENDANCE REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Students: {len(student_roster)}\n")
        f.write(f"Present: {present_count}\n")
        f.write(f"Absent: {len(student_roster) - present_count}\n")
        f.write(f"Attendance Threshold: {threshold * 100:.0f}%\n\n")

        f.write("-" * 60 + "\n")
        f.write("PRESENT STUDENTS\n")
        f.write("-" * 60 + "\n\n")

        for student in sorted(student_roster):
            student_data = metrics["students"][student]
            if student_data["present"]:
                f.write(f"{student}:\n")
                f.write(f"  Attendance Rate: {student_data['attendance_rate'] * 100:.1f}% ")
                f.write(f"({student_data['checks_detected']}/{student_data['total_checks']} checks)\n")
                f.write(f"  Hand Raises: {student_data['hand_raises']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("ABSENT STUDENTS\n")
        f.write("-" * 60 + "\n\n")

        for student in sorted(student_roster):
            student_data = metrics["students"][student]
            if not student_data["present"]:
                f.write(f"{student}:\n")
                f.write(f"  Attendance Rate: {student_data['attendance_rate'] * 100:.1f}% ")
                f.write(f"({student_data['checks_detected']}/{student_data['total_checks']} checks)\n\n")

    return json_path, txt_path


def main():
    """Main execution function"""
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
    is_camera = args.source.isdigit()

    if not is_camera and not source_path.exists():
        print(f"[ERROR] Source not found: {args.source}")
        return 1

    # ===================== LOAD STUDENT ROSTER =====================
    student_roster = load_student_roster(args.student_ids_dir)
    if args.verbose:
        print(f"[INFO] Loaded {len(student_roster)} students from roster")
        if student_roster:
            print(f"[INFO] Students: {', '.join(student_roster[:5])}" +
                  (f"... (+{len(student_roster) - 5} more)" if len(student_roster) > 5 else ""))

    # ===================== OUTPUT SETUP =====================
    if args.output:
        output_video_path = Path(args.output)
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        if is_camera:
            output_video_path = output_dir / f"camera_{args.source}_annotated.mp4"
        else:
            output_video_path = output_dir / (source_path.stem + "_annotated.mp4")

    # Metrics output path (same name as video)
    metrics_output_path = output_video_path.with_suffix('')  # Remove extension

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

    # ===================== GET VIDEO INFO =====================
    cap = cv2.VideoCapture(args.source)
    if args.fps:
        fps = args.fps
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0

    # Get total frames for video mode
    total_frames = None
    if not is_camera:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if args.verbose:
        print(f"[INFO] Video FPS: {fps:.2f}")
        if total_frames:
            print(f"[INFO] Total frames: {total_frames}")
        print(f"[INFO] Checkpoint window: ±{args.checkpoint_window} frames")
        print(f"[INFO] Hand raise cooldown: {args.hand_raise_cooldown}s")

    # ===================== SETUP ATTENDANCE TRACKING =====================
    # For video: use 5 equally spaced checkpoints with windows
    # For camera: check every N seconds with windows

    if is_camera:
        check_interval_frames = int(fps * args.camera_check_interval)
        checkpoints = None  # Will check dynamically
        if args.verbose:
            print(f"[INFO] Camera mode: checking every {args.camera_check_interval}s ({check_interval_frames} frames)")
    else:
        checkpoints = calculate_checkpoints(total_frames, num_checks=5)
        check_interval_frames = None
        if args.verbose:
            print(f"[INFO] Video mode: checkpoints at frames {checkpoints}")

    # Track attendance and hand raises
    attendance_checks: Dict[str, List[bool]] = {student: [] for student in student_roster}
    hand_raises: Dict[str, int] = {student: 0 for student in student_roster}

    # Track which students detected during current checkpoint window
    checkpoint_window_detections: Dict[int, Set[str]] = defaultdict(set)

    # Track active checkpoint windows (for video mode)
    active_checkpoints: List[int] = []
    finalized_checkpoints: Set[int] = set()

    # For camera mode: track when we start a new checkpoint window
    current_checkpoint_start: Optional[int] = None
    current_checkpoint_index = 0

    # Hand raise cooldown tracking: student_name -> last_raise_frame
    last_hand_raise_frame: Dict[str, int] = {}
    hand_raise_cooldown_frames = int(fps * args.hand_raise_cooldown)

    # ===================== VIDEO WRITER INITIALIZATION =====================
    writer: Optional[cv2.VideoWriter] = None

    # ===================== VALIDATE WEIGHTS =====================
    if args.appearance_weight < 0:
        print(f"[WARNING] Appearance weight is negative and will be set to 0 automatically")
        args.appearance_weight = 0
    if args.face_weight < 0:
        print(f"[WARNING] Face weight is negative and will be set to 0 automatically")
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
     
    last_good_face_embedding: Dict[int, torch.Tensor] = {}
    last_good_appearance_embedding: Dict[int, torch.Tensor] = {}
    
    try:
        for r in model.track(source=args.source, stream=True, classes=[0]):  # class 0 = person
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

            # ===================== CHECKPOINT WINDOW MANAGEMENT =====================
            if is_camera:
                # Camera mode: start new checkpoint window every N frames
                if current_checkpoint_start is None:
                    current_checkpoint_start = frame_count
                    if args.verbose:
                        print(f"[INFO] Starting checkpoint window {current_checkpoint_index} at frame {frame_count}")

                # Check if we're within the current window
                in_window = (frame_count <= current_checkpoint_start + args.checkpoint_window)

                # Check if we need to finalize this window and start next
                if frame_count >= current_checkpoint_start + check_interval_frames:
                    # Finalize current checkpoint
                    detected_students = checkpoint_window_detections[current_checkpoint_index]
                    for student in student_roster:
                        was_detected = student in detected_students
                        attendance_checks[student].append(was_detected)

                    if args.verbose:
                        print(
                            f"[INFO] Finalized checkpoint {current_checkpoint_index}: {len(detected_students)} students detected")

                    # Start new checkpoint
                    current_checkpoint_index += 1
                    current_checkpoint_start = frame_count
                    checkpoint_window_detections[current_checkpoint_index] = set()

                    if args.verbose:
                        print(f"[INFO] Starting checkpoint window {current_checkpoint_index} at frame {frame_count}")

            else:
                # Video mode: determine which checkpoint windows we're in
                for checkpoint_idx, checkpoint_frame in enumerate(checkpoints):
                    if checkpoint_idx in finalized_checkpoints:
                        continue

                    # Check if we're in this checkpoint's window
                    if is_in_checkpoint_window(frame_count - 1, checkpoint_frame, args.checkpoint_window):
                        if checkpoint_idx not in active_checkpoints:
                            active_checkpoints.append(checkpoint_idx)
                            if args.verbose:
                                print(
                                    f"[INFO] Entered checkpoint window {checkpoint_idx} (frame {checkpoint_frame}±{args.checkpoint_window})")

                    # Check if we've passed this checkpoint's window
                    elif frame_count - 1 > checkpoint_frame + args.checkpoint_window:
                        if checkpoint_idx in active_checkpoints:
                            # Finalize this checkpoint
                            detected_students = checkpoint_window_detections[checkpoint_idx]
                            for student in student_roster:
                                was_detected = student in detected_students
                                attendance_checks[student].append(was_detected)

                            active_checkpoints.remove(checkpoint_idx)
                            finalized_checkpoints.add(checkpoint_idx)

                            if args.verbose:
                                print(
                                    f"[INFO] Finalized checkpoint {checkpoint_idx}: {len(detected_students)} students detected")

            # ===================== PROCESS DETECTIONS =====================
            
            if r.boxes is not None:
                
                for box in r.boxes:
                    if box is None or box.id is None:
                        continue
                    try:
                        track_id = int(box.id.item())

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        # ===================== FACE EMBEDDING =====================
                        face_tensor = mtcnn(crop)
                        if face_tensor is None:
                            face_embedding = last_good_face_embedding.get(track_id, None)
                            
    
                         # if no face history, try appearance-only recognition
                            if face_embedding is None:
                                face_embedding = last_good_appearance_embedding.get(
                                track_id, torch.zeros(512, device=device)
                                )
                        else:
                            face_tensor = face_tensor.to(device)
                            with torch.no_grad():
                                face_embedding = resnet(face_tensor.unsqueeze(0)).squeeze(0)
                                last_good_face_embedding[track_id] = face_embedding  # cache it

                        appearance_embedding = image_feature_extractor(crop).squeeze(0).flatten().to(device)
                        last_good_appearance_embedding[track_id] = appearance_embedding

                        

                        # ===================== HYBRID EMBEDDING =====================
                        if appearance_embedding.numel() == 0 and face_embedding.numel() == 0:
                            continue

                    
                        if 0 in appearance_embedding.shape or 0 in face_embedding.shape:
                            continue
                    

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

                        # ===================== ATTENDANCE TRACKING =====================
                        if name != "Unknown":

                            color = (0, 255, 0)  # green for recognized
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                "Logged!",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )
                        
                            # Add to all active checkpoint windows
                            if is_camera:
                                if in_window:
                                    checkpoint_window_detections[current_checkpoint_index].add(name)
                            else:
                                for checkpoint_idx in active_checkpoints:
                                    checkpoint_window_detections[checkpoint_idx].add(name)
                        else:
                            color = (255, 0, 0)  # red for pending
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                "Tracking",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )

                        # ===================== HAND DETECTION =====================
                        result = detector.detect(crop)
                        hand_raised = result["left"] or result["right"]

                        # Track hand raises with cooldown for recognized students
                        if hand_raised and name != "Unknown":
                            last_raise = last_hand_raise_frame.get(name, -float('inf'))
                            frames_since_last_raise = frame_count - last_raise

                            # Only count if cooldown period has passed
                            if frames_since_last_raise >= hand_raise_cooldown_frames:
                                hand_raises[name] = hand_raises.get(name, 0) + 1
                                last_hand_raise_frame[name] = frame_count

                                if args.verbose:
                                    print(f"[INFO] Hand raise counted for {name} at frame {frame_count}")

                        # ===================== LABEL =====================
                        label = f"{name} ({score:.2f})"

                        if hand_raised:
                            label += " | HAND RAISED"

                        
                    except:
                        print("failed")
                        continue
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
        # ===================== FINALIZE REMAINING CHECKPOINTS =====================
        if is_camera and current_checkpoint_start is not None:
            # Finalize the last checkpoint window
            detected_students = checkpoint_window_detections[current_checkpoint_index]
            for student in student_roster:
                was_detected = student in detected_students
                attendance_checks[student].append(was_detected)

            if args.verbose:
                print(
                    f"[INFO] Finalized final checkpoint {current_checkpoint_index}: {len(detected_students)} students detected")

        else:
            # Finalize any remaining active checkpoints in video mode
            for checkpoint_idx in active_checkpoints:
                if checkpoint_idx not in finalized_checkpoints:
                    detected_students = checkpoint_window_detections[checkpoint_idx]
                    for student in student_roster:
                        was_detected = student in detected_students
                        attendance_checks[student].append(was_detected)

                    if args.verbose:
                        print(
                            f"[INFO] Finalized remaining checkpoint {checkpoint_idx}: {len(detected_students)} students detected")

        # ===================== CLEANUP =====================
        if args.display:
            cv2.destroyAllWindows()

        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved to: {output_video_path}")

        # ===================== SAVE ATTENDANCE METRICS =====================
        if student_roster:
            try:
                json_path, txt_path = save_attendance_metrics(
                    metrics_output_path,
                    student_roster,
                    attendance_checks,
                    hand_raises,
                    args.attendance_threshold
                )
                print(f"[INFO] Attendance metrics saved:")
                print(f"       JSON: {json_path}")
                print(f"       Text: {txt_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save attendance metrics: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        print(f"[INFO] Processed {frame_count} frames total")

    return 0


if __name__ == "__main__":
    exit(main())