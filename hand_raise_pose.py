"""
YOLO pose (COCO keypoints) + geometry rules for classroom-style hand raises.

Examples:
  python hand_raise_pose.py --source 0
  python hand_raise_pose.py --source clip.mp4 --single-person --save

Requires: ultralytics, opencv-python, numpy (see requirements.txt).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# --- COCO 17 keypoint indices (Ultralytics YOLO pose)
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12

# --- Defaults (seated classroom + duplicate box suppression)
DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_KP_CONF = 0.28
DEFAULT_RAISE_FRAC = 0.07
DEFAULT_DET_CONF = 0.35
DEFAULT_DUP_AREA_RATIO = 0.22
DEFAULT_DUP_IOU = 0.12
DEFAULT_SAVE_DIR = "runs/hand_raise"


@dataclass
class ArmRaiseState:
    left_raised: bool
    right_raised: bool
    left_score: float
    right_score: float


def _tensor_to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_numpy_xy_conf(kp_result: Any) -> tuple[np.ndarray, np.ndarray]:
    xy = _tensor_to_numpy(kp_result.xy)
    conf = kp_result.conf
    if conf is None:
        conf_arr = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    else:
        conf_arr = _tensor_to_numpy(conf)
    return xy.astype(np.float32), conf_arr.astype(np.float32)


def _shoulder_span_px(xy: np.ndarray, conf: np.ndarray, kp_conf: float) -> float | None:
    if conf[KP_L_SHOULDER] < kp_conf or conf[KP_R_SHOULDER] < kp_conf:
        return None
    return float(abs(xy[KP_L_SHOULDER, 0] - xy[KP_R_SHOULDER, 0]))


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xi1 = max(float(a[0]), float(b[0]))
    yi1 = max(float(a[1]), float(b[1]))
    xi2 = min(float(a[2]), float(b[2]))
    yi2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (aa + ab - inter))


def _center_in_box(cx: float, cy: float, xyxy: np.ndarray) -> bool:
    return bool(xyxy[0] <= cx <= xyxy[2] and xyxy[1] <= cy <= xyxy[3])


def suppress_spurious_person_detections(
    result: Any,
    *,
    area_ratio: float = DEFAULT_DUP_AREA_RATIO,
    iou_thresh: float = DEFAULT_DUP_IOU,
) -> Any:
    """Remove small overlapping person boxes (common when an arm is mistaken for a body)."""
    if result.boxes is None or len(result.boxes) <= 1:
        return result

    xyxy = _tensor_to_numpy(result.boxes.xyxy)
    n = xyxy.shape[0]
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    largest = int(np.argmax(areas))
    largest_area = float(areas[largest])
    largest_box = xyxy[largest]

    keep: list[int] = []
    for i in range(n):
        ai = float(areas[i])
        if ai >= area_ratio * largest_area:
            keep.append(i)
            continue
        iou = _bbox_iou_xyxy(xyxy[i], largest_box)
        cx = float((xyxy[i, 0] + xyxy[i, 2]) / 2.0)
        cy = float((xyxy[i, 1] + xyxy[i, 3]) / 2.0)
        inside = _center_in_box(cx, cy, largest_box)
        if iou >= iou_thresh or inside:
            continue
        keep.append(i)

    keep = sorted(set(keep))
    if len(keep) == n:
        return result
    return result[keep]


def _postprocess_detections(
    result: Any,
    *,
    single_person: bool,
    suppress_spurious: bool,
    dup_area_ratio: float,
    dup_iou: float,
) -> Any:
    if single_person and result.boxes is not None and len(result.boxes) > 1:
        xyxy = _tensor_to_numpy(result.boxes.xyxy)
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return result[[int(np.argmax(areas))]]
    if suppress_spurious:
        return suppress_spurious_person_detections(
            result, area_ratio=dup_area_ratio, iou_thresh=dup_iou
        )
    return result


def arm_raise_from_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    kp_conf: float = DEFAULT_KP_CONF,
    raise_frac: float = DEFAULT_RAISE_FRAC,
    seated: bool = True,
) -> ArmRaiseState:
    """Infer left/right hand raise from one person's (17, 2) xy and conf."""
    y_arr = xy[:, 1]
    c = conf

    def torso_scale(sh_y: float, hip_i: int) -> float:
        if c[hip_i] >= kp_conf:
            hip_y = float(y_arr[hip_i])
            return max(abs(sh_y - hip_y), 15.0)
        if c[KP_NOSE] >= kp_conf:
            nose_y = float(y_arr[KP_NOSE])
            return max(abs(sh_y - nose_y) * 2.5, 60.0)
        span = _shoulder_span_px(xy, conf, kp_conf)
        if span is not None:
            return max(span * 2.2, 80.0)
        return 120.0

    def side_state(
        sh_i: int,
        wr_i: int,
        hip_i: int,
        elbow_i: int,
    ) -> tuple[bool, float]:
        if c[sh_i] < kp_conf or c[wr_i] < kp_conf:
            return False, 0.0

        sh_y = float(y_arr[sh_i])
        wr_y = float(y_arr[wr_i])
        torso_h = torso_scale(sh_y, hip_i)

        vertical_lift = (sh_y - wr_y) / max(torso_h, 1e-3)

        nose_y = float(y_arr[KP_NOSE]) if c[KP_NOSE] >= kp_conf else None
        el_y = float(y_arr[elbow_i]) if c[elbow_i] >= kp_conf else None

        raised_vertical = vertical_lift >= raise_frac

        head_band = 0.42 * torso_h
        raised_head_zone = bool(
            nose_y is not None and wr_y <= nose_y + head_band
        )

        raised_elbow_high = False
        if el_y is not None:
            raised_elbow_high = el_y < sh_y + 0.12 * torso_h and wr_y <= el_y + 0.18 * torso_h

        raised_chain = False
        if el_y is not None:
            if c[hip_i] >= kp_conf:
                hip_y = float(y_arr[hip_i])
                raised_chain = (
                    wr_y < el_y
                    and el_y < sh_y + 0.2 * torso_h
                    and el_y < hip_y - 0.05 * torso_h
                )
            else:
                raised_chain = wr_y < el_y < sh_y + 0.15 * torso_h

        if seated:
            raised = raised_vertical or raised_head_zone or raised_elbow_high or raised_chain
        else:
            raised = raised_vertical or (
                nose_y is not None
                and wr_y < nose_y
                and vertical_lift >= raise_frac * 0.65
            )

        nose_bonus = 0.4 if nose_y is not None and wr_y < nose_y + 20.0 else 0.0
        elbow_bonus = 0.2 if raised_elbow_high else 0.0
        hip_bonus = 0.15 if c[hip_i] >= kp_conf else 0.0
        score = float(
            np.clip(vertical_lift + nose_bonus + elbow_bonus + hip_bonus, 0.0, 5.0)
        )
        return raised, score

    left_ok, left_s = side_state(KP_L_SHOULDER, KP_L_WRIST, KP_L_HIP, KP_L_ELBOW)
    right_ok, right_s = side_state(KP_R_SHOULDER, KP_R_WRIST, KP_R_HIP, KP_R_ELBOW)

    return ArmRaiseState(
        left_raised=left_ok,
        right_raised=right_ok,
        left_score=left_s,
        right_score=right_s,
    )


def annotate_title(states: list[ArmRaiseState]) -> str:
    parts = []
    for i, s in enumerate(states):
        flags = []
        if s.left_raised:
            flags.append("L-hand")
        if s.right_raised:
            flags.append("R-hand")
        label = " | ".join(flags) if flags else "hands down"
        parts.append(f"P{i}: {label}")
    return "  ".join(parts)


def run(
    source: str | int,
    *,
    model_name: str = DEFAULT_MODEL,
    kp_conf: float = DEFAULT_KP_CONF,
    raise_frac: float = DEFAULT_RAISE_FRAC,
    seated: bool = True,
    det_conf: float = DEFAULT_DET_CONF,
    suppress_spurious: bool = True,
    dup_area_ratio: float = DEFAULT_DUP_AREA_RATIO,
    dup_iou: float = DEFAULT_DUP_IOU,
    single_person: bool = False,
    save: bool = False,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    show: bool = True,
) -> None:
    model = YOLO(model_name)
    out_dir = Path(save_dir)
    frame_idx = 0
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    stream = model.predict(
        source=source,
        stream=True,
        verbose=False,
        conf=det_conf,
    )

    highlight = (20, 220, 20)
    muted = (200, 200, 200)

    for result in stream:
        result = _postprocess_detections(
            result,
            single_person=single_person,
            suppress_spurious=suppress_spurious,
            dup_area_ratio=dup_area_ratio,
            dup_iou=dup_iou,
        )

        states: list[ArmRaiseState] = []
        if result.keypoints is not None and len(result.keypoints):
            xy_all, conf_all = _to_numpy_xy_conf(result.keypoints)
            for person in range(xy_all.shape[0]):
                states.append(
                    arm_raise_from_keypoints(
                        xy_all[person],
                        conf_all[person],
                        kp_conf=kp_conf,
                        raise_frac=raise_frac,
                        seated=seated,
                    )
                )

        title = annotate_title(states)
        frame = result.plot()
        if title:
            color = highlight if any(s.left_raised or s.right_raised for s in states) else muted
            cv2.putText(
                frame,
                title,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        if save:
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        if show:
            cv2.imshow("Hand raise (pose)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect raised hands using Ultralytics YOLO pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0", help="Webcam index or path to image / video / folder")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Pose weights filename or path")
    p.add_argument(
        "--kp-conf",
        type=float,
        default=DEFAULT_KP_CONF,
        help="Minimum confidence for shoulder and wrist keypoints",
    )
    p.add_argument(
        "--raise-frac",
        type=float,
        default=DEFAULT_RAISE_FRAC,
        help="Minimum vertical lift (wrist above shoulder, normalized) for strict rule",
    )
    p.add_argument(
        "--no-seated",
        action="store_true",
        help="Standing-oriented geometry only (disable seated shortcuts)",
    )
    p.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DET_CONF,
        help="YOLO person detection confidence threshold",
    )
    p.add_argument(
        "--no-suppress-spurious",
        action="store_true",
        help="Disable filtering of small duplicate person boxes",
    )
    p.add_argument(
        "--dup-area-ratio",
        type=float,
        default=DEFAULT_DUP_AREA_RATIO,
        help="Suppress tiny boxes smaller than this × largest box area",
    )
    p.add_argument(
        "--dup-iou",
        type=float,
        default=DEFAULT_DUP_IOU,
        help="Suppress tiny boxes overlapping the largest box above this IoU",
    )
    p.add_argument(
        "--single-person",
        action="store_true",
        help="Keep only the largest person box (recommended for solo webcam)",
    )
    p.add_argument("--save", action="store_true", help=f"Save annotated frames under {DEFAULT_SAVE_DIR}/")
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Output directory when --save is set")
    p.add_argument("--no-show", action="store_true", help="Run without opening a window")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    src: str | int = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    run(
        src,
        model_name=args.model,
        kp_conf=args.kp_conf,
        raise_frac=args.raise_frac,
        seated=not args.no_seated,
        det_conf=args.det_conf,
        suppress_spurious=not args.no_suppress_spurious,
        dup_area_ratio=args.dup_area_ratio,
        dup_iou=args.dup_iou,
        single_person=args.single_person,
        save=args.save,
        save_dir=args.save_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
