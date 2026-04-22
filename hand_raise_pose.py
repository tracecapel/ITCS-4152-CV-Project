"""
Pose-based hand-raise detection using Ultralytics YOLO pose (COCO 17 keypoints).

Run:
  python hand_raise_pose.py --source 0
  python hand_raise_pose.py --source path/to/video.mp4 --save
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

# COCO pose keypoint indices (YOLO pose models)
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12


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
    """Ultralytics Keypoints -> (N, 17, 2) xy and (N, 17) conf."""
    xy = _tensor_to_numpy(kp_result.xy)
    conf = kp_result.conf
    if conf is None:
        conf_arr = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    else:
        conf_arr = _tensor_to_numpy(conf)
    return xy.astype(np.float32), conf_arr.astype(np.float32)


def arm_raise_from_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    kp_conf: float = 0.35,
    raise_frac: float = 0.18,
    nose_boost_y: float = 12.0,
) -> ArmRaiseState:
    """
    Decide if left/right arm is raised using shoulder–wrist geometry.

    Image coordinates: smaller y is higher on the screen.
    Raised ≈ wrist clearly above shoulder; optional boost if wrist is near/above nose height.
    """
    x, y = xy[:, 0], xy[:, 1]
    c = conf

    def side_state(
        sh_i: int,
        wr_i: int,
        hip_i: int,
        elbow_i: int,
    ) -> tuple[bool, float]:
        if not (
            c[sh_i] >= kp_conf
            and c[wr_i] >= kp_conf
            and c[hip_i] >= kp_conf
        ):
            return False, 0.0

        sh_y, wr_y = float(y[sh_i]), float(y[wr_i])
        hip_y = float(y[hip_i])

        torso_h = max(abs(sh_y - hip_y), 1e-3)
        # Positive when wrist is above shoulder (smaller wr_y).
        vertical_lift = (sh_y - wr_y) / torso_h

        nose_ok = c[KP_NOSE] >= kp_conf
        nose_y = float(y[KP_NOSE]) if nose_ok else None
        nose_boost = 0.0
        if nose_y is not None and wr_y < nose_y + nose_boost_y:
            nose_boost = 0.35

        elbow_bonus = 0.0
        if c[elbow_i] >= kp_conf:
            el_y = float(y[elbow_i])
            if el_y < sh_y + 0.15 * torso_h:
                elbow_bonus = 0.15

        score = float(np.clip(vertical_lift + nose_boost + elbow_bonus, 0.0, 5.0))
        raised = vertical_lift >= raise_frac or (
            nose_y is not None and wr_y < nose_y and vertical_lift >= raise_frac * 0.65
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
    return "  ".join(parts) if parts else ""


def run(
    source: str | int,
    *,
    model_name: str = "yolov8n-pose.pt",
    kp_conf: float = 0.35,
    raise_frac: float = 0.18,
    save: bool = False,
    save_dir: str | Path = "runs/hand_raise",
    show: bool = True,
) -> None:
    import cv2

    model = YOLO(model_name)
    out_dir = Path(save_dir)
    frame_idx = 0
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Stream generator; pull boxes + keypoints each frame.
    stream = model.predict(
        source=source,
        stream=True,
        verbose=False,
        conf=0.25,
    )

    for result in stream:
        states: list[ArmRaiseState] = []
        if result.keypoints is not None and len(result.keypoints):
            xy_all, conf_all = _to_numpy_xy_conf(result.keypoints)
            for person in range(xy_all.shape[0]):
                st = arm_raise_from_keypoints(
                    xy_all[person],
                    conf_all[person],
                    kp_conf=kp_conf,
                    raise_frac=raise_frac,
                )
                states.append(st)

        title = annotate_title(states)
        frame = result.plot()
        if title:
            cv2.putText(
                frame,
                title,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 220, 20) if any(s.left_raised or s.right_raised for s in states) else (200, 200, 200),
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


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO pose: detect raised hands")
    p.add_argument(
        "--source",
        default="0",
        help='Webcam index (0), or path to image/video/folder',
    )
    p.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Ultralytics pose weights (e.g. yolov8n-pose.pt, yolov8s-pose.pt)",
    )
    p.add_argument("--kp-conf", type=float, default=0.35, help="Min keypoint confidence")
    p.add_argument(
        "--raise-frac",
        type=float,
        default=0.18,
        help="Min (shoulder_y - wrist_y) / torso height to count as raised",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated frames to --save-dir (default: runs/hand_raise/)",
    )
    p.add_argument(
        "--save-dir",
        default="runs/hand_raise",
        help="Directory for annotated frames when --save is set",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    args = p.parse_args()

    src: str | int = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    run(
        src,
        model_name=args.model,
        kp_conf=args.kp_conf,
        raise_frac=args.raise_frac,
        save=args.save,
        save_dir=args.save_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
