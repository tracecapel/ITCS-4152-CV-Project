import numpy as np
from dataclasses import dataclass

# ====== Keypoint indices (COCO format from YOLO pose) ======
KP_NOSE = 0

KP_L_SHOULDER = 5
KP_R_SHOULDER = 6

KP_L_ELBOW = 7
KP_R_ELBOW = 8

KP_L_WRIST = 9
KP_R_WRIST = 10

KP_L_HIP = 11
KP_R_HIP = 12

# ====== Defaults ======
DEFAULT_KP_CONF = 0.3
DEFAULT_RAISE_FRAC = 0.25


# ====== Data Structure ======
@dataclass
class ArmRaiseState:
    left_raised: bool
    right_raised: bool
    left_score: float
    right_score: float


# ====== Utility Converters ======
def _to_numpy_xy_conf(keypoints):
    """
    Converts Ultralytics keypoints object into numpy arrays.

    Returns:
        xy: (N, 17, 2)
        conf: (N, 17)
    """
    xy = keypoints.xy.cpu().numpy()
    conf = keypoints.conf.cpu().numpy()
    return xy, conf


def _shoulder_span_px(xy, conf, kp_conf):
    if conf[KP_L_SHOULDER] >= kp_conf and conf[KP_R_SHOULDER] >= kp_conf:
        return abs(xy[KP_L_SHOULDER][0] - xy[KP_R_SHOULDER][0])
    return None


# ====== Core Logic ======
def arm_raise_from_keypoints(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    kp_conf: float = DEFAULT_KP_CONF,
    raise_frac: float = DEFAULT_RAISE_FRAC,
    seated: bool = True,
) -> ArmRaiseState:
    """
    Determines if left/right arms are raised using pose keypoints.
    """

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

    def side_state(sh_i, wr_i, hip_i, elbow_i):
        if c[sh_i] < kp_conf or c[wr_i] < kp_conf:
            return False, 0.0

        sh_y = float(y_arr[sh_i])
        wr_y = float(y_arr[wr_i])
        torso_h = torso_scale(sh_y, hip_i)

        vertical_lift = (sh_y - wr_y) / max(torso_h, 1e-3)

        nose_y = float(y_arr[KP_NOSE]) if c[KP_NOSE] >= kp_conf else None
        el_y = float(y_arr[elbow_i]) if c[elbow_i] >= kp_conf else None

        # ===== Core conditions =====
        raised_vertical = vertical_lift >= raise_frac

        head_band = 0.42 * torso_h
        raised_head_zone = (
            nose_y is not None and wr_y <= nose_y + head_band
        )

        raised_elbow_high = False
        if el_y is not None:
            raised_elbow_high = (
                el_y < sh_y + 0.12 * torso_h
                and wr_y <= el_y + 0.18 * torso_h
            )

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

        # ===== Final decision =====
        if seated:
            raised = (
                raised_vertical
                or raised_head_zone
                or raised_elbow_high
                or raised_chain
            )
        else:
            raised = raised_vertical or (
                nose_y is not None
                and wr_y < nose_y
                and vertical_lift >= raise_frac * 0.65
            )

        # ===== Confidence score =====
        nose_bonus = 0.4 if (nose_y is not None and wr_y < nose_y + 20.0) else 0.0
        elbow_bonus = 0.2 if raised_elbow_high else 0.0
        hip_bonus = 0.15 if c[hip_i] >= kp_conf else 0.0

        score = float(
            np.clip(vertical_lift + nose_bonus + elbow_bonus + hip_bonus, 0.0, 5.0)
        )

        return raised, score

    # ===== Left / Right =====
    left_ok, left_s = side_state(
        KP_L_SHOULDER, KP_L_WRIST, KP_L_HIP, KP_L_ELBOW
    )

    right_ok, right_s = side_state(
        KP_R_SHOULDER, KP_R_WRIST, KP_R_HIP, KP_R_ELBOW
    )

    return ArmRaiseState(
        left_raised=left_ok,
        right_raised=right_ok,
        left_score=left_s,
        right_score=right_s,
    )