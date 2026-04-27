from detectors.base import HandRaiseDetector
from utils.pose_utils import arm_raise_from_keypoints, _to_numpy_xy_conf

class PoseRuleBasedDetector(HandRaiseDetector):
    def __init__(self, pose_model, kp_conf=0.3, raise_frac=0.25):
        self.pose_model = pose_model
        self.kp_conf = kp_conf
        self.raise_frac = raise_frac

    def detect(self, crop):
        result = self.pose_model(crop)

        if not result or result[0].keypoints is None:
            return {"left": False, "right": False, "confidence": 0.0}

        xy, conf = _to_numpy_xy_conf(result[0].keypoints)

        if len(xy) == 0:
            return {"left": False, "right": False, "confidence": 0.0}

        state = arm_raise_from_keypoints(
            xy[0],
            conf[0],
            kp_conf=self.kp_conf,
            raise_frac=self.raise_frac,
            seated=True
        )

        return {
            "left": state.left_raised,
            "right": state.right_raised,
            "confidence": max(state.left_score, state.right_score)
        }