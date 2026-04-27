from detectors.base import HandRaiseDetector

class PalmCNNDetector(HandRaiseDetector):
    def __init__(self, model):
        self.model = model

    def detect(self, crop):
        # TODO: implement
        return {"left": False, "right": False, "confidence": 0.0}