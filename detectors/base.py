class HandRaiseDetector:
    def detect(self, crop):
        """
        Args:
            crop (np.ndarray): image of a person

        Returns:
            dict:
            {
                "left": bool,
                "right": bool,
                "confidence": float
            }
        """
        raise NotImplementedError