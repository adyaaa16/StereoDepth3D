import cv2
import numpy as np

class DisparityPostProcessor:
    """Cleans up disparity maps using edge-preserving filtering."""

    @staticmethod
    def apply_bilateral_filter(disparity: np.ndarray) -> np.ndarray:
        disp_min, disp_max = disparity.min(), disparity.max()
        if disp_max - disp_min == 0:
            return disparity
        norm_disp = (disparity - disp_min) / (disp_max - disp_min)
        smoothed = cv2.bilateralFilter(norm_disp.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)
        return smoothed * (disp_max - disp_min) + disp_min

    @staticmethod
    def apply_median_filter(disparity: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.medianBlur(disparity.astype(np.float32), ksize)