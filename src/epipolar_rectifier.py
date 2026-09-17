import cv2
import numpy as np

class EpipolarRectifier:
    """Estimates the Fundamental Matrix F and validates epipolar alignment."""

    @staticmethod
    def estimate_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray):
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        return F, mask

    @staticmethod
    def compute_epipolar_error(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
        p1 = np.array([pt1[0], pt1[1], 1.0], dtype=np.float64)
        p2 = np.array([pt2[0], pt2[1], 1.0], dtype=np.float64)
        return float(np.abs(p2.T @ F @ p1))