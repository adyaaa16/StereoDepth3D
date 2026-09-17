import cv2
import numpy as np

class DisparityEstimator:
    """Computes dense disparity map using OpenCV's StereoSGBM."""

    def __init__(self, min_disp: int = 0, num_disp: int = 64, block_size: int = 5):
        self.num_disp = (num_disp // 16) * 16
        self.block_size = block_size
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 3 * (self.block_size ** 2),
            P2=32 * 3 * (self.block_size ** 2),
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """Computes pixel-wise horizontal disparity d = xL - xR."""
        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if img_left.ndim == 3 else img_left
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if img_right.ndim == 3 else img_right

        raw_disp = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        # Prevent division by zero
        raw_disp[raw_disp <= 0.0] = 0.1
        return raw_disp