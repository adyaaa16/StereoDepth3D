import numpy as np

class CameraModel:
    """Manages intrinsic camera parameters and stereo baseline geometry."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, baseline: float):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.baseline = float(baseline)  # Baseline in meters

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

    def __repr__(self):
        return f"CameraModel(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, baseline={self.baseline}m)"