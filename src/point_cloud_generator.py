import cv2
import numpy as np
from src.camera_calibrator import CameraModel

class PointCloudGenerator:
    """Transforms 2D disparity pixels into metric 3D points and writes .ply files."""

    def __init__(self, camera: CameraModel):
        self.camera = camera

    def reconstruct(self, disparity: np.ndarray, color_img: np.ndarray, min_z: float = 0.2, max_z: float = 10.0):
        h, w = disparity.shape[:2]
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

        # Triangulation formula: Z = (f * B) / d
        Z = (self.camera.fx * self.camera.baseline) / disparity
        X = (u_grid - self.camera.cx) * Z / self.camera.fx
        Y = (v_grid - self.camera.cy) * Z / self.camera.fy

        valid_mask = (Z >= min_z) & (Z <= max_z) & (disparity > 0.5)

        points = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=-1)
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if color_img.ndim == 3 else color_img
        colors = rgb[valid_mask]

        return points, colors

    def write_ply(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        num_vertices = points.shape[0]
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_vertices}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(filepath, "w") as f:
            f.write(header)
            for pt, clr in zip(points, colors):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")