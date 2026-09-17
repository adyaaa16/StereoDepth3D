import numpy as np
from src.camera_calibrator import CameraModel
from src.point_cloud_generator import PointCloudGenerator

def test_depth_triangulation():
    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    baseline = 0.12
    camera = CameraModel(fx, fy, cx, cy, baseline)
    generator = PointCloudGenerator(camera)

    test_disp = np.full((5, 5), 40.0, dtype=np.float32)
    test_color = np.zeros((5, 5, 3), dtype=np.uint8)

    points, _ = generator.reconstruct(test_disp, test_color)
    expected_depth = (800.0 * 0.12) / 40.0  # 2.4 meters

    assert np.allclose(points[:, 2], expected_depth, atol=1e-3)