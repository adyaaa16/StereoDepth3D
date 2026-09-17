import cv2
import numpy as np
from src.camera_calibrator import CameraModel
from src.disparity_estimator import DisparityEstimator
from src.post_processor import DisparityPostProcessor
from src.point_cloud_generator import PointCloudGenerator

class StereoReconstructionPipeline:
    """Orchestrates the entire stereopsis workflow."""

    def __init__(self, camera: CameraModel, num_disp: int = 64, block_size: int = 5):
        self.camera = camera
        self.estimator = DisparityEstimator(num_disp=num_disp, block_size=block_size)
        self.filter = DisparityPostProcessor()
        self.generator = PointCloudGenerator(camera)

    def run(self, img_l: np.ndarray, img_r: np.ndarray, ply_out: str):
        raw_disp = self.estimator.compute(img_l, img_r)
        clean_disp = self.filter.apply_bilateral_filter(raw_disp)
        points, colors = self.generator.reconstruct(clean_disp, img_l)
        self.generator.write_ply(ply_out, points, colors)
        return clean_disp, points, colors