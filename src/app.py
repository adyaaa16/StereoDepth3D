import argparse
import cv2
import matplotlib.pyplot as plt
from src.camera_calibrator import CameraModel
from src.pipeline import StereoReconstructionPipeline

def main():
    parser = argparse.ArgumentParser(description="StereoDepth3D Reconstruction")
    parser.add_argument("--left", type=str, default="data/stereo_pairs/left.png", help="Left image path")
    parser.add_argument("--right", type=str, default="data/stereo_pairs/right.png", help="Right image path")
    parser.add_argument("--fx", type=float, default=800.0, help="Focal length in pixels")
    parser.add_argument("--baseline", type=float, default=0.12, help="Camera baseline in meters")
    parser.add_argument("--out", type=str, default="data/output_3d/scene_model.ply", help="Output .ply path")
    args = parser.parse_args()

    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)

    if img_l is None or img_r is None:
        print(f"Error: Unable to load images from {args.left} and {args.right}.")
        print("Please place matching 'left.png' and 'right.png' inside data/stereo_pairs/ folder.")
        return

    h, w = img_l.shape[:2]
    camera = CameraModel(fx=args.fx, fy=args.fx, cx=w / 2.0, cy=h / 2.0, baseline=args.baseline)
    print(f"Initialized: {camera}")

    pipeline = StereoReconstructionPipeline(camera)
    print("Running disparity estimation and 3D reconstruction...")
    disparity, points, _ = pipeline.run(img_l, img_r, args.out)

    print(f"Success! Generated {len(points)} 3D points.")
    print(f"3D Model saved to: {args.out}")

    # Plot visual output
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Left Reference Image")
    plt.imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Estimated Disparity Heatmap")
    plt.imshow(disparity, cmap="inferno")
    plt.colorbar(label="Disparity (pixels)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("data/output_3d/disparity_map.png")
    plt.show()

if __name__ == "__main__":
    main()