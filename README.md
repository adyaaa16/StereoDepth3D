# StereoDepth3D: Binocular Stereopsis & 3D Point Cloud Reconstruction Pipeline

**StereoDepth3D** is a modular Computer Vision application that estimates dense metric depth maps and reconstructs interactive 3D point clouds from rectified binocular stereo image pairs. It utilizes epipolar geometry constraints, semi-global disparity estimation, and pinhole camera reprojection.

---

## Features

- **Epipolar Geometry Verification:** Algebraic validation of point-line epipolar constraints ($x'^T F x = 0$).
- **Dense Disparity Engine:** Semi-Global Block Matching (StereoSGBM) with speckle window filtering.
- **Edge-Preserving Post-Processing:** Bilateral and median filtering to eliminate boundary blur and impulse noise.
- **3D Metric Triangulation:** Direct mapping from pixel disparity to real-world Cartesian coordinates:
  $$Z = \frac{f \cdot B}{d}, \quad X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$
- **Standard 3D Export:** Generates colorized Stanford `.ply` files compatible with Open3D, MeshLab, and CloudCompare.

---

## Tech Stack & Tools

- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (cv2)
- **Scientific Computing:** NumPy
- **Visualization:** Matplotlib, Open3D
- **Testing:** Pytest

---

## Project Structure
