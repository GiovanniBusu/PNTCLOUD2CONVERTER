"""splatconv: convert point clouds (E57, PTS/XYZ, LAS) into 3D Gaussian Splatting PLY files.

The splats are *synthetic surface splats* built directly from a colored/classified
point cloud (position + color + local normal + local density), not learned by
gradient-based 3DGS optimization. No paired photography is required.
"""

__version__ = "0.1.0"
