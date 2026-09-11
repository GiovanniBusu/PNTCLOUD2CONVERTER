"""Local normal estimation + consistent orientation.

Uses Open3D when it is installed (`estimate_normals` +
`orient_normals_consistent_tangent_plane`, as recommended in the project
brief). Open3D is an optional, fairly heavy dependency that does not
install cleanly on every platform, so a pure numpy/scipy fallback is
provided and used automatically when Open3D is unavailable:

  1. PCA over each point's k-NN neighborhood -> normal = eigenvector of the
     smallest eigenvalue of the local covariance matrix (vectorized via
     batched np.linalg.eigh).
  2. Consistent orientation via minimum-spanning-tree propagation over the
     k-NN graph (Hoppe et al., 1992): traverse the MST and flip a normal
     whenever it disagrees in sign with its parent's.

Either path yields the same PointCloudData contract: unit-length (N, 3)
float32 normals.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree


def estimate_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError:
        return _estimate_normals_numpy(points, k=k)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(k, 3)))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=max(k, 3))
    except Exception:
        # Open3D can raise on degenerate/very small clouds; keep unoriented normals.
        pass
    return np.asarray(pcd.normals, dtype=np.float32)


def _estimate_normals_numpy(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    k = max(3, min(k, n - 1))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1, workers=-1)

    neighborhoods = points[idx]  # (N, k+1, 3)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k  # (N, 3, 3)

    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = eigvecs[:, :, 0].astype(np.float32)  # smallest-eigenvalue axis
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return _orient_consistently(points, normals, idx[:, 1:])


def _orient_consistently(points: np.ndarray, normals: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    k = neighbor_idx.shape[1]

    rows = np.repeat(np.arange(n), k)
    cols = neighbor_idx.reshape(-1)
    dots = np.abs(np.einsum("ij,ij->i", normals[rows], normals[cols]))
    weights = 1.0 - dots  # low weight = well-aligned normals

    graph = coo_matrix((weights, (rows, cols)), shape=(n, n))
    # symmetrize (keep the smaller weight on each undirected pair)
    graph = graph.minimum(graph.T)
    mst = minimum_spanning_tree(graph.tocsr())
    mst = mst.maximum(mst.T)  # make traversable both ways

    root = int(np.argmax(points[:, 2]))  # arbitrary but deterministic root
    order, predecessors = breadth_first_order(mst, i_start=root, directed=False, return_predecessors=True)

    oriented = normals.copy()
    for node in order:
        parent = predecessors[node]
        if parent < 0:
            continue
        if np.dot(oriented[parent], oriented[node]) < 0:
            oriented[node] = -oriented[node]

    # Points unreachable in a disconnected graph keep their PCA sign as-is.
    return oriented.astype(np.float32)
