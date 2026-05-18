
import numpy as np
import trimesh


def compute_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.vertex_normals.astype(np.float32)


def compute_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Simple edge-angle-based mean curvature proxy used in your pretrain code.
    Returns (V, 1).
    """
    V = mesh.vertices.shape[0]
    curvatures = np.zeros(V, dtype=np.float32)
    if mesh.face_adjacency is None or mesh.face_adjacency_angles is None:
        return curvatures.reshape(-1, 1)

    for (edge, angle) in zip(mesh.face_adjacency_edges, mesh.face_adjacency_angles):
        i, j = edge
        length = np.linalg.norm(mesh.vertices[i] - mesh.vertices[j])
        cur = 0.25 * length * angle
        curvatures[i] += cur
        curvatures[j] += cur

    return curvatures.reshape(-1, 1)


def compute_radial_distance(mesh: trimesh.Trimesh) -> np.ndarray:
    centroid = mesh.vertices.mean(axis=0)
    distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
    return distances.reshape(-1, 1).astype(np.float32)


def sample_or_pad(feat: np.ndarray, n_points: int = 12500, seed: int | None = None) -> np.ndarray:
    """
    feat: (V, C). Return (n_points, C) via random sampling or edge padding.
    """
    rng = np.random.default_rng(seed)
    V = feat.shape[0]
    if V > n_points:
        idx = rng.choice(V, n_points, replace=False)
        return feat[idx]
    if V < n_points:
        pad = n_points - V
        return np.pad(feat, ((0, pad), (0, 0)), mode="edge")
    return feat


def mesh_to_feat5(mesh_path: str, n_points: int = 12500, seed: int | None = None) -> np.ndarray:
    """
    Returns (n_points, 5): normals(3) + curvature(1) + radial(1)
    """
    mesh = trimesh.load(mesh_path, process=True)
    normals = compute_vertex_normals(mesh)
    curvature = compute_mean_curvature(mesh)
    radial = compute_radial_distance(mesh)
    feat = np.concatenate([normals, curvature, radial], axis=1).astype(np.float32)
    feat = sample_or_pad(feat, n_points=n_points, seed=seed)
    return feat
