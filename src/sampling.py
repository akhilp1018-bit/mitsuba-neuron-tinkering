import numpy as np
import trimesh

def sample_thickshell_emitters_nm(
    mesh_path: str,
    num_emitters: int,
    thickness_um: float,
    jitter_um: float,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Sample emitters on mesh surface and push them inward (thick-shell model).
    Returns points in nanometers, shape (N, 3).
    """
    tm = trimesh.load(mesh_path, force="mesh", process=False)
    rng = np.random.default_rng(rng_seed)

    surf_pts, face_idx = trimesh.sample.sample_surface(tm, num_emitters)
    face_normals = tm.face_normals[face_idx]

    thickness_nm = thickness_um * 1000.0
    jitter_nm = jitter_um * 1000.0

    depth_nm = rng.uniform(0.0, thickness_nm, size=(num_emitters, 1)).astype(np.float32)
    pts_in = surf_pts.astype(np.float32) - face_normals.astype(np.float32) * depth_nm
    pts_in += rng.normal(0.0, jitter_nm, size=pts_in.shape).astype(np.float32)

    return pts_in