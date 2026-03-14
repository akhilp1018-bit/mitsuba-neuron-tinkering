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



def sample_mesh_surface_deterministic(mesh_path, spacing_nm=200.0):
    """
    Deterministic surface sampling instead of random emitters.
    """

    mesh = trimesh.load(mesh_path, process=False)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    pts = []

    for f in faces:
        v0, v1, v2 = vertices[f]

        # triangle area
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # number of samples based on area
        n = max(1, int(np.ceil(area / (spacing_nm**2))))

        m = int(np.ceil(np.sqrt(n)))

        for i in range(m + 1):
            for j in range(m + 1 - i):

                a = i / max(m, 1)
                b = j / max(m, 1)
                c = 1 - a - b

                p = a * v0 + b * v1 + c * v2
                pts.append(p)

    return np.array(pts, dtype=np.float32)