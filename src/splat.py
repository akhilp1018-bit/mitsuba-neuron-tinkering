import numpy as np

def splat_emitters_with_psf_zyx(
    u: np.ndarray,
    v: np.ndarray,
    z_nm: np.ndarray,
    zmin_nm: float,
    num_slices: int,
    H: int,
    W: int,
    z_step_nm: float,
    psf_zyx: np.ndarray,
) -> np.ndarray:
    """
    Splat emitters into a (Z,Y,X) volume using an explicit PSF kernel.
    u, v are pixel coordinates (float), z_nm in nanometers.
    Returns vol float16 (Z,Y,X).
    """
    pz, py, px = psf_zyx.shape
    cz, cy, cx = pz // 2, py // 2, px // 2

    vol = np.zeros((num_slices, H, W), dtype=np.float16)

    k = np.round((z_nm - zmin_nm) / z_step_nm).astype(np.int32)
    valid = (k >= 0) & (k < num_slices)

    u_i = np.round(u[valid]).astype(np.int32)
    v_i = np.round(v[valid]).astype(np.int32)
    k_i = k[valid]

    for x0, y0, z0 in zip(u_i, v_i, k_i):
        oz0 = z0 - cz
        oz1 = oz0 + pz
        oy0 = y0 - cy
        oy1 = oy0 + py
        ox0 = x0 - cx
        ox1 = ox0 + px

        vz0 = max(0, oz0); vz1 = min(num_slices, oz1)
        vy0 = max(0, oy0); vy1 = min(H, oy1)
        vx0 = max(0, ox0); vx1 = min(W, ox1)

        pz0 = vz0 - oz0; pz1 = pz0 + (vz1 - vz0)
        py0 = vy0 - oy0; py1 = py0 + (vy1 - vy0)
        px0 = vx0 - ox0; px1 = px0 + (vx1 - vx0)

        vol[vz0:vz1, vy0:vy1, vx0:vx1] += psf_zyx[pz0:pz1, py0:py1, px0:px1].astype(np.float16)

    return vol