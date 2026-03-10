import numpy as np
from scipy.signal import fftconvolve

def focal_stack_from_density(rho_zyx, psf_zyx):
    """
    Memory-light focal plane simulation:
    For each z slice, sum XY convolutions of nearby rho slices weighted by PSF(z).
    rho_zyx: (Z,Y,X)
    psf_zyx: (Zp,Yp,Xp) with odd Yp/Xp recommended
    returns: vol_zyx (Z,Y,X)
    """
    Z, H, W = rho_zyx.shape
    Zp, Yp, Xp = psf_zyx.shape
    cz = Zp // 2

    vol = np.zeros_like(rho_zyx, dtype=np.float32)

    rho_zyx = rho_zyx.astype(np.float32, copy=False)
    psf_zyx = psf_zyx.astype(np.float32, copy=False)

    for k in range(Z):
        acc = np.zeros((H, W), dtype=np.float32)
        for dz in range(-cz, cz + 1):
            zsrc = k + dz
            if zsrc < 0 or zsrc >= Z:
                continue
            kz = dz + cz
            kernel_xy = psf_zyx[kz]
            acc += fftconvolve(rho_zyx[zsrc], kernel_xy, mode="same").astype(np.float32, copy=False)
        vol[k] = acc
    return vol


def points_to_density_zyx(points_nm, origin_nm, voxel_size_nm_xyz, shape_zyx, weights=None):
    """
    points_nm: (N,3) in nm, columns (x,y,z)
    origin_nm: (3,) nm of grid origin (x0,y0,z0) corresponding to voxel (0,0,0)
    voxel_size_nm_xyz: (3,) voxel size in nm (sx,sy,sz)
    shape_zyx: (Z,Y,X)
    weights: optional (N,) intensities per point
    """
    Z, Y, X = shape_zyx
    rho = np.zeros((Z, Y, X), dtype=np.float32)

    if points_nm is None or len(points_nm) == 0:
        return rho

    sx, sy, sz = voxel_size_nm_xyz
    x0, y0, z0 = origin_nm

    ix = np.floor((points_nm[:, 0] - x0) / sx).astype(np.int32)
    iy = np.floor((points_nm[:, 1] - y0) / sy).astype(np.int32)
    iz = np.floor((points_nm[:, 2] - z0) / sz).astype(np.int32)

    m = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    ix, iy, iz = ix[m], iy[m], iz[m]

    if weights is None:
        w = np.ones(ix.shape[0], dtype=np.float32)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != points_nm.shape[0]:
            raise ValueError("weights must have same length as points_nm")
        w = weights[m].astype(np.float32)

    np.add.at(rho, (iz, iy, ix), w)
    return rho


def ensure_psf_odd_xy(psf_zyx, renormalize=False):
    """Pad PSF to odd Y,X if needed (keeps center well-defined)."""
    Z, Y, X = psf_zyx.shape
    pad_y = 1 if (Y % 2 == 0) else 0
    pad_x = 1 if (X % 2 == 0) else 0
    if pad_y or pad_x:
        psf_zyx = np.pad(psf_zyx, ((0, 0), (0, pad_y), (0, pad_x)), mode="constant")

    if renormalize:
        s = float(psf_zyx.sum())
        if s > 0:
            psf_zyx = psf_zyx / s
    return psf_zyx