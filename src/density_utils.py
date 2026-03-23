import math
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_torch(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def focal_stack_from_density(rho_zyx, psf_zyx, device=None):
    """
    Full 3D convolution in PyTorch.

    rho_zyx: (Z, Y, X)
    psf_zyx: (Zp, Yp, Xp)
    returns: vol_zyx (Z, Y, X)
    """
    device = get_device(device)

    rho_zyx = as_torch(rho_zyx, device=device, dtype=torch.float32)
    psf_zyx = as_torch(psf_zyx, device=device, dtype=torch.float32)

    if rho_zyx.ndim != 3 or psf_zyx.ndim != 3:
        raise ValueError("rho_zyx and psf_zyx must both be 3D")

    kernel = torch.flip(psf_zyx, dims=(0, 1, 2))

    inp = rho_zyx.unsqueeze(0).unsqueeze(0)   # (1,1,Z,Y,X)
    ker = kernel.unsqueeze(0).unsqueeze(0)    # (1,1,Zp,Yp,Xp)

    Zp, Yp, Xp = psf_zyx.shape
    out = F.conv3d(inp, ker, padding=(Zp // 2, Yp // 2, Xp // 2))

    return out[0, 0]


def _triangle_barycentric_grid(m, device):
    """
    Return barycentric coefficients (a,b,c) for deterministic triangle sampling.
    Number of samples = (m+1)(m+2)/2
    """
    if m <= 0:
        return torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    rows = []
    for i in range(m + 1):
        j = torch.arange(m + 1 - i, device=device, dtype=torch.float32)
        ii = torch.full_like(j, float(i))
        rows.append(torch.stack([ii, j], dim=1))

    ij = torch.cat(rows, dim=0)  # (N,2)
    a = ij[:, 0] / m
    b = ij[:, 1] / m
    c = 1.0 - a - b
    return torch.stack([a, b, c], dim=1)  # (N,3)


def mesh_to_density_zyx(
    mesh_path,
    origin_nm,
    voxel_size_nm_xyz,
    shape_zyx,
    spacing_nm=150.0,
    device=None,
    batch_faces=2048,
):
    """
    Deterministic mesh surface -> density grid rho[z,y,x].
    This corresponds to membrane/surface labeling.

    Batched GPU-friendly version:
    - compute face areas vectorized
    - group faces by same barycentric grid size m
    - process many triangles at once
    - do one scatter_add_ per batch
    """
    device = get_device(device)

    Z, Y, X = shape_zyx
    sx, sy, sz = voxel_size_nm_xyz
    x0, y0, z0 = origin_nm

    mesh = trimesh.load(mesh_path, process=False)
    vertices = torch.as_tensor(
        np.asarray(mesh.vertices, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    faces = torch.as_tensor(
        np.asarray(mesh.faces, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    rho = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    rho_flat = rho.view(-1)

    tris = vertices[faces]  # (F,3,3)
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]

    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

    valid_faces = areas > 0
    if not torch.any(valid_faces):
        return rho

    tris = tris[valid_faces]
    areas = areas[valid_faces]

    n_per_face = torch.clamp(
        torch.ceil(areas / (spacing_nm ** 2)).long(),
        min=1,
    )
    m_per_face = torch.ceil(torch.sqrt(n_per_face.float())).long()

    unique_m = torch.unique(m_per_face)

    for m_val in unique_m.tolist():
        sel = (m_per_face == m_val)
        tris_m = tris[sel]  # (Fm,3,3)

        if tris_m.shape[0] == 0:
            continue

        bary = _triangle_barycentric_grid(int(m_val), device=device)  # (P,3)

        for start in range(0, tris_m.shape[0], batch_faces):
            tri_batch = tris_m[start:start + batch_faces]  # (B,3,3)

            vb0 = tri_batch[:, 0, :]  # (B,3)
            vb1 = tri_batch[:, 1, :]
            vb2 = tri_batch[:, 2, :]

            pts = (
                bary[None, :, 0:1] * vb0[:, None, :] +
                bary[None, :, 1:2] * vb1[:, None, :] +
                bary[None, :, 2:3] * vb2[:, None, :]
            )  # (B,P,3)

            pts = pts.reshape(-1, 3)  # (B*P,3)

            ix = torch.floor((pts[:, 0] - x0) / sx).long()
            iy = torch.floor((pts[:, 1] - y0) / sy).long()
            iz = torch.floor((pts[:, 2] - z0) / sz).long()

            iy = (Y - 1) - iy

            valid = (
                (ix >= 0) & (ix < X) &
                (iy >= 0) & (iy < Y) &
                (iz >= 0) & (iz < Z)
            )

            if not torch.any(valid):
                continue

            ix = ix[valid]
            iy = iy[valid]
            iz = iz[valid]

            flat_idx = iz * (Y * X) + iy * X + ix
            vals = torch.ones(flat_idx.shape[0], dtype=torch.float32, device=device)

            rho_flat.scatter_add_(0, flat_idx, vals)

    return rho


def mesh_filled_to_density_zyx(mesh_path, origin_nm, voxel_size_nm_xyz, shape_zyx, device=None):
    """
    Filled neuron labeling:
    all voxels whose centers lie inside the mesh get fluorophore density.
    This corresponds to cytoplasmic / GFP-like filling.

    Note:
    mesh.contains(...) is still CPU-based via trimesh.
    Output is returned as torch tensor.
    """
    device = get_device(device)

    Z, Y, X = shape_zyx
    sx, sy, sz = voxel_size_nm_xyz
    x0, y0, z0 = origin_nm

    mesh = trimesh.load(mesh_path, process=False)

    xs = x0 + (np.arange(X) + 0.5) * sx
    ys = y0 + (np.arange(Y) + 0.5) * sy
    zs = z0 + (np.arange(Z) + 0.5) * sz

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    inside = mesh.contains(pts)

    rho_yxz = inside.reshape(Y, X, Z).astype(np.float32)
    rho = np.transpose(rho_yxz, (2, 0, 1))
    rho = rho[:, ::-1, :].copy()

    return torch.as_tensor(rho, dtype=torch.float32, device=device)


def gaussian_kernel1d_torch(sigma, truncate=3.0, device=None, dtype=torch.float32):
    """
    Create normalized 1D Gaussian kernel in torch.
    sigma is in voxel units.
    """
    device = get_device(device)

    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)

    radius = int(math.ceil(truncate * sigma))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k


def smooth_density_zyx(rho_zyx, sigma_zyx=(0.6, 0.8, 0.8), normalize_sum=True, device=None):
    """
    Smooth the binned density slightly so it behaves more like a continuous field.

    sigma_zyx: Gaussian smoothing in voxel units (z, y, x)
    normalize_sum: preserve total mass after smoothing
    """
    device = get_device(device)
    rho_zyx = as_torch(rho_zyx, device=device, dtype=torch.float32)

    s0 = rho_zyx.sum()

    sz, sy, sx = sigma_zyx
    kz = gaussian_kernel1d_torch(sz, device=device)
    ky = gaussian_kernel1d_torch(sy, device=device)
    kx = gaussian_kernel1d_torch(sx, device=device)

    x = rho_zyx.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

    wz = kz.view(1, 1, -1, 1, 1)
    x = F.conv3d(x, wz, padding=(kz.numel() // 2, 0, 0))

    wy = ky.view(1, 1, 1, -1, 1)
    x = F.conv3d(x, wy, padding=(0, ky.numel() // 2, 0))

    wx = kx.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, wx, padding=(0, 0, kx.numel() // 2))

    rho_s = x[0, 0]

    if normalize_sum:
        s1 = rho_s.sum()
        if float(s1) > 0.0:
            rho_s = rho_s * (s0 / s1)

    return rho_s


def ensure_psf_odd_xy(psf_zyx, renormalize=False, device=None):
    """
    Pad PSF to odd Y,X if needed (keeps center well-defined).
    Returns torch tensor.
    """
    device = get_device(device)
    psf_zyx = as_torch(psf_zyx, device=device, dtype=torch.float32)

    Z, Y, X = psf_zyx.shape
    pad_y = 1 if (Y % 2 == 0) else 0
    pad_x = 1 if (X % 2 == 0) else 0

    if pad_y or pad_x:
        psf_zyx = F.pad(psf_zyx, (0, pad_x, 0, pad_y, 0, 0), mode="constant", value=0.0)

    if renormalize:
        s = psf_zyx.sum()
        if float(s) > 0.0:
            psf_zyx = psf_zyx / s

    return psf_zyx