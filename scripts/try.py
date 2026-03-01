# make_zstack_realpsf_roi_EMonly_thickshell_imagejstack.py
# Updated according to Andreas' feedback + fixes Fiji Z/T issue:
# 1) Use PSF_em only (baseline)
# 2) Mesh is NOT watertight -> thick-shell emitters (surface sample + push inward + jitter)
# 3) ROI in XY, full depth in Z
# 4) Save as a proper ImageJ Z-stack in ONE write call so Fiji reads Z correctly (Z=NUM_SLICES, T=1)

import os
import numpy as np
import mitsuba as mi
import tifffile
import trimesh

mi.set_variant("scalar_rgb")

# -----------------------------
# Paths
# -----------------------------
MESH_PATH = "../neuron/mesh_centered.ply"
OUT_DIR = "zstack_out"
PSF_EM_TIF = r"C:\mitsuba_work\scripts\psf_confocal_488nm_NA1_64x64x13.tif"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Physical sampling (REAL)
# -----------------------------
XY_UM_PER_PX = 0.2   # µm/px
Z_STEP_UM = 0.5      # µm/slice
Z_STEP_NM = Z_STEP_UM * 1000.0

# -----------------------------
# ROI settings
# -----------------------------
USE_ROI = True
ROI_SIZE_UM_X = 200.0
ROI_SIZE_UM_Y = 200.0
ROI_CENTER_MODE = "bbox_center"
MARGIN = 0.05

# -----------------------------
# Thick-shell fluorophore settings (mesh is not watertight)
# -----------------------------
NUM_EMITTERS = 2_000_000   # try 2_000_000 if still speckly
THICKNESS_UM = 2.0         # try 3–5 µm if needed
JITTER_UM = 0.3            # try 0.5 µm if needed

RNG_SEED = 0

print("=== SETTINGS ===")
print(f"XY_UM_PER_PX={XY_UM_PER_PX} µm/px, Z_STEP_UM={Z_STEP_UM} µm")
print(f"ROI={USE_ROI} ({ROI_SIZE_UM_X}×{ROI_SIZE_UM_Y} µm), center={ROI_CENTER_MODE}, margin={MARGIN}")
print(f"NUM_EMITTERS={NUM_EMITTERS:,}, THICKNESS_UM={THICKNESS_UM}, JITTER_UM={JITTER_UM}")
print(f"PSF_EM_TIF={PSF_EM_TIF}")
print("===============")

# -----------------------------
# Helpers
# -----------------------------
def load_psf_zyx(path: str) -> np.ndarray:
    """Load PSF TIFF and return float32 array in (Z,Y,X), normalized to max=1."""
    arr = tifffile.imread(path).astype(np.float32)
    if arr.shape == (64, 64, 13):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape != (13, 64, 64):
        arr = np.moveaxis(arr, int(np.argmin(arr.shape)), 0)
    arr /= (arr.max() + 1e-12)
    return arr

def save_psf_and_mips(psf: np.ndarray, prefix: str):
    """Save PSF stack + XY and XZ MIPs for Fiji inspection."""
    pz, py, px = psf.shape
    psf_out = os.path.join(OUT_DIR, f"{prefix}_{pz}x{py}x{px}.tif")
    tifffile.imwrite(psf_out, (np.clip(psf, 0, 1) * 65535).astype(np.uint16))

    psf_xy = psf.max(axis=0)  # (Y,X)
    psf_xz = psf.max(axis=1)  # (Z,X)
    tifffile.imwrite(
        os.path.join(OUT_DIR, f"{prefix}_mip_xy.tif"),
        (psf_xy / (psf_xy.max() + 1e-12) * 65535).astype(np.uint16),
    )
    tifffile.imwrite(
        os.path.join(OUT_DIR, f"{prefix}_mip_xz.tif"),
        (psf_xz / (psf_xz.max() + 1e-12) * 65535).astype(np.uint16),
    )
    print(f"Saved {prefix}: {psf_out}")

# -----------------------------
# 1) Load mesh bbox with Mitsuba (units: nm)
# -----------------------------
mesh = mi.load_dict({"type": "ply", "filename": MESH_PATH})
bbox = mesh.bbox()
xmin0, ymin0, zmin = float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])
xmax0, ymax0, zmax = float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])
print(f"Mesh bbox (nm): x[{xmin0:.1f},{xmax0:.1f}] y[{ymin0:.1f},{ymax0:.1f}] z[{zmin:.1f},{zmax:.1f}]")

# Expand bbox slightly for ROI centering
xrange = xmax0 - xmin0
yrange = ymax0 - ymin0
xmin_m = xmin0 - MARGIN * xrange
xmax_m = xmax0 + MARGIN * xrange
ymin_m = ymin0 - MARGIN * yrange
ymax_m = ymax0 + MARGIN * yrange

# Define ROI bbox
if USE_ROI:
    if ROI_CENTER_MODE != "bbox_center":
        raise ValueError("ROI_CENTER_MODE not recognized. Use 'bbox_center'.")
    cx_nm = 0.5 * (xmin_m + xmax_m)
    cy_nm = 0.5 * (ymin_m + ymax_m)
    halfx_nm = (ROI_SIZE_UM_X * 1000.0) * 0.5
    halfy_nm = (ROI_SIZE_UM_Y * 1000.0) * 0.5
    xmin = cx_nm - halfx_nm
    xmax = cx_nm + halfx_nm
    ymin = cy_nm - halfy_nm
    ymax = cy_nm + halfy_nm
    print(f"Using ROI bbox (nm): x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}]")
else:
    xmin, xmax, ymin, ymax = xmin_m, xmax_m, ymin_m, ymax_m
    print("Using full bbox (with margin) for rendering.")

# Compute image size from physical pixel size
xspan_um = (xmax - xmin) / 1000.0
yspan_um = (ymax - ymin) / 1000.0
W = int(np.ceil(xspan_um / XY_UM_PER_PX)) + 1
H = int(np.ceil(yspan_um / XY_UM_PER_PX)) + 1
actual_x = xspan_um / (W - 1)
actual_y = yspan_um / (H - 1)

print(f"Auto image size: W={W}, H={H}")
print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")
print(f"Actual pixel size: {actual_x:.4f} µm/px × {actual_y:.4f} µm/px (target {XY_UM_PER_PX} µm/px)")

# -----------------------------
# 2) Generate thick-shell emitters using Trimesh
# -----------------------------
tm = trimesh.load(MESH_PATH, force="mesh", process=False)
rng = np.random.default_rng(RNG_SEED)

surf_pts, face_idx = trimesh.sample.sample_surface(tm, NUM_EMITTERS)
face_normals = tm.face_normals[face_idx]

THICKNESS_NM = THICKNESS_UM * 1000.0
JITTER_NM = JITTER_UM * 1000.0

depth_nm = rng.uniform(0.0, THICKNESS_NM, size=(NUM_EMITTERS, 1)).astype(np.float32)
pts_in = surf_pts.astype(np.float32) - face_normals.astype(np.float32) * depth_nm
pts_in += rng.normal(0.0, JITTER_NM, size=pts_in.shape).astype(np.float32)

points = pts_in
print(f"Generated {points.shape[0]:,} thick-shell emitters")

# -----------------------------
# 3) Map emitters to pixels + filter to ROI
# -----------------------------
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

u = (x - xmin) / (xmax - xmin) * (W - 1)
v = (y - ymin) / (ymax - ymin) * (H - 1)
v = (H - 1) - v

inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u = u[inside]
v = v[inside]
z = z[inside]
print(f"Emitters inside ROI/FOV: {len(u):,}")

# -----------------------------
# 4) Load PSF_em only (baseline) + save PSF for inspection
# -----------------------------
psf_eff = load_psf_zyx(PSF_EM_TIF)
psf_eff /= (psf_eff.max() + 1e-12)

pz, py, px = psf_eff.shape
cz, cy, cx = pz // 2, py // 2, px // 2
print("Loaded PSF_em:", psf_eff.shape, "center:", (cz, cy, cx))
save_psf_and_mips(psf_eff, "psf_em_only")

# -----------------------------
# 5) Full-depth Z stack
# -----------------------------
depth_nm = float(zmax - zmin)
NUM_SLICES = int(np.ceil(depth_nm / Z_STEP_NM)) + 1
print(f"Neuron depth: {depth_nm/1000.0:.2f} µm -> NUM_SLICES={NUM_SLICES} at {Z_STEP_UM} µm step")

# -----------------------------
# 6) Splat into volume (Z,Y,X)
# -----------------------------
vol = np.zeros((NUM_SLICES, H, W), dtype=np.float16)

k = np.round((z - zmin) / Z_STEP_NM).astype(np.int32)
valid = (k >= 0) & (k < NUM_SLICES)

u_i = np.round(u[valid]).astype(np.int32)
v_i = np.round(v[valid]).astype(np.int32)
k_i = k[valid]

print(f"Emitters used after Z indexing: {len(k_i):,}")

for x0, y0, z0 in zip(u_i, v_i, k_i):
    oz0 = z0 - cz
    oz1 = oz0 + pz
    oy0 = y0 - cy
    oy1 = oy0 + py
    ox0 = x0 - cx
    ox1 = ox0 + px

    vz0 = max(0, oz0); vz1 = min(NUM_SLICES, oz1)
    vy0 = max(0, oy0); vy1 = min(H, oy1)
    vx0 = max(0, ox0); vx1 = min(W, ox1)

    pz0 = vz0 - oz0; pz1 = pz0 + (vz1 - vz0)
    py0 = vy0 - oy0; py1 = py0 + (vy1 - vy0)
    px0 = vx0 - ox0; px1 = px0 + (vx1 - vx0)

    vol[vz0:vz1, vy0:vy1, vx0:vx1] += psf_eff[pz0:pz1, py0:py1, px0:px1].astype(np.float16)

# -----------------------------
# 7) Save as ImageJ Z-stack (single write so Fiji reads Z correctly)
# -----------------------------
tag = f"EMonly_thickshell_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_imagej"
tiff_path = os.path.join(OUT_DIR, f"zstack_{tag}.tif")
mip_path = os.path.join(OUT_DIR, f"mip_{tag}.tif")
meta_txt = os.path.join(OUT_DIR, f"metadata_{tag}.txt")

# Normalize for visualization
vol_f = vol.astype(np.float32)
vol_f /= (vol_f.max() + 1e-12)
stack_u16 = (np.clip(vol_f, 0, 1) * 65535).astype(np.uint16)

tifffile.imwrite(
    tiff_path,
    stack_u16,  # (Z,Y,X)
    imagej=True,
    resolution=(1.0 / XY_UM_PER_PX, 1.0 / XY_UM_PER_PX),  # px per µm
    metadata={
        "axes": "ZYX",
        "spacing": Z_STEP_UM,
        "unit": "um",
    },
)
print("Saved TIFF stack (ImageJ Z-stack):", tiff_path, "shape:", stack_u16.shape)

# MIP for quick check
mip = vol_f.max(axis=0)
mip_u16 = (np.clip(mip, 0, 1) * 65535).astype(np.uint16)
tifffile.imwrite(mip_path, mip_u16)
print("Saved MIP:", mip_path)

# Sidecar metadata
with open(meta_txt, "w") as f:
    f.write("=== Render metadata ===\n")
    f.write("MODE=PSF_em_only (baseline)\n")
    f.write("EMITTERS=thick_shell_surface_sampled\n")
    f.write(f"XY_UM_PER_PX={XY_UM_PER_PX}\n")
    f.write(f"Z_STEP_UM={Z_STEP_UM}\n")
    f.write("AXES=ZYX\n")
    f.write(f"W={W}\nH={H}\nNUM_SLICES={NUM_SLICES}\n")
    f.write(f"FOV_um_x={xspan_um}\nFOV_um_y={yspan_um}\n")
    f.write(f"Neuron_depth_um={depth_nm/1000.0}\n")
    f.write(f"ROI_used={USE_ROI}\n")
    f.write(f"ROI_size_um_x={ROI_SIZE_UM_X}\n")
    f.write(f"ROI_size_um_y={ROI_SIZE_UM_Y}\n")
    f.write(f"THICKNESS_UM={THICKNESS_UM}\n")
    f.write(f"JITTER_UM={JITTER_UM}\n")
    f.write(f"NUM_EMITTERS={NUM_EMITTERS}\n")
    f.write(f"PSF_shape_ZYX={psf_eff.shape}\n")
print("Saved sidecar metadata:", meta_txt)

print("Done.")