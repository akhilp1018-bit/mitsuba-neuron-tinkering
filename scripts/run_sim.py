import os
import numpy as np
import mitsuba as mi
import tifffile
import trimesh

from src.psf_utils import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.sampling import sample_thickshell_emitters_nm
from src.splat import splat_emitters_with_psf_zyx
from src.io_utils import save_stack_imagej_zyx_u16, save_run_metadata_txt

mi.set_variant("scalar_rgb")

# -----------------------------
# Paths
# -----------------------------
MESH_PATH = "neuron/mesh_centered.ply"
OUT_DIR = "scripts/zstack_out"
PSF_EM_TIF = "scripts/psf_bornwolf_widefield_488nm_NA1_64x64x13.tif"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Physical sampling
# -----------------------------
XY_UM_PER_PX = 0.2   # 200 nm/px
Z_STEP_UM = 0.5      # 500 nm/slice
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
# Thick-shell emitters
# -----------------------------
NUM_EMITTERS = 1_000_000
THICKNESS_UM = 2.0
JITTER_UM = 0.3
RNG_SEED = 0

# -----------------------------
# PSF selection + optics (match Fiji)
# -----------------------------
USE_GAUSSIAN_PSF = False   # True: Gaussian in Python, False: Born&Wolf TIFF from Fiji

LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.0
GAUSS_PSF_SHAPE_ZYX = (13, 64, 64)

print("=== SETTINGS ===")
print(f"XY_UM_PER_PX={XY_UM_PER_PX} µm/px, Z_STEP_UM={Z_STEP_UM} µm")
print(f"ROI={USE_ROI} ({ROI_SIZE_UM_X}×{ROI_SIZE_UM_Y} µm), center={ROI_CENTER_MODE}, margin={MARGIN}")
print(f"NUM_EMITTERS={NUM_EMITTERS:,}, THICKNESS_UM={THICKNESS_UM}, JITTER_UM={JITTER_UM}")
print(f"PSF mode: {'GAUSSIAN' if USE_GAUSSIAN_PSF else 'BORN&WOLF (Fiji TIFF)'}")
print(f"Optics: lambda={LAMBDA_NM} nm, NA={NA}, n={REF_INDEX}")
print("===============")

# -----------------------------
# Optional helper: save PSF stack + MIPs for inspection in Fiji
# (Safe to delete later if you don't need it)
# -----------------------------
def save_psf_and_mips(psf: np.ndarray, prefix: str):
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
    print(f"Saved PSF + MIPs: {prefix}")

# -----------------------------
# 1) Load mesh bbox with Mitsuba (nm)
# -----------------------------
mesh = mi.load_dict({"type": "ply", "filename": MESH_PATH})
bbox = mesh.bbox()
xmin0, ymin0, zmin = float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])
xmax0, ymax0, zmax = float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])
print(f"Mesh bbox (nm): x[{xmin0:.1f},{xmax0:.1f}] y[{ymin0:.1f},{ymax0:.1f}] z[{zmin:.1f},{zmax:.1f}]")

# Expand bbox for ROI centering
xrange_nm = xmax0 - xmin0
yrange_nm = ymax0 - ymin0
xmin_m = xmin0 - MARGIN * xrange_nm
xmax_m = xmax0 + MARGIN * xrange_nm
ymin_m = ymin0 - MARGIN * yrange_nm
ymax_m = ymax0 + MARGIN * yrange_nm

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
print(f"Auto image size: W={W}, H={H}")
print(f"FOV: {xspan_um:.2f} µm × {yspan_um:.2f} µm")

# -----------------------------
# 2) Thick-shell emitters (from src/sampling.py)
# -----------------------------
points = sample_thickshell_emitters_nm(
    mesh_path=MESH_PATH,
    num_emitters=NUM_EMITTERS,
    thickness_um=THICKNESS_UM,
    jitter_um=JITTER_UM,
    rng_seed=RNG_SEED,
)
print(f"Generated {points.shape[0]:,} thick-shell emitters")

# -----------------------------
# 3) Map emitters to pixels + filter ROI
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
# 4) PSF selection
# -----------------------------
if USE_GAUSSIAN_PSF:
    psf_eff = make_gaussian_psf_matched_zyx(
        shape_zyx=GAUSS_PSF_SHAPE_ZYX,
        lambda_nm=LAMBDA_NM,
        na=NA,
        n=REF_INDEX,
        xy_um_per_px=XY_UM_PER_PX,
        z_step_um=Z_STEP_UM,
    )
    save_psf_and_mips(psf_eff, "psf_gaussian_matched")
    psf_tag = "gaussian_matched"
else:
    psf_eff = load_psf_zyx(PSF_EM_TIF)
    save_psf_and_mips(psf_eff, "psf_bornwolf_fiji")
    psf_tag = "bornwolf_fiji"

# -----------------------------
# 5) Full-depth Z stack
# -----------------------------
depth_nm_total = float(zmax - zmin)
NUM_SLICES = int(np.ceil(depth_nm_total / Z_STEP_NM)) + 1
print(f"Neuron depth: {depth_nm_total/1000.0:.2f} µm -> NUM_SLICES={NUM_SLICES}")

# -----------------------------
# 6) Splat into volume (Z,Y,X)
# -----------------------------
vol = splat_emitters_with_psf_zyx(
    u=u,
    v=v,
    z_nm=z,
    zmin_nm=zmin,
    num_slices=NUM_SLICES,
    H=H,
    W=W,
    z_step_nm=Z_STEP_NM,
    psf_zyx=psf_eff,
)

# -----------------------------
# 7) Normalize + Save Z-stack + metadata
# -----------------------------
tag = f"EMonly_thickshell_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_{psf_tag}"

vol_f = vol.astype(np.float32)
vol_f /= (vol_f.max() + 1e-12)
stack_u16 = (np.clip(vol_f, 0, 1) * 65535).astype(np.uint16)

tiff_path = save_stack_imagej_zyx_u16(
    out_dir=OUT_DIR,
    tag=tag,
    stack_u16_zyx=stack_u16,
    xy_um_per_px=XY_UM_PER_PX,
    z_step_um=Z_STEP_UM,
)
print("Saved stack:", tiff_path, "shape:", stack_u16.shape)

meta_lines = [
    "=== Render metadata ===",
    f"PSF_MODE={psf_tag}",
    f"lambda_nm={LAMBDA_NM}",
    f"NA={NA}",
    f"refractive_index={REF_INDEX}",
    f"XY_UM_PER_PX={XY_UM_PER_PX}",
    f"Z_STEP_UM={Z_STEP_UM}",
    f"NUM_EMITTERS={NUM_EMITTERS}",
    f"THICKNESS_UM={THICKNESS_UM}",
    f"JITTER_UM={JITTER_UM}",
    f"W={W}",
    f"H={H}",
    f"NUM_SLICES={NUM_SLICES}",
]
meta_txt = save_run_metadata_txt(OUT_DIR, tag, meta_lines)
print("Saved metadata:", meta_txt)
print("Done.")