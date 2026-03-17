import os
import numpy as np
import mitsuba as mi

from src.psf_utils import load_psf_zyx, make_gaussian_psf_matched_zyx
from src.sampling import sample_thickshell_emitters_nm
from src.splat import splat_emitters_with_psf_zyx
from src.io_utils import save_stack_imagej_zyx_u16, save_run_metadata_txt
from src.density_utils import (
    points_to_density_zyx,
    mesh_to_density_zyx,
    mesh_filled_to_density_zyx,
    smooth_density_zyx,
    ensure_psf_odd_xy,
    focal_stack_from_density,
)

mi.set_variant("scalar_rgb")

# -----------------------------
# Paths
# -----------------------------
MESH_PATH = "neuron/mesh_centered.ply"
OUT_DIR = "scripts/zstack_out"
PSF_EM_TIF = "scripts/psf_bornwolf_488nm_NA1_xy200nm_z500nm_65x65x13.tif"

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
# Thick-shell emitters (used only for splat mode
# or old points-based density mode)
# -----------------------------
NUM_EMITTERS = 4_000_000
THICKNESS_UM = 2.0
JITTER_UM = 0.3
RNG_SEED = 0

# -----------------------------
# Labeling / density settings
# -----------------------------
LABELING_MODE = "filled"   # "membrane" or "filled"
MESH_DENSITY_SPACING_NM = 150.0
DENSITY_SOURCE = "mesh"      # "mesh" or "points"

# -----------------------------
# PSF selection + optics
# -----------------------------
USE_GAUSSIAN_PSF = False   # True: Gaussian in Python, False: Born&Wolf TIFF from Fiji
LAMBDA_NM = 488.0
NA = 1.0
REF_INDEX = 1.33
GAUSS_PSF_SHAPE_ZYX = (13, 65, 65)

# -----------------------------
# Choose image formation mode
# -----------------------------
# "splat": original emitter splatting
# "density": density field + focal-plane PSF rendering
MODE = "density"

# -----------------------------
# Density regularization
# -----------------------------
DENSITY_SMOOTH_SIGMA_ZYX = (0.6, 0.8, 0.8)
DENSITY_NORMALIZE_SUM = True

# -----------------------------
# Intensity variation
# -----------------------------
USE_INTENSITY_VARIATION = True
INTENSITY_VAR_STD = 0.10
INTENSITY_VAR_SIGMA_ZYX = (2.0, 4.0, 4.0)
INTENSITY_VAR_SEED = 0

print("=== SETTINGS ===")
print(f"XY_UM_PER_PX={XY_UM_PER_PX} µm/px, Z_STEP_UM={Z_STEP_UM} µm")
print(f"ROI={USE_ROI} ({ROI_SIZE_UM_X}×{ROI_SIZE_UM_Y} µm), center={ROI_CENTER_MODE}, margin={MARGIN}")
print(f"NUM_EMITTERS={NUM_EMITTERS:,}, THICKNESS_UM={THICKNESS_UM}, JITTER_UM={JITTER_UM}")
print(f"LABELING_MODE={LABELING_MODE}")
print(f"DENSITY_SOURCE={DENSITY_SOURCE}, MESH_DENSITY_SPACING_NM={MESH_DENSITY_SPACING_NM}")
print(f"PSF mode: {'GAUSSIAN' if USE_GAUSSIAN_PSF else 'BORN&WOLF (Fiji TIFF)'}")
print(f"Optics: lambda={LAMBDA_NM} nm, NA={NA}, n={REF_INDEX}")
print(f"Image formation MODE={MODE}")
print(f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}")
print(f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}")
print(f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}, INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}")
print("===============")

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
# 2) Only generate emitter points if needed
# -----------------------------
u_in = v_in = z_in = None
points_roi_nm = None

if MODE == "splat" or DENSITY_SOURCE == "points":
    points = sample_thickshell_emitters_nm(
        mesh_path=MESH_PATH,
        num_emitters=NUM_EMITTERS,
        thickness_um=THICKNESS_UM,
        jitter_um=JITTER_UM,
        rng_seed=RNG_SEED,
    )

    print(f"Generated {points.shape[0]:,} thick-shell emitters")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (x - xmin) / (xmax - xmin) * (W - 1)
    v = (y - ymin) / (ymax - ymin) * (H - 1)
    v = (H - 1) - v

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_in = u[inside]
    v_in = v[inside]
    z_in = z[inside]

    print(f"Emitters inside ROI/FOV: {len(u_in):,}")

    points_roi_nm = np.stack([x[inside], y[inside], z_in], axis=1).astype(np.float32)

# -----------------------------
# 3) PSF selection
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
    psf_tag = "gaussian_matched"
else:
    psf_eff = load_psf_zyx(PSF_EM_TIF)
    psf_tag = "bornwolf_fiji"

psf_eff = ensure_psf_odd_xy(psf_eff, renormalize=True)

# -----------------------------
# 4) Full-depth Z stack
# -----------------------------
depth_nm_total = float(zmax - zmin)
NUM_SLICES = int(np.ceil(depth_nm_total / Z_STEP_NM)) + 1

print(f"Neuron depth: {depth_nm_total / 1000.0:.2f} µm -> NUM_SLICES={NUM_SLICES}")

# -----------------------------
# 5) Image formation -> volume (Z,Y,X)
# -----------------------------
if MODE == "splat":
    vol = splat_emitters_with_psf_zyx(
        u=u_in,
        v=v_in,
        z_nm=z_in,
        zmin_nm=zmin,
        num_slices=NUM_SLICES,
        H=H,
        W=W,
        z_step_nm=Z_STEP_NM,
        psf_zyx=psf_eff,
    )

elif MODE == "density":
    voxel_x_nm = XY_UM_PER_PX * 1000.0
    voxel_y_nm = XY_UM_PER_PX * 1000.0
    voxel_z_nm = Z_STEP_NM

    origin_nm = (xmin, ymin, zmin)

    if DENSITY_SOURCE == "mesh":
        if LABELING_MODE == "membrane":
            rho = mesh_to_density_zyx(
                mesh_path=MESH_PATH,
                origin_nm=origin_nm,
                voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
                shape_zyx=(NUM_SLICES, H, W),
                spacing_nm=MESH_DENSITY_SPACING_NM,
            )
        elif LABELING_MODE == "filled":
            rho = mesh_filled_to_density_zyx(
                mesh_path=MESH_PATH,
                origin_nm=origin_nm,
                voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
                shape_zyx=(NUM_SLICES, H, W),
            )
        else:
            raise ValueError("LABELING_MODE must be 'membrane' or 'filled'")

    elif DENSITY_SOURCE == "points":
        rho = points_to_density_zyx(
            points_nm=points_roi_nm,
            origin_nm=origin_nm,
            voxel_size_nm_xyz=(voxel_x_nm, voxel_y_nm, voxel_z_nm),
            shape_zyx=(NUM_SLICES, H, W),
        )
    else:
        raise ValueError("DENSITY_SOURCE must be 'mesh' or 'points'")

    print(
        "rho_raw:",
        rho.shape,
        "sum=",
        float(rho.sum()),
        "max=",
        float(rho.max()),
    )

    rho = smooth_density_zyx(
        rho,
        sigma_zyx=DENSITY_SMOOTH_SIGMA_ZYX,
        normalize_sum=DENSITY_NORMALIZE_SUM,
    )

    print(
        "rho_smooth:",
        rho.shape,
        "sum=",
        float(rho.sum()),
        "max=",
        float(rho.max()),
    )

    # Apply smooth random intensity variation to fluorophore density
    if USE_INTENSITY_VARIATION:
        rng = np.random.default_rng(INTENSITY_VAR_SEED)

        weights = rng.normal(
            loc=1.0,
            scale=INTENSITY_VAR_STD,
            size=rho.shape,
        ).astype(np.float32)

        weights = smooth_density_zyx(
            weights,
            sigma_zyx=INTENSITY_VAR_SIGMA_ZYX,
            normalize_sum=False,
        )

        weights = np.clip(weights, 0.0, None)

        rho = rho * weights

        print(
            "rho_varied:",
            rho.shape,
            "sum=",
            float(rho.sum()),
            "max=",
            float(rho.max()),
        )

    vol = focal_stack_from_density(rho, psf_eff)

else:
    raise ValueError("MODE must be 'splat' or 'density'")

print("psf:", psf_eff.shape, "sum=", float(psf_eff.sum()))
print("vol:", vol.shape, "min/max=", float(vol.min()), float(vol.max()))

# -----------------------------
# 6) Normalize + Save Z-stack + metadata
# -----------------------------
tag = f"EMonly_{LABELING_MODE}_ROI{int(ROI_SIZE_UM_X)}x{int(ROI_SIZE_UM_Y)}um_{psf_tag}_{MODE}_{DENSITY_SOURCE}"

vol_f = vol.astype(np.float32, copy=False)
vol_f /= (vol_f.max() + 1e-12)

np.clip(vol_f, 0.0, 1.0, out=vol_f)
vol_f *= 65535.0
stack_u16 = vol_f.astype(np.uint16)

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
    f"MODE={MODE}",
    f"LABELING_MODE={LABELING_MODE}",
    f"DENSITY_SOURCE={DENSITY_SOURCE}",
    f"MESH_DENSITY_SPACING_NM={MESH_DENSITY_SPACING_NM}",
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
    f"DENSITY_SMOOTH_SIGMA_ZYX={DENSITY_SMOOTH_SIGMA_ZYX}",
    f"DENSITY_NORMALIZE_SUM={DENSITY_NORMALIZE_SUM}",
    f"USE_INTENSITY_VARIATION={USE_INTENSITY_VARIATION}",
    f"INTENSITY_VAR_STD={INTENSITY_VAR_STD}",
    f"INTENSITY_VAR_SIGMA_ZYX={INTENSITY_VAR_SIGMA_ZYX}",
    f"INTENSITY_VAR_SEED={INTENSITY_VAR_SEED}",
]

meta_txt = save_run_metadata_txt(OUT_DIR, tag, meta_lines)
print("Saved metadata:", meta_txt)

print("Done.")