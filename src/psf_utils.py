import numpy as np
import tifffile

def load_psf_zyx(path: str) -> np.ndarray:
    """Load PSF TIFF and return float32 array in (Z,Y,X), normalized to sum=1."""
    arr = tifffile.imread(path).astype(np.float32)

    if arr.shape == (64, 64, 13):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape != (13, 64, 64):
        arr = np.moveaxis(arr, int(np.argmin(arr.shape)), 0)

    arr /= (arr.sum() + 1e-12)
    return arr

def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def make_gaussian_psf_matched_zyx(
    shape_zyx=(13, 64, 64),
    lambda_nm=488.0,
    na=1.0,
    n=1.0,
    xy_um_per_px=0.2,
    z_step_um=0.5,
) -> np.ndarray:
    """
    Make a 3D Gaussian PSF in (Z,Y,X) with sigma chosen from diffraction-limited
    FWHM approximations (widefield-like):
        FWHM_xy ≈ 0.61 * lambda / NA
        FWHM_z  ≈ 2 * n * lambda / NA^2
    """
    lam_um = lambda_nm * 1e-3  # nm -> µm

    fwhm_xy_um = 0.61 * lam_um / na
    fwhm_z_um = (2.0 * n * lam_um) / (na ** 2)

    sigma_xy_um = fwhm_to_sigma(fwhm_xy_um)
    sigma_z_um = fwhm_to_sigma(fwhm_z_um)

    sigma_x_px = sigma_xy_um / xy_um_per_px
    sigma_y_px = sigma_xy_um / xy_um_per_px
    sigma_z_px = sigma_z_um / z_step_um

    # Broaden Gaussian slightly to better match Born-Wolf effective spread
    GAUSSIAN_SIGMA_SCALE_XY = 1.3
    GAUSSIAN_SIGMA_SCALE_Z = 1.3

    sigma_x_px *= GAUSSIAN_SIGMA_SCALE_XY
    sigma_y_px *= GAUSSIAN_SIGMA_SCALE_XY
    sigma_z_px *= GAUSSIAN_SIGMA_SCALE_Z

    print("Gaussian PSF matched (approx):")
    print(f"  FWHM_xy ≈ {fwhm_xy_um:.3f} µm -> sigma_xy ≈ {sigma_xy_um:.3f} µm -> {sigma_x_px:.2f} px")
    print(f"  FWHM_z  ≈ {fwhm_z_um:.3f} µm -> sigma_z  ≈ {sigma_z_um:.3f} µm -> {sigma_z_px:.2f} px")

    pz, py, px = shape_zyx
    z = np.arange(pz) - (pz // 2)
    y = np.arange(py) - (py // 2)
    x = np.arange(px) - (px // 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    psf = np.exp(
        -(zz**2 / (2.0 * sigma_z_px**2) +
          yy**2 / (2.0 * sigma_y_px**2) +
          xx**2 / (2.0 * sigma_x_px**2))
    ).astype(np.float32)

    # Normalize total PSF energy
    psf /= (psf.sum() + 1e-12)
    return psf