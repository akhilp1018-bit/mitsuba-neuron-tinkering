import torch


def add_microscopy_noise_torch(
    vol,
    peak_photons=500.0,
    read_noise_std=5.0,
    seed=0,
    gaussian_chunk_slices=16,
):
    """
    Add microscopy noise using PyTorch on GPU with lower memory usage.

    Parameters
    ----------
    vol : torch.Tensor (Z,Y,X)
        Input volume
    peak_photons : float
        Max photon scaling. Lower -> stronger shot noise
    read_noise_std : float
        Gaussian read noise std
    seed : int or None
        RNG seed
    gaussian_chunk_slices : int
        Number of z-slices per chunk for Gaussian noise
    """
    if seed is not None:
        torch.manual_seed(seed)

    vol = vol.float()

    # normalize to [0,1]
    vol = vol / (vol.max() + 1e-12)

    # scale to photon counts
    vol = vol * peak_photons

    # Poisson shot noise
    noisy = torch.poisson(vol)

    # Gaussian read noise in chunks to avoid full-volume randn_like allocation
    if read_noise_std > 0:
        Z = noisy.shape[0]
        for z0 in range(0, Z, gaussian_chunk_slices):
            z1 = min(z0 + gaussian_chunk_slices, Z)
            chunk = noisy[z0:z1]
            chunk.add_(torch.randn_like(chunk) * read_noise_std)

    noisy.clamp_(min=0.0)
    return noisy