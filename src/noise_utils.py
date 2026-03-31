import torch


def add_microscopy_noise_torch(
    vol,
    peak_photons=500.0,
    read_noise_std=5.0,
    seed=0,
):
    """
    Add realistic microscopy noise using PyTorch (GPU-compatible)

    Parameters
    ----------
    vol : torch.Tensor (Z,Y,X)
        Input volume
    peak_photons : float
        Max photon scaling
    read_noise_std : float
        Gaussian read noise std
    seed : int
        RNG seed
    """
    if seed is not None:
        torch.manual_seed(seed)

    vol = vol.float()

    # Normalize to [0,1]
    vol = vol / (vol.max() + 1e-12)

    # Convert to photon counts
    photons = vol * peak_photons

    # Poisson noise (shot noise)
    noisy = torch.poisson(photons)

    # Gaussian read noise
    noisy = noisy + torch.randn_like(noisy) * read_noise_std

    # Clip negatives
    noisy = torch.clamp(noisy, min=0.0)

    return noisy