import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def fft_denoise(image, keep_percentage=0.1):
    # Apply FFT on the image
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)

    # Threshold the frequency coefficients
    thresh = int(keep_percentage * f_shifted.size)
    sorted_coeffs = np.sort(np.abs(f_shifted.ravel()))
    threshold = sorted_coeffs[thresh]
    mask = np.abs(f_shifted) > threshold
    denoised_f_shifted = f_shifted * mask

    # Apply inverse FFT
    denoised_f = ifft2(fftshift(denoised_f_shifted))
    denoised_image = np.abs(denoised_f)

    return denoised_image
