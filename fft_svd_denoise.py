import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def fft_svd_denoise(image, keep_percentage=0.1):
    # Apply FFT on the image
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)

    # Apply SVD
    U, S, Vt = np.linalg.svd(f_shifted, full_matrices=False)
    thresh = int(keep_percentage * len(S))
    S[thresh:] = 0
    denoised_f_shifted = U @ np.diag(S) @ Vt

    # Apply inverse FFT
    denoised_f = ifft2(fftshift(denoised_f_shifted))
    denoised_image = np.abs(denoised_f)

    return denoised_image
