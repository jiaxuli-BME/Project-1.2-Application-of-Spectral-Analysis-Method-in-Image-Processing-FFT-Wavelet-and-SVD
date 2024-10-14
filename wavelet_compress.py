import numpy as np
import pywt

def wavelet_compress(image, compression_ratio=0.1):
    # Apply wavelet decomposition
    coeffs = pywt.wavedec2(image, 'db1', level=3)

    # Threshold the wavelet coefficients
    coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)
    Csort = np.sort(np.abs(coeff_arr.ravel()))
    thresh = Csort[int(np.floor((1-compression_ratio)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    compressed_coeff_arr = coeff_arr * ind

    # Convert back to wavelet coefficients
    coeffs_compressed = pywt.array_to_coeffs(compressed_coeff_arr, coeffs_slices, output_format='wavedec2')

    # Apply inverse wavelet transform
    compressed_image = pywt.waverec2(coeffs_compressed, 'db1')

    return compressed_image
