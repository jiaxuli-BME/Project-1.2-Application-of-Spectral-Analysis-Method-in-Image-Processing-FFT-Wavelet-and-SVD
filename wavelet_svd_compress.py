import numpy as np
import pywt
from sklearn.decomposition import TruncatedSVD

def wavelet_svd_compress(image, compression_ratio=0.1):
    # Apply wavelet decomposition
    coeffs = pywt.wavedec2(image, 'db1', level=3)

    # Flatten coefficient arrays
    coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    # Reshape the array to have at least 2 columns
    rows, cols = coeff_arr.shape
    reshaped_arr = coeff_arr.reshape(rows * cols // 2, 2)

    # Apply SVD
    n_components = max(2, int(compression_ratio * min(reshaped_arr.shape)))
    svd = TruncatedSVD(n_components=n_components)
    compressed_arr = svd.fit_transform(reshaped_arr)

    # Reshape back to original shape
    compressed_coeff_arr = compressed_arr.reshape(coeff_arr.shape)

    # Convert back to wavelet coefficients
    coeffs_compressed = pywt.array_to_coeffs(compressed_coeff_arr, coeffs_slices, output_format='wavedec2')

    # Apply inverse wavelet transform
    compressed_image = pywt.waverec2(coeffs_compressed, 'db1')

    return compressed_image
