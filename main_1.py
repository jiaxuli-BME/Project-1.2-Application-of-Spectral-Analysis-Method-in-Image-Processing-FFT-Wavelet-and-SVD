#This file is for FFT + SVD Noise Reduction and Wavelet + SVD Compression

import matplotlib.pyplot as plt
from utils import load_image
from fft_svd_denoise import fft_svd_denoise
from wavelet_svd_compress import wavelet_svd_compress

def main():
    # Load image
    image_path = r"E:\HuaweiMoveData\Users\HUAWEI\Desktop\Victoria_Falls.jpg"
    gray_image = load_image(image_path)

    # Noise reduction using FFT and SVD
    fft_svd_denoised_image = fft_svd_denoise(gray_image, keep_percentage=0.1)

    # Compression using Wavelet and SVD
    wavelet_svd_compressed_image = wavelet_svd_compress(gray_image, compression_ratio=0.1)

    # Plot results
    plt.figure(figsize=(16, 16))

    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(fft_svd_denoised_image, cmap='gray')
    plt.title('FFT + SVD Denoised Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(wavelet_svd_compressed_image, cmap='gray')
    plt.title('Wavelet + SVD Compressed Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
