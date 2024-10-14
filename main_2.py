#This file is for FFT Noise Reduction and Wavelet Compression

import matplotlib.pyplot as plt
from utils import load_image
from fft_denoise import fft_denoise
from wavelet_compress import wavelet_compress

def main():
    # Load image
    image_path = r"E:\HuaweiMoveData\Users\HUAWEI\Desktop\Victoria_Falls.jpg"
    gray_image = load_image(image_path)

    # Noise reduction using FFT
    fft_denoised_image = fft_denoise(gray_image, keep_percentage=0.1)

    # Compression using Wavelet
    wavelet_compressed_image = wavelet_compress(gray_image, compression_ratio=0.1)

    # Plot results
    plt.figure(figsize=(16, 16))

    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(fft_denoised_image, cmap='gray')
    plt.title('FFT Denoised Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(wavelet_compressed_image, cmap='gray')
    plt.title('Wavelet Compressed Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
