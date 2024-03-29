from odin.utilities import timer

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet

from odin.base import CONFIG

class ImageEnhancer:
    
    @timer(return_execution_time=False)
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to an image.

        This function applies histogram equalization to the input image, which can improve the contrast of the
        image.

        Parameters:
            image (np.ndarray):
                The input image as a NumPy array.

        Returns:
            np.ndarray:
                The equalized image as a NumPy array.
        """

        equalized_image = cv2.equalizeHist(image)

        return equalized_image
    
    @timer(return_execution_time=False)
    def CLAHE_contrast_limited_adaptative_histogram_equalization(input_image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

        This function applies CLAHE to the input image, which can improve the contrast of the image by limiting
        the contrast enhancement to the local neighborhood of each pixel.

        Parameters:
            input_image (np.ndarray):
                The input image as a NumPy array.

        Returns:
            np.ndarray:
                The image with CLAHE applied as a NumPy array.
        """

        clahe_config = CONFIG.enhancement.clahe
        clahe_transformer = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
        equalized_image = clahe_transformer.apply(input_image)

        return equalized_image
    
    
    @timer(return_execution_time=False)  
    def total_variance_denoising(image: np.ndarray) -> np.ndarray:
        total_variance_config = CONFIG.enhancement.total_variance
        denoised_image = denoise_tv_chambolle(
            image,
            weight=total_variance_config.weight,
        )

        clipped_image = np.clip(denoised_image * 255, 0, 255).astype("uint8")

        return clipped_image
    
    @timer(return_execution_time=False)
    def bilateral_filter(image: np.ndarray) -> np.ndarray:
        """
        Apply a bilateral filter to an image.

        This function applies a bilateral filter to the input image, which can reduce noise while preserving
        edges.

        Parameters:
            image (np.ndarray):
                The input image as a NumPy array.

        Returns:
            np.ndarray:
                The filtered image as a NumPy array.
        """

        bilateral_config = CONFIG.enhancement.bilateral
        filtered_image = cv2.bilateralFilter(
            image,
            d=bilateral_config.diameter,
            sigmaColor=bilateral_config.sigma_color,
            sigmaSpace=bilateral_config.sigma_space,
        )

        return filtered_image

    @timer(return_execution_time=False)
    def denoWavelet(img1):
        return denoise_wavelet(img1, mode="soft", rescale_sigma=True)