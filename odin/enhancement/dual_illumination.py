import cv2
import numpy as np

from odin.base import CONFIG
from odin.enhancement.lime_enhancer import correct_underexposure, create_gaussian_kernel
from odin.image_processing import resize_image
from odin.utilities import timer


def fuse_exposure_corrected_images(
    image: np.ndarray,
    underexposure_corrected: np.ndarray,
    overexposure_corrected: np.ndarray,
    contrast_weight: float = 1,
    saturation_weight: float = 1,
    well_exposedness_weight: float = 1,
) -> np.ndarray:
    """
    Fuses multiple exposure-corrected images using the Mertens's method.

    This function takes an input image and two exposure-corrected images (one for under-exposure and one for
    over-exposure) and fuses them using Mertens's method, which is a fusion method that considers contrast,
    saturation, and well-exposedness.

    Parameters:
        image (np.ndarray):
            The input image to be enhanced.
        underexposure_corrected (np.ndarray):
            The under-exposure corrected image, same dimension as `image`.
        overexposure_corrected (np.ndarray):
            The over-exposure corrected image, same dimension as `image`.
        contrast_weight (float, optional):
            Parameter for controlling the influence of Mertens's contrast measure. Defaults to 1.
        saturation_weight (float, optional):
            Parameter for controlling the influence of Mertens's saturation measure. Defaults to 1.
        well_exposedness_weight (float, optional):
            Parameter for controlling the influence of Mertens's well-exposedness measure. Defaults to 1.

    Returns:
        np.ndarray:
            The fused image, same dimension as `image`.
    """

    merge_mertens = cv2.createMergeMertens(contrast_weight, saturation_weight, well_exposedness_weight)
    images = [
        np.clip(x * 255, 0, 255).astype("uint8")
        for x in [image, underexposure_corrected, overexposure_corrected]
    ]
    fused_images = merge_mertens.process(images)

    return fused_images


@timer(return_execution_time=True)
def dual_illumination_estimation(image: np.ndarray) -> np.ndarray:
    """
    Estimates and corrects dual illumination in an image using a dual exposure correction method.

    This function first normalizes the input image, then corrects underexposure and overexposure separately.
    The corrected images are then fused using a method that considers contrast, saturation, and
    well-exposedness. The process involves creating a Gaussian kernel for spatial affinity-based Gaussian
    weights.

    Parameters:
        image (np.ndarray):
            The input image to be corrected.

    Returns:
        np.ndarray:
            The corrected image with dual illumination estimated and corrected.
    """

    dual_config = CONFIG.enhancement.dual

    original_shape = image.shape
    if dual_config.optimal_dimensions is not None:
        dimensions = tuple(dual_config.optimal_dimensions)
        image = resize_image(image, dimensions=dimensions, interpolation=cv2.INTER_CUBIC)

    normalized_image = image.astype(float) / 255.0
    if len(normalized_image.shape) == 2:
        normalized_image = normalized_image[..., np.newaxis]

    kernel = create_gaussian_kernel(dual_config.sigma)
    corrected_underexposed_image = correct_underexposure(
        normalized_image, dual_config.gamma, dual_config.alpha, kernel, dual_config.epsilon
    )

    inverse_image_normalized = 1 - normalized_image
    corrected_overexposed_image = 1 - correct_underexposure(
        inverse_image_normalized, dual_config.gamma, dual_config.alpha, kernel, dual_config.epsilon
    )

    corrected_image = fuse_exposure_corrected_images(
        normalized_image,
        corrected_underexposed_image,
        corrected_overexposed_image,
        dual_config.contrast_weight,
        dual_config.saturation_weight,
        dual_config.well_exposedness_weight,
    )

    clipped_image = np.clip(corrected_image * 255, 0, 255).astype("uint8")

    if dual_config.optimal_dimensions is not None:
        clipped_image = resize_image(
            clipped_image, dimensions=original_shape[:2], interpolation=cv2.INTER_CUBIC
        )

    return clipped_image
