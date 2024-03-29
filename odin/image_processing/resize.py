from typing import Optional, Tuple

import cv2
import numpy as np


def resize_image(
    image: np.ndarray,
    scale: Optional[float] = None,
    dimensions: Optional[Tuple[int, ...]] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resizes an image using a scaling factor or specific dimensions.

    Parameters:
        image (np.ndarray):
            The image to be resized.
        scale (Optional[float]):
            The scaling factor to resize the image. If provided, width and height must be None.
        width (Optional[int]):
            The new width of the image. If provided, height must be provided and scale must be None.
        height (Optional[int]):
            The new height of the image. If provided, width must be provided and scale must be None.
        interpolation (int):
            The interpolation method to use. Defaults to cv2.INTER_AREA.

    Returns:
        np.ndarray:
            The resized image.

    Raises:
        ValueError:
            If neither scale, width, nor height is provided, or if only one of them is provided without the
            other.
    """

    if dimensions is None:
        if scale is None:
            raise ValueError("Either scale or width and height must be provided.")

        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dimensions = (width, height)
    else:
        if len(dimensions) > 2:
            raise ValueError("Dimensions must be of length 2.")

    resized_image = cv2.resize(image, dimensions, interpolation=interpolation)

    return resized_image
