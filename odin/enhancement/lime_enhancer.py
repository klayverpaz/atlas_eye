import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from odin.base import CONFIG
from odin.image_processing import resize_image
from odin.utilities import timer


def create_gaussian_kernel(sigma: float, kernel_size: int = 15) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel.

    Parameters:
        sigma (float):
            The standard deviation of the Gaussian distribution.
        kernel_size (int):
            The size of the kernel. Default is 15.

    Returns:
        np.ndarray:
            A 2D Gaussian kernel.
    """

    y, x = np.mgrid[0:kernel_size, 0:kernel_size]

    distance_from_center = np.linalg.norm(
        np.stack((x, y), axis=-1) - np.array([kernel_size // 2, kernel_size // 2]), axis=-1
    )

    gaussian_kernel = np.exp(-0.5 * (distance_from_center**2) / (sigma**2))

    return gaussian_kernel


def compute_weights(
    illumination_map: np.ndarray, direction: int, kernel: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    """
    Computes the weights for refining the illumination map.

    Parameters:
        illumination_map (np.ndarray):
            The initial illumination map to be refined.
        direction (int):
            The direction of the weights. 1 for horizontal, 0 for vertical.
        kernel (np.ndarray):
            A Gaussian kernel with standard deviation sigma.
        epsilon (float):
            Small constant to avoid computation instability. Default is 1e-3.

    Returns:
        np.ndarray:
            Weights according to the specified direction, with the same dimensions as the illumination map.
    """

    gradient_map = cv2.Sobel(illumination_map, cv2.CV_64F, int(direction == 1), int(direction == 0), ksize=1)
    convolved_gradient = convolve(gradient_map, kernel, mode="constant")
    total_weight = convolve(np.ones_like(illumination_map), kernel, mode="constant")
    adjusted_weights = total_weight / (np.abs(convolved_gradient) + epsilon)
    adjusted_weights = adjusted_weights / (np.abs(gradient_map) + epsilon)

    return adjusted_weights


def refine_illumination_map(
    illumination_map: np.ndarray, gamma: float, alpha: float, kernel: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    """
    Refines the illumination map by solving an optimization problem.

    This function computes the weights for refining the illumination map in both horizontal and vertical
    directions, constructs a sparse matrix representing the optimization problem, and solves it to obtain
    the refined illumination map.

    Parameters:
        illumination_maps (np.ndarray):
            The initial illumination maps to be refined.
        gamma (float):
            A parameter to adjust the gamma correction of the refined illumination map.
        alpha (float):
            A parameter to control the strength of the regularization term in the optimization problem to
            preserve the overall structure and smooth the textural details.
        kernel (np.ndarray):
            A Gaussian kernel with standard deviation sigma.
        epsilon (float):
            A small constant to avoid computation instability. Default is 1e-3.

    Returns:
        np.ndarray:
            The refined illumination map, with the same dimensions as the input illumination map.
    """

    weights_x = compute_weights(illumination_map, direction=1, kernel=kernel, epsilon=epsilon)
    weights_y = compute_weights(illumination_map, direction=0, kernel=kernel, epsilon=epsilon)

    height, width = illumination_map.shape
    n_pixels = height * width
    row_indices, column_indices, data = [], [], []
    for index in range(n_pixels):
        row, column = index // width, index % width
        neighbors = [(row - 1, column, 0), (row + 1, column, 0), (row, column - 1, 1), (row, column + 1, 1)]
        valid_neighbors = [(r, c, x) for r, c, x in neighbors if 0 <= r < height and 0 <= c < width]

        diagonal = 0
        for neighbor_row, neighbor_column, is_x_direction in valid_neighbors:
            neighbor_index = neighbor_row * width + neighbor_column
            weight = (
                weights_x[neighbor_row, neighbor_column]
                if is_x_direction
                else weights_y[neighbor_row, neighbor_column]
            )
            row_indices.append(index)
            column_indices.append(neighbor_index)
            data.append(-weight)
            diagonal += weight

        row_indices.append(index)
        column_indices.append(index)
        data.append(diagonal)

    sparse_weights_matrix = csr_matrix((data, (row_indices, column_indices)), shape=(n_pixels, n_pixels))

    identity = diags([np.ones(n_pixels)], [0])
    matrix = identity + alpha * sparse_weights_matrix

    flattened_map = illumination_map.copy().flatten()
    refined_illumination_map = spsolve(csr_matrix(matrix), flattened_map).reshape((height, width))
    refined_illumination_map = np.clip(refined_illumination_map, epsilon, 1) ** gamma

    return refined_illumination_map


def correct_underexposure(
    image: np.ndarray, gamma: float, alpha: float, kernel: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    """
    Corrects underexposure in an image by refining the illumination map.

    This function first estimates the illumination map of the input image and then refines it. The refined
    illumination map is used to correct the underexposure in the image.

    Parameters:
        image (np.ndarray):
            The input image with shape (height, width, channels).
        gamma (float):
            A parameter for the illumination map refinement.
        alpha (float):
            A parameter for the illumination map refinement.
        kernel (np.ndarray):
            A Gaussian kernel with standard deviation sigma.
        epsilon (float):
            A small constant to avoid division by zero. Defaults to 1e-3.

    Returns:
        np.ndarray:
            The corrected image with underexposure.
    """

    illumination_map = np.max(image, axis=-1)
    refined_illumination_map = refine_illumination_map(illumination_map, gamma, alpha, kernel, epsilon)
    expanded_illumination_map = np.repeat(refined_illumination_map[..., None], image.shape[2], axis=-1)

    corrected_image = image / expanded_illumination_map

    return corrected_image


@timer(return_execution_time=True)
def low_light_image_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhances a low-light image by correcting underexposure and applying gamma correction.

    This function normalizes the input image, corrects underexposure using a refined illumination map, and
    applies gamma correction.

    Parameters:
        image (np.ndarray):
            The input low-light image to be enhanced.

    Returns:
        np.ndarray:
            The enhanced image with corrected exposure.
    """

    lime_config = CONFIG.enhancement.lime

    original_shape = image.shape
    if lime_config.optimal_dimensions is not None:
        dimensions = tuple(lime_config.optimal_dimensions[:2])
        image = resize_image(image, dimensions=dimensions, interpolation=cv2.INTER_CUBIC)

    image_normalized = image.astype(float) / 255.0
    if len(image_normalized.shape) == 2:
        image_normalized = image_normalized[..., np.newaxis]

    kernel = create_gaussian_kernel(lime_config.sigma)
    corrected_image = correct_underexposure(
        image_normalized, lime_config.gamma, lime_config.alpha, kernel, lime_config.epsilon
    )

    clipped_image = np.clip(corrected_image * 255, 0, 255).astype("uint8")

    if lime_config.optimal_dimensions is not None:
        clipped_image = resize_image(
            clipped_image, dimensions=original_shape[:2], interpolation=cv2.INTER_CUBIC
        )

    return clipped_image
