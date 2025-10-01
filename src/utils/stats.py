

import numpy as np
from scipy.stats import skew, kurtosis


def mask_stats(image: np.ndarray, mask: np.ndarray, bins: int = 100) -> dict:
    """Compute summary statistics of image voxel values under a mask.

    Args:
        image (numpy.ndarray): 3D array with signal values.
        mask (numpy.ndarray): 3D binary mask (same shape as image).
        bins (int): Number of bins for histogram-based mode estimation.

    Returns:
        dict: Dictionary with summary statistics.
    """
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")
    
    # Extract voxel values under the mask
    values = image[mask > 0]
    
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "mode": None,
            "skewness": None,
            "kurtosis": None,
        }
    
    # Histogram-based mode
    hist, edges = np.histogram(values, bins=bins)
    mode_idx = np.argmax(hist)
    mode_value = (edges[mode_idx] + edges[mode_idx + 1]) / 2.0  # bin center

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),  # sample std
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "mode": float(mode_value),
        "skewness": float(skew(values)),
        "kurtosis": float(kurtosis(values)),  # excess kurtosis (0 = normal distribution)
    }
