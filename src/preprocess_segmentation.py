"""
Segmentation Preprocessing (Branch B).

Wraps the existing preprocess logic: outlier clip + normalise [0,1] + Gaussian smooth sigma=0.5.
Used only for ROI masking (not for FFT analysis).
"""

import logging
import numpy as np
from scipy import ndimage

from src.pipeline_config import SegPreprocRecord, SegmentationPreprocConfig

logger = logging.getLogger(__name__)


def preprocess_segmentation(image: np.ndarray,
                            config: SegmentationPreprocConfig = None) -> SegPreprocRecord:
    """Segmentation preprocessing (Branch B).

    Steps:
    1. Outlier clip at configured percentile
    2. Normalise to [0, 1]
    3. Gaussian smooth with sigma=0.5

    No hard gate on this branch -- it is only used for ROI.
    """
    if config is None:
        config = SegmentationPreprocConfig()

    diagnostics = {
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
    }

    img = image.astype(np.float32)

    # Clip outliers
    p_low = np.percentile(img, config.clip_percentile)
    p_high = np.percentile(img, 100 - config.clip_percentile)
    img = np.clip(img, p_low, p_high)

    # Normalise to [0, 1]
    imin, imax = img.min(), img.max()
    if imax - imin > 1e-8:
        img = (img - imin) / (imax - imin)
    else:
        img = np.zeros_like(img)

    # Gaussian smooth
    if config.smooth_sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=config.smooth_sigma)
        diagnostics["smoothed"] = True
        diagnostics["smooth_sigma"] = config.smooth_sigma
    else:
        diagnostics["smoothed"] = False

    diagnostics["output_min"] = float(np.min(img))
    diagnostics["output_max"] = float(np.max(img))

    return SegPreprocRecord(
        image_seg=img.astype(np.float32),
        diagnostics=diagnostics,
    )
