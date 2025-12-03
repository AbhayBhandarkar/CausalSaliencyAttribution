"""Feature extraction for image regions."""

import numpy as np
import cv2
from typing import Tuple


class LocalFeatureExtractor:
    """Extract cropped regions with context for concept matching."""
    
    def __init__(self, context_margin: float = 0.2):
        """Initialize feature extractor.
        
        Args:
            context_margin: Fraction of region size to add as context margin
        """
        self.context_margin = context_margin
    
    def crop_region(self, image_np: np.ndarray, bbox: Tuple[int, int, int, int],
                    target_size: int = 224) -> np.ndarray:
        """Crop region with context margin.
        
        Args:
            image_np: RGB image array (H, W, 3)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            target_size: Size to resize crop to
            
        Returns:
            Cropped and resized image (target_size, target_size, 3)
        """
        x_min, y_min, x_max, y_max = bbox
        h, w = image_np.shape[:2]
        
        region_w, region_h = x_max - x_min, y_max - y_min
        margin_x = int(region_w * self.context_margin)
        margin_y = int(region_h * self.context_margin)
        
        # Expand with margin (clamped to image bounds)
        x_min_ctx = max(0, x_min - margin_x)
        y_min_ctx = max(0, y_min - margin_y)
        x_max_ctx = min(w, x_max + margin_x)
        y_max_ctx = min(h, y_max + margin_y)
        
        crop = image_np[y_min_ctx:y_max_ctx, x_min_ctx:x_max_ctx]
        
        if crop.size == 0:
            crop = image_np[y_min:y_max, x_min:x_max]
        
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
