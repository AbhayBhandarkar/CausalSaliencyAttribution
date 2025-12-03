"""Counterfactual validation for concept importance."""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple


class CounterfactualValidator:
    """Validate concept importance through counterfactual masking.
    
    Measures the causal impact of a region by comparing classifier
    confidence before and after masking (blurring) the region.
    """
    
    def __init__(self, classifier, method: str = "blur", device: str = "cuda"):
        """Initialize counterfactual validator.
        
        Args:
            classifier: Classifier instance
            method: Masking method ("blur")
            device: Device to run on
        """
        self.classifier = classifier
        self.method = method
        self.device = device
    
    def validate_concept_importance(self, image_np: np.ndarray, image_tensor: torch.Tensor,
                                    region_bbox: Tuple[int, int, int, int],
                                    concept_name: str,
                                    original_confidence: float) -> float:
        """Measure concept importance via counterfactual masking.
        
        Args:
            image_np: Original image (H, W, 3) in [0, 255]
            image_tensor: Normalized image tensor (1, 3, H, W)
            region_bbox: Region bounding box (x_min, y_min, x_max, y_max)
            concept_name: Name of concept being tested
            original_confidence: Original prediction confidence
            
        Returns:
            Importance score (confidence drop ratio) in [0, 1]
        """
        masked_image_np = self._blur_region(image_np, region_bbox)
        
        masked_image_pil = Image.fromarray(masked_image_np.astype('uint8'))
        masked_tensor = self.classifier.preprocess(masked_image_pil)
        _, masked_confidence, _ = self.classifier.get_prediction(masked_tensor)
        
        importance = (original_confidence - masked_confidence) / (original_confidence + 1e-8)
        return max(0.0, float(importance))
    
    def _blur_region(self, image_np: np.ndarray, bbox: Tuple[int, int, int, int],
                     kernel_size: int = 51, sigma: float = 25) -> np.ndarray:
        """Blur a region using Gaussian blur.
        
        Args:
            image_np: RGB image (H, W, 3)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            kernel_size: Gaussian kernel size (must be odd)
            sigma: Gaussian sigma
            
        Returns:
            Image with blurred region (H, W, 3)
        """
        image_masked = image_np.copy()
        x_min, y_min, x_max, y_max = bbox
        
        region = image_masked[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return image_masked
        
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Clamp kernel to region size
        k_size = min(kernel_size, min(region.shape[0], region.shape[1]))
        if k_size % 2 == 0:
            k_size -= 1
        k_size = max(1, k_size)
        
        blurred_region = cv2.GaussianBlur(region, (k_size, k_size), sigma)
        image_masked[y_min:y_max, x_min:x_max] = blurred_region
        
        return image_masked
