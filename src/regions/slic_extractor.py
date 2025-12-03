"""SLIC superpixel region extractor."""

import numpy as np
from skimage.segmentation import slic
from typing import List, Dict


class SLICRegionExtractor:
    """Region extraction using SLIC superpixels.
    
    Uses Simple Linear Iterative Clustering to segment images into
    perceptually uniform superpixels, filtered by saliency scores.
    """
    
    def __init__(self, n_segments: int = 50, compactness: float = 10):
        """Initialize SLIC region extractor.
        
        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color and spatial proximity (higher = more compact)
        """
        self.n_segments = n_segments
        self.compactness = compactness
    
    def extract_regions(self, image_np: np.ndarray, saliency_map: np.ndarray,
                        threshold: float = 0.3, top_k: int = 5,
                        min_region_size: int = 100) -> List[Dict]:
        """Extract high-saliency regions using SLIC superpixels.
        
        Args:
            image_np: RGB image array (H, W, 3) in [0, 255]
            saliency_map: Saliency map (H, W) in [0, 1]
            threshold: Minimum mean saliency for region inclusion
            top_k: Number of top regions to return
            min_region_size: Minimum region size in pixels
            
        Returns:
            List of region dictionaries with mask, bbox, and saliency info
        """
        superpixel_map = slic(
            image_np,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
            convert2lab=True
        )
        
        regions = []
        for segment_id in np.unique(superpixel_map):
            mask = (superpixel_map == segment_id)
            pixel_count = mask.sum()
            
            if pixel_count < min_region_size:
                continue
            
            mean_saliency = saliency_map[mask].mean()
            if mean_saliency < threshold:
                continue
            
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            regions.append({
                'segment_id': int(segment_id),
                'mask': mask,
                'bbox': (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1),
                'mean_saliency': float(mean_saliency),
                'pixel_count': int(pixel_count),
                'area': float(pixel_count)
            })
        
        return sorted(regions, key=lambda r: r['mean_saliency'], reverse=True)[:top_k]
