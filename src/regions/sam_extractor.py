"""SAM (Segment Anything Model) region extractor."""

import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMRegionExtractor:
    """Region extraction using Segment Anything Model (SAM).
    
    Uses SAM's automatic mask generation to extract semantically 
    meaningful regions from images, filtered by saliency scores.
    """
    
    def __init__(self, checkpoint_path: str = None, model_type: str = "vit_h",
                 device: str = "cuda", points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88, stability_score_thresh: float = 0.95):
        """Initialize SAM region extractor.
        
        Args:
            checkpoint_path: Path to SAM checkpoint (.pth file)
            model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run on
            points_per_side: Grid density for mask generation
            pred_iou_thresh: IoU threshold for mask prediction
            stability_score_thresh: Stability score threshold
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything not installed. Run: pip install segment-anything")
        
        self.device = device
        self.model_type = model_type
        self.mask_generator = None
        
        if checkpoint_path and Path(checkpoint_path).exists() and torch.cuda.is_available():
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
        else:
            print(f"Warning: SAM not initialized. Checkpoint: {checkpoint_path}, CUDA: {torch.cuda.is_available()}")
    
    def extract_regions(self, image_np: np.ndarray, saliency_map: np.ndarray,
                        threshold: float = 0.3, top_k: int = 5,
                        min_region_size: int = 100) -> List[Dict]:
        """Extract high-saliency regions using SAM.
        
        Args:
            image_np: RGB image array (H, W, 3) in [0, 255]
            saliency_map: Saliency map (H, W) in [0, 1]
            threshold: Minimum mean saliency for region inclusion
            top_k: Number of top regions to return
            min_region_size: Minimum region size in pixels
            
        Returns:
            List of region dictionaries with mask, bbox, and saliency info
        """
        if self.mask_generator is None:
            return self._fallback_regions(image_np.shape[:2], saliency_map, threshold, top_k)
        
        masks = self.mask_generator.generate(image_np)
        regions = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            pixel_count = mask.sum()
            
            if pixel_count < min_region_size:
                continue
            
            mean_saliency = saliency_map[mask].mean()
            if mean_saliency < threshold:
                continue
            
            x, y, w, h = mask_data['bbox']
            regions.append({
                'segment_id': i,
                'mask': mask,
                'bbox': (int(x), int(y), int(x + w), int(y + h)),
                'mean_saliency': float(mean_saliency),
                'pixel_count': int(pixel_count),
                'area': float(mask_data.get('area', pixel_count)),
                'stability_score': float(mask_data.get('stability_score', 0.0)),
                'predicted_iou': float(mask_data.get('predicted_iou', 0.0))
            })
        
        return sorted(regions, key=lambda r: r['mean_saliency'], reverse=True)[:top_k]
    
    def _fallback_regions(self, shape: Tuple[int, int], saliency_map: np.ndarray,
                          threshold: float, top_k: int) -> List[Dict]:
        """Fallback grid-based regions when SAM unavailable."""
        H, W = shape
        regions = []
        grid_size = 4
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_min, y_max = i * H // grid_size, (i + 1) * H // grid_size
                x_min, x_max = j * W // grid_size, (j + 1) * W // grid_size
                
                mask = np.zeros((H, W), dtype=bool)
                mask[y_min:y_max, x_min:x_max] = True
                
                mean_saliency = saliency_map[mask].mean()
                if mean_saliency > threshold:
                    regions.append({
                        'segment_id': i * grid_size + j,
                        'mask': mask,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'mean_saliency': float(mean_saliency),
                        'pixel_count': int(mask.sum()),
                        'area': float(mask.sum())
                    })
        
        return sorted(regions, key=lambda r: r['mean_saliency'], reverse=True)[:top_k]
