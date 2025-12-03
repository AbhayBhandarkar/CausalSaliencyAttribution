"""Concept ranking based on causal importance."""

import torch
import numpy as np
from typing import List, Dict


class ConceptRanker:
    """Rank concepts based on causal importance scores."""
    
    def __init__(self, validator, concept_matcher, feature_extractor):
        """Initialize concept ranker.
        
        Args:
            validator: CounterfactualValidator instance
            concept_matcher: CLIPConceptMatcher instance
            feature_extractor: LocalFeatureExtractor instance
        """
        self.validator = validator
        self.matcher = concept_matcher
        self.feature_extractor = feature_extractor
    
    def rank_concepts_for_image(self, image_np: np.ndarray, image_tensor: torch.Tensor,
                                regions: List[Dict], original_confidence: float,
                                top_k: int = 5) -> List[Dict]:
        """Rank concepts by causal importance for each region.
        
        Args:
            image_np: Original image (H, W, 3)
            image_tensor: Normalized image tensor (1, 3, H, W)
            regions: List of region dictionaries
            original_confidence: Original prediction confidence
            top_k: Number of top concepts to return
            
        Returns:
            List of concept results sorted by importance
        """
        concept_results = []
        
        for i, region in enumerate(regions):
            bbox = region.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            
            try:
                # Crop and match concept
                crop = self.feature_extractor.crop_region(image_np, bbox)
                matches = self.matcher.match_region(crop, top_k=1)
                
                if not matches:
                    continue
                
                concept, similarity = matches[0]
                
                # Validate importance via counterfactual
                importance = self.validator.validate_concept_importance(
                    image_np=image_np,
                    image_tensor=image_tensor,
                    region_bbox=bbox,
                    concept_name=concept,
                    original_confidence=original_confidence
                )
                
                concept_results.append({
                    'region_id': i,
                    'bbox': bbox,
                    'concept': concept,
                    'similarity_score': float(similarity),
                    'importance_raw': float(importance),
                    'saliency_magnitude': float(region.get('mean_saliency', 0.0)),
                    'pixel_count': region.get('pixel_count', 0)
                })
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                continue
        
        return sorted(concept_results, key=lambda x: x.get('importance_raw', 0.0), reverse=True)[:top_k]
