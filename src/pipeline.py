"""End-to-end semantic saliency attribution pipeline."""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional
from pathlib import Path

from src.classifier import Classifier
from src.saliency.gradcam import GradCAM
from src.regions.sam_extractor import SAMRegionExtractor
from src.regions.slic_extractor import SLICRegionExtractor
from src.regions.feature_extractor import LocalFeatureExtractor
from src.concepts.clip_matcher import CLIPConceptMatcher
from src.concepts.vocabulary import ConceptVocabulary
from src.validation.counterfactual import CounterfactualValidator
from src.validation.ranker import ConceptRanker


class SemanticSaliencyPipeline:
    """Complete pipeline for semantic attribution of saliency maps.
    
    Combines saliency generation, region extraction, concept discovery,
    and counterfactual validation into a single end-to-end system.
    """
    
    def __init__(self, config: Dict):
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with model, saliency, regions,
                    concepts, and counterfactual settings
        """
        self.config = config
        self.device = config['model'].get('device', 'cuda')
        
        self._init_classifier()
        self._init_saliency()
        self._init_region_extractor()
        self._init_concept_matcher()
        self._init_validator()
        self._init_ranker()
    
    def _init_classifier(self) -> None:
        """Initialize image classifier."""
        self.classifier = Classifier(
            model_name=self.config['model']['classifier'],
            pretrained=self.config['model']['pretrained'],
            device=self.device
        )
    
    def _init_saliency(self) -> None:
        """Initialize saliency generator."""
        target_layer = self.classifier.get_target_layer(
            self.config['saliency'].get('target_layer')
        )
        self.saliency_gen = GradCAM(self.classifier.model, target_layer)
    
    def _init_region_extractor(self) -> None:
        """Initialize region extractor (SAM or SLIC)."""
        method = self.config['regions']['method']
        
        if method == 'sam':
            self.region_extractor = SAMRegionExtractor(
                checkpoint_path=self.config['regions'].get('sam_checkpoint'),
                model_type=self.config['regions'].get('sam_model_type', 'vit_h'),
                device=self.device
            )
        elif method == 'slic':
            self.region_extractor = SLICRegionExtractor(
                n_segments=self.config['regions'].get('slic_n_segments', 50),
                compactness=self.config['regions'].get('slic_compactness', 10)
            )
        else:
            raise ValueError(f"Unsupported region method: {method}")
        
        self.feature_extractor = LocalFeatureExtractor(
            context_margin=self.config['regions'].get('context_margin', 0.2)
        )
    
    def _init_concept_matcher(self) -> None:
        """Initialize concept matcher."""
        vocab_path = self.config['concepts'].get('vocabulary_path')
        
        if vocab_path and Path(vocab_path).exists():
            vocabulary = ConceptVocabulary.from_file(vocab_path)
        else:
            vocabulary = ConceptVocabulary()
        
        self.concept_matcher = CLIPConceptMatcher(
            concept_vocabulary=vocabulary.concepts,
            model_name=self.config['concepts'].get('clip_model', 'ViT-L-14'),
            device=self.device
        )
    
    def _init_validator(self) -> None:
        """Initialize counterfactual validator."""
        self.validator = CounterfactualValidator(
            classifier=self.classifier,
            method=self.config['counterfactual']['method'],
            device=self.device
        )
    
    def _init_ranker(self) -> None:
        """Initialize concept ranker."""
        self.ranker = ConceptRanker(
            validator=self.validator,
            concept_matcher=self.concept_matcher,
            feature_extractor=self.feature_extractor
        )
    
    def process_image(self, image_input, validate: bool = True) -> Dict:
        """Process a single image through the complete pipeline.
        
        Args:
            image_input: Path to image (str) or PIL Image
            validate: Whether to run counterfactual validation
            
        Returns:
            Dictionary with prediction, saliency, regions, and concepts
        """
        try:
            # Load and preprocess image
            if isinstance(image_input, str):
                image_pil = Image.open(image_input).convert('RGB')
            else:
                image_pil = image_input.convert('RGB')
            
            image_pil = image_pil.resize((224, 224))
            image_np = np.array(image_pil)
            image_tensor = self.classifier.preprocess(image_pil)
            
            # Step 1: Classification
            predicted_class, confidence, probs = self.classifier.get_prediction(image_tensor)
            
            # Step 2: Saliency generation
            saliency_map = self.saliency_gen.generate(image_tensor, predicted_class, image_size=224)
            
            # Step 3: Region extraction
            regions = self.region_extractor.extract_regions(
                image_np=image_np,
                saliency_map=saliency_map,
                threshold=self.config['regions'].get('saliency_threshold', 0.3),
                top_k=self.config['regions'].get('top_k_regions', 5),
                min_region_size=self.config['regions'].get('min_region_size', 100)
            )
            
            if not regions:
                return {
                    'image_np': image_np,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'saliency_map': saliency_map,
                    'regions': [],
                    'concepts': [],
                    'error': 'No high-saliency regions found'
                }
            
            # Step 4: Concept ranking with validation
            if validate:
                concepts = self.ranker.rank_concepts_for_image(
                    image_np=image_np,
                    image_tensor=image_tensor,
                    regions=regions,
                    original_confidence=confidence,
                    top_k=self.config['regions'].get('top_k_regions', 5)
                )
            else:
                concepts = []
            
            return {
                'image_np': image_np,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'saliency_map': saliency_map,
                'regions': regions,
                'concepts': concepts
            }
            
        except Exception as e:
            import traceback
            return {'error': str(e), 'traceback': traceback.format_exc()}
