"""CLIP-based concept matching."""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import open_clip


class CLIPConceptMatcher:
    """Match image regions to semantic concepts using CLIP.
    
    Uses OpenCLIP's vision-language model to find the best matching
    concept from a vocabulary for each image region.
    """
    
    def __init__(self, concept_vocabulary: List[str],
                 model_name: str = "ViT-L-14",
                 pretrained: str = "openai",
                 device: str = "cuda",
                 prompt_template: str = "a photo of a {concept}"):
        """Initialize CLIP concept matcher.
        
        Args:
            concept_vocabulary: List of concept strings
            model_name: CLIP model architecture
            pretrained: Pretrained weights source
            device: Device to run on
            prompt_template: Template for text prompts
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.concept_vocabulary = concept_vocabulary
        self.prompt_template = prompt_template
        
        print(f"Loading CLIP model {model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        print("Precomputing concept embeddings...")
        self.concept_embeddings = self._encode_concepts(concept_vocabulary)
        print(f"✓ {len(concept_vocabulary)} concepts ready.")
    
    def _encode_concepts(self, concepts: List[str]) -> np.ndarray:
        """Encode all concepts to text embeddings."""
        embeddings = []
        
        with torch.no_grad():
            for concept in concepts:
                text = self.prompt_template.format(concept=concept)
                tokens = self.tokenizer([text]).to(self.device)
                embedding = self.model.encode_text(tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding[0].cpu().numpy())
        
        return np.array(embeddings)
    
    def encode_image(self, image_np: np.ndarray) -> np.ndarray:
        """Encode image to CLIP embedding."""
        image_pil = Image.fromarray(image_np.astype('uint8'))
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding[0].cpu().numpy()
    
    def match_region(self, region_image_np: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find best matching concepts for a region.
        
        Args:
            region_image_np: Cropped region image (H, W, 3)
            top_k: Number of top matches to return
            
        Returns:
            List of (concept, similarity_score) tuples
        """
        region_embedding = self.encode_image(region_image_np)
        similarities = np.dot(self.concept_embeddings, region_embedding)
        top_indices = np.argsort(-similarities)[:top_k]
        
        return [(self.concept_vocabulary[i], float(similarities[i])) for i in top_indices]
