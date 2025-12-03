"""Image classifier wrapper for pretrained models."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional
import numpy as np
from PIL import Image


class Classifier:
    """Wrapper for pretrained ImageNet classification models.
    
    Supports ResNet-50, ResNet-101, and ViT-B-16 architectures with
    ImageNet-1K pretrained weights.
    """
    
    SUPPORTED_MODELS = {"resnet50", "resnet101", "vit_b_16"}
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True, 
                 device: str = "cuda"):
        """Initialize classifier.
        
        Args:
            model_name: Model architecture (resnet50, resnet101, vit_b_16)
            pretrained: Whether to load ImageNet pretrained weights
            device: Device to run model on (cuda/cpu)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Load pretrained model architecture."""
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet50(weights=weights)
        elif model_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet101(weights=weights)
        elif model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            return models.vit_b_16(weights=weights)
        raise ValueError(f"Unknown model: {model_name}")
    
    def preprocess(self, image_pil: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor.
        
        Args:
            image_pil: PIL Image (224x224 recommended)
            
        Returns:
            Normalized tensor (1, 3, H, W) on device
        """
        img_array = np.array(image_pil)
        if img_array.ndim == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        return (img_tensor - self.mean) / self.std
    
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through model.
        
        Args:
            image_tensor: Normalized input tensor (1, 3, H, W)
            
        Returns:
            Logits tensor (1, 1000)
        """
        with torch.no_grad():
            return self.model(image_tensor)
    
    def get_prediction(self, image_tensor: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """Get top-1 prediction with confidence.
        
        Args:
            image_tensor: Normalized input tensor (1, 3, H, W)
            
        Returns:
            Tuple of (predicted_class_id, confidence, all_probabilities)
        """
        logits = self.predict(image_tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        return top_class.item(), top_prob.item(), probs[0]
    
    def get_target_layer(self, layer_name: Optional[str] = None) -> nn.Module:
        """Get target layer for gradient-based saliency methods.
        
        Args:
            layer_name: Layer name (e.g., "layer4" for ResNet)
            
        Returns:
            Target layer module
        """
        if layer_name is None:
            if "resnet" in self.model_name:
                layer_name = "layer4"
            elif "vit" in self.model_name:
                return self.model.encoder.layers[-1].ln_1
            else:
                raise ValueError(f"Cannot auto-detect layer for {self.model_name}")
        
        if "resnet" in self.model_name:
            return getattr(self.model, layer_name)
        
        # Generic attribute navigation
        module = self.model
        for part in layer_name.split('.'):
            module = getattr(module, part)
        return module
