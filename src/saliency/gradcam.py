"""Grad-CAM saliency map generation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations for CNN predictions by computing
    gradient-weighted activation maps from convolutional layers.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
               via Gradient-based Localization", ICCV 2017.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (e.g., ResNet-50)
            target_layer: Target convolutional layer for CAM extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Handle Sequential layers
        target = self.target_layer[-1] if isinstance(self.target_layer, torch.nn.Sequential) else self.target_layer
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate(self, image_tensor: torch.Tensor, target_class: int, 
                 image_size: int = 224) -> np.ndarray:
        """Generate Grad-CAM saliency map.
        
        Args:
            image_tensor: Normalized input image tensor (1, 3, H, W)
            target_class: Target class index for explanation
            image_size: Output size for upsampling
            
        Returns:
            Saliency map as numpy array (H, W) normalized to [0, 1]
        """
        image_tensor = image_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture gradients/activations. Check target_layer.")
        
        # Compute weights via global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Keep only positive influences
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample to image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return cam.cpu().numpy()
    
    def __call__(self, image_tensor: torch.Tensor, target_class: int, 
                 image_size: int = 224) -> np.ndarray:
        """Alias for generate()."""
        return self.generate(image_tensor, target_class, image_size)
