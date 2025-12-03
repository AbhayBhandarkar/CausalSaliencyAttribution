"""Concept vocabulary management."""

import json
from pathlib import Path
from typing import List


DEFAULT_CONCEPTS = [
    # Objects
    "wheel", "door", "window", "headlight", "mirror", "antenna",
    "handle", "bumper", "license plate", "tire", "rim",
    # Animal parts
    "head", "ear", "eye", "nose", "mouth", "snout", "tail",
    "paw", "leg", "fur", "whiskers", "teeth", "tongue",
    # Textures
    "fur texture", "metal", "glass", "plastic", "leather",
    "wood", "fabric", "smooth surface", "rough texture",
    # Colors
    "red color", "blue color", "green color", "yellow color",
    "black color", "white color", "brown color", "gray color",
    # Patterns
    "stripes", "spots", "checkered pattern", "plain",
    # Human parts
    "face", "hair", "hand", "arm", "shoulder", "neck",
    # Clothing
    "shirt", "pants", "dress", "hat", "shoe", "jacket",
    "tie", "button", "collar", "sleeve",
    # Background
    "sky", "cloud", "grass", "tree", "building", "road",
    "pavement", "wall", "floor", "water",
    # Shapes
    "round shape", "square shape", "rectangular", "circular",
    "triangular", "curved", "straight edge",
]


class ConceptVocabulary:
    """Manage concept vocabulary for CLIP matching."""
    
    def __init__(self, vocabulary: List[str] = None):
        """Initialize vocabulary.
        
        Args:
            vocabulary: List of concepts (uses defaults if None)
        """
        self.concepts = vocabulary if vocabulary is not None else DEFAULT_CONCEPTS.copy()
    
    @classmethod
    def from_file(cls, filepath: str) -> "ConceptVocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            concepts = data
        elif isinstance(data, dict) and 'concepts' in data:
            concepts = data['concepts']
        else:
            raise ValueError("Invalid vocabulary format")
        
        return cls(concepts)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({'concepts': self.concepts, 'count': len(self.concepts)}, f, indent=2)
    
    def __len__(self) -> int:
        return len(self.concepts)
    
    def __iter__(self):
        return iter(self.concepts)
