#!/usr/bin/env python
"""Run the complete pipeline on a single image."""

import argparse
import yaml
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SemanticSaliencyPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Causal Saliency Attribution Pipeline")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--no-validate", action="store_true", help="Skip counterfactual validation")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SemanticSaliencyPipeline(config)
    
    # Process image
    print(f"Processing: {args.image}")
    result = pipeline.process_image(args.image, validate=not args.no_validate)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Regions Found: {len(result['regions'])}")
    print(f"\nConcept Attributions:")
    print("-" * 40)
    
    for i, concept in enumerate(result['concepts']):
        print(f"  {i+1}. {concept['concept']}: {concept['importance_raw']*100:.1f}% causal impact")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    result_json = {
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence'],
        'concepts': result['concepts']
    }
    with open(output_dir / "result.json", 'w') as f:
        json.dump(result_json, f, indent=2)
    
    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(result['image_np'])
    axes[0].set_title(f"Input\n{result['predicted_class']}: {result['confidence']:.1%}")
    axes[0].axis('off')
    
    axes[1].imshow(result['image_np'])
    axes[1].imshow(result['saliency_map'], cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM Saliency")
    axes[1].axis('off')
    
    if result['concepts']:
        concepts = result['concepts'][:5]
        labels = [c['concept'] for c in concepts]
        values = [c['importance_raw'] * 100 for c in concepts]
        axes[2].barh(range(len(labels)), values, color='steelblue')
        axes[2].set_yticks(range(len(labels)))
        axes[2].set_yticklabels(labels)
        axes[2].set_xlabel('Causal Impact (%)')
        axes[2].set_title("Concept Attributions")
    else:
        axes[2].text(0.5, 0.5, "No concepts", ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
