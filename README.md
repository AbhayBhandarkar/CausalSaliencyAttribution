# Causal Saliency Attribution

[![Paper](https://img.shields.io/badge/Paper-AAAI%20XAI4Science%202025-blue)](https://abhaybhandarkar.github.io/SemanticAttributionofSaliencyMaps)
[![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B)](https://huggingface.co/spaces/AbhayBhandarkar/SemanticSaliency)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Causal Quantification of Saliency Attributions via Semantic Decomposition and Counterfactual Validation**

*Abhay Bhandarkar*  
*AAAI XAI4Science Workshop 2025*

---

## Overview

Traditional saliency methods (Grad-CAM, SHAP, LIME) show **where** models look but fail to explain **what** concepts they recognize or **why** those regions matter. This framework bridges the semantic gap through:

1. **Saliency Generation** — Grad-CAM identifies important pixels
2. **Semantic Region Extraction** — SAM or SLIC decomposes salient regions
3. **Concept Discovery** — CLIP assigns human-interpretable labels
4. **Counterfactual Validation** — Gaussian blur proves causal importance

```
Input Image → ResNet-50 → Grad-CAM → SAM/SLIC → CLIP → Blur → Concept Attributions
                ↓            ↓           ↓         ↓        ↓
           Prediction   Saliency    Regions   Concepts  Causal Impact
```

### Example Output

```
Prediction: German Shepherd (92% confidence)

Concept Attributions:
  1. head: 45.2% causal impact
  2. pointed ears: 28.1% causal impact  
  3. fur texture: 15.3% causal impact
  4. body: 11.4% causal impact
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/AbhayBhandarkar/CausalSaliencyAttribution.git
cd CausalSaliencyAttribution

# Create environment
conda create -n causal_saliency python=3.10
conda activate causal_saliency

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint (optional, for SAM segmentation)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Quick Start

### Streamlit Demo

```bash
streamlit run app.py
```

### Command Line

```bash
python scripts/run_pipeline.py path/to/image.jpg --output results/
```

### Python API

```python
from src.pipeline import SemanticSaliencyPipeline
import yaml

# Load config
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = SemanticSaliencyPipeline(config)

# Process image
result = pipeline.process_image('image.jpg')

# Access results
print(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
for concept in result['concepts']:
    print(f"  {concept['concept']}: {concept['importance_raw']*100:.1f}%")
```

---

## Project Structure

```
CausalSaliencyAttribution/
├── app.py                    # Streamlit demo
├── requirements.txt          # Dependencies
├── config/
│   └── default.yaml          # Configuration
├── data/
│   └── concept_vocabulary.json
├── src/
│   ├── classifier.py         # ResNet-50 wrapper
│   ├── pipeline.py           # End-to-end pipeline
│   ├── saliency/
│   │   └── gradcam.py        # Grad-CAM implementation
│   ├── regions/
│   │   ├── sam_extractor.py  # SAM segmentation
│   │   ├── slic_extractor.py # SLIC superpixels
│   │   └── feature_extractor.py
│   ├── concepts/
│   │   ├── clip_matcher.py   # CLIP matching
│   │   └── vocabulary.py     # Concept vocabulary
│   └── validation/
│       ├── counterfactual.py # Blur validation
│       └── ranker.py         # Concept ranking
└── scripts/
    └── run_pipeline.py       # CLI script
```

---

## Key Results

| Method | Avg Causal Impact | Success Rate |
|--------|-------------------|--------------|
| **SAM (Ours)** | **80.97%** | 90.5% |
| SLIC (Baseline) | 37.72% | 100% |

SAM achieves **2.15× higher causal impact** than SLIC, revealing a sensitivity-reliability trade-off.

---

## Configuration

Edit `config/default.yaml`:

```yaml
model:
  classifier: "resnet50"      # resnet50, resnet101, vit_b_16
  device: "cuda"

regions:
  method: "sam"               # sam or slic
  saliency_threshold: 0.3     # Min saliency for regions
  top_k_regions: 5            # Regions to analyze

concepts:
  clip_model: "ViT-L-14"      # CLIP architecture

counterfactual:
  method: "blur"              # Gaussian blur
```

---

## Citation

```bibtex
@inproceedings{bhandarkar2025causal,
  title={Causal Quantification of Saliency Attributions via Semantic Decomposition and Counterfactual Validation},
  author={Bhandarkar, Abhay},
  booktitle={AAAI Workshop on XAI for Science (XAI4Science)},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - LAION
- [Grad-CAM](https://arxiv.org/abs/1610.02391) - Selvaraju et al.
