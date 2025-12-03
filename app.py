"""
Causal Saliency Attribution - Interactive Demo
AAAI XAI4Science 2025

Author: Abhay Bhandarkar
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Causal Saliency Attribution",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .concept-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .sam-box { border-left-color: #e74c3c; }
    .slic-box { border-left-color: #2ecc71; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_classifier():
    from src.classifier import Classifier
    return Classifier(model_name="resnet50", pretrained=True, 
                      device="cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_gradcam(_classifier):
    from src.saliency.gradcam import GradCAM
    target_layer = _classifier.get_target_layer("layer4")
    return GradCAM(_classifier.model, target_layer)


@st.cache_resource(show_spinner=False)
def load_sam_extractor():
    from src.regions.sam_extractor import SAMRegionExtractor
    checkpoint = "sam_vit_h_4b8939.pth"
    if Path(checkpoint).exists():
        return SAMRegionExtractor(checkpoint_path=checkpoint, model_type="vit_h",
                                   device="cuda" if torch.cuda.is_available() else "cpu")
    return None


@st.cache_resource(show_spinner=False)
def load_slic_extractor():
    from src.regions.slic_extractor import SLICRegionExtractor
    return SLICRegionExtractor(n_segments=100, compactness=20)


@st.cache_resource(show_spinner=False)
def load_clip_matcher():
    from src.concepts.clip_matcher import CLIPConceptMatcher
    from src.concepts.vocabulary import ConceptVocabulary
    
    vocab_path = "data/concept_vocabulary.json"
    vocabulary = ConceptVocabulary.from_file(vocab_path) if Path(vocab_path).exists() else ConceptVocabulary()
    
    return CLIPConceptMatcher(
        concept_vocabulary=vocabulary.concepts,
        model_name="ViT-L-14",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    from src.regions.feature_extractor import LocalFeatureExtractor
    return LocalFeatureExtractor(context_margin=0.2)


@st.cache_data(show_spinner=False)
def load_imagenet_labels():
    try:
        import requests
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return {i: label.strip() for i, label in enumerate(response.text.strip().split("\n"))}
    except:
        pass
    return {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def blur_region(image_np, bbox, kernel_size=51, sigma=25):
    """Apply Gaussian blur to a region."""
    image_masked = image_np.copy()
    x_min, y_min, x_max, y_max = bbox
    region = image_masked[y_min:y_max, x_min:x_max]
    
    if region.size == 0:
        return image_masked
    
    k_size = min(kernel_size | 1, min(region.shape[0], region.shape[1]) | 1)
    k_size = max(1, k_size)
    
    blurred = cv2.GaussianBlur(region, (k_size, k_size), sigma)
    image_masked[y_min:y_max, x_min:x_max] = blurred
    return image_masked


def compute_confidence_drop(classifier, blurred_image_np, original_confidence):
    """Compute confidence drop after blurring."""
    blurred_pil = Image.fromarray(blurred_image_np.astype('uint8'))
    blurred_tensor = classifier.preprocess(blurred_pil)
    _, blurred_confidence, _ = classifier.get_prediction(blurred_tensor)
    drop = (original_confidence - blurred_confidence) / (original_confidence + 1e-8)
    return max(0.0, drop), blurred_confidence


def create_saliency_overlay(image_np, saliency_map, alpha=0.6):
    """Create saliency overlay visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    
    s_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    im = ax.imshow(s_norm, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    ax.set_title("Grad-CAM Saliency Map", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def create_regions_overlay(image_np, regions, title, color):
    """Create regions overlay visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    
    for i, region in enumerate(regions):
        x_min, y_min, x_max, y_max = region['bbox']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"R{i+1}", color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def create_concept_bar_chart(concepts, title, color):
    """Create bar chart of concept importance."""
    if not concepts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No concepts found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    labels = [c['concept'] for c in concepts]
    importance = [c['importance_raw'] * 100 for c in concepts]
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(concepts) * 0.6)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, importance, color=color, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Causal Impact (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(importance) * 1.2 if importance else 100)
    
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
    
    plt.tight_layout()
    return fig


def process_regions(image_np, saliency_map, regions, classifier, image_tensor, 
                    original_confidence, feature_extractor, clip_matcher):
    """Process regions to find concepts and compute importance."""
    results = []
    
    for i, region in enumerate(regions):
        bbox = region['bbox']
        crop = feature_extractor.crop_region(image_np, bbox)
        matches = clip_matcher.match_region(crop, top_k=1)
        
        if not matches:
            continue
        
        concept, similarity = matches[0]
        blurred_np = blur_region(image_np, bbox)
        drop, _ = compute_confidence_drop(classifier, blurred_np, original_confidence)
        
        results.append({
            'region_id': i,
            'bbox': bbox,
            'concept': concept,
            'similarity_score': similarity,
            'importance_raw': drop,
            'saliency_magnitude': region.get('mean_saliency', 0.0)
        })
    
    return sorted(results, key=lambda x: x['importance_raw'], reverse=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<p class="main-header">🔍 Causal Saliency Attribution</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AAAI XAI4Science 2025 | Abhay Bhandarkar</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        saliency_threshold = st.slider("Saliency Threshold", 0.1, 0.9, 0.3, 0.05)
        top_k_regions = st.slider("Top-K Regions", 1, 10, 5)
        min_region_size = st.slider("Min Region Size", 50, 500, 100, 50)
        
        st.markdown("---")
        st.markdown("### 📊 Methods")
        show_sam = st.checkbox("SAM Segmentation", value=True)
        show_slic = st.checkbox("SLIC Superpixels", value=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **Pipeline:**
        1. Grad-CAM saliency
        2. SAM/SLIC regions
        3. CLIP concepts
        4. Counterfactual validation
        
        [GitHub](https://github.com/AbhayBhandarkar/CausalSaliencyAttribution)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        st.markdown("##### Or try a sample:")
        sample_col1, sample_col2 = st.columns(2)
        use_sample = None
        with sample_col1:
            if st.button("🐕 Dog", use_container_width=True):
                use_sample = "dog"
        with sample_col2:
            if st.button("🚗 Car", use_container_width=True):
                use_sample = "car"
    
    # Load image
    image_pil = None
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert('RGB')
    elif use_sample:
        try:
            import requests
            urls = {
                "dog": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02088364_beagle.JPEG",
                "car": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02814533_beach_wagon.JPEG"
            }
            response = requests.get(urls[use_sample], timeout=10)
            image_pil = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            st.error(f"Could not load sample: {e}")
    
    if image_pil:
        with col1:
            st.image(image_pil, caption="Input Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔄 Loading models..."):
                classifier = load_classifier()
                gradcam = load_gradcam(classifier)
                sam_extractor = load_sam_extractor() if show_sam else None
                slic_extractor = load_slic_extractor() if show_slic else None
                clip_matcher = load_clip_matcher()
                feature_extractor = load_feature_extractor()
                imagenet_labels = load_imagenet_labels()
            
            st.markdown("### 🔬 Analysis Results")
            
            # Process
            image_pil_resized = image_pil.resize((224, 224))
            image_np = np.array(image_pil_resized)
            image_tensor = classifier.preprocess(image_pil_resized)
            
            predicted_class, confidence, _ = classifier.get_prediction(image_tensor)
            class_name = imagenet_labels.get(predicted_class, f"Class {predicted_class}")
            
            # Display prediction
            st.markdown("#### 🎯 Classification")
            m1, m2, m3 = st.columns(3)
            m1.metric("Class", class_name)
            m2.metric("Confidence", f"{confidence:.1%}")
            m3.metric("ID", predicted_class)
            
            # Saliency
            with st.spinner("Generating Grad-CAM..."):
                saliency_map = gradcam.generate(image_tensor, predicted_class, image_size=224)
            
            st.markdown("#### 🔥 Saliency Map")
            fig = create_saliency_overlay(image_np, saliency_map)
            st.pyplot(fig)
            plt.close(fig)
            
            results = {}
            
            # SAM
            if show_sam and sam_extractor:
                st.markdown("#### 🔴 SAM Segmentation")
                with st.spinner("Running SAM..."):
                    sam_regions = sam_extractor.extract_regions(
                        image_np, saliency_map, saliency_threshold, top_k_regions, min_region_size
                    )
                
                if sam_regions:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = create_regions_overlay(image_np, sam_regions, f"SAM ({len(sam_regions)} regions)", '#e74c3c')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with st.spinner("Computing SAM concepts..."):
                        sam_concepts = process_regions(image_np, saliency_map, sam_regions, classifier,
                                                        image_tensor, confidence, feature_extractor, clip_matcher)
                    
                    with c2:
                        fig = create_concept_bar_chart(sam_concepts, "SAM Concepts", '#e74c3c')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    results['sam'] = {'regions': sam_regions, 'concepts': sam_concepts}
            
            # SLIC
            if show_slic:
                st.markdown("#### 🟢 SLIC Superpixels")
                with st.spinner("Running SLIC..."):
                    slic_regions = slic_extractor.extract_regions(
                        image_np, saliency_map, saliency_threshold, top_k_regions, min_region_size
                    )
                
                if slic_regions:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = create_regions_overlay(image_np, slic_regions, f"SLIC ({len(slic_regions)} regions)", '#2ecc71')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with st.spinner("Computing SLIC concepts..."):
                        slic_concepts = process_regions(image_np, saliency_map, slic_regions, classifier,
                                                         image_tensor, confidence, feature_extractor, clip_matcher)
                    
                    with c2:
                        fig = create_concept_bar_chart(slic_concepts, "SLIC Concepts", '#2ecc71')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    results['slic'] = {'regions': slic_regions, 'concepts': slic_concepts}
            
            # Comparison
            if 'sam' in results and 'slic' in results:
                st.markdown("---")
                st.markdown("#### 📊 Comparison")
                
                sam_impact = sum(c['importance_raw'] for c in results['sam']['concepts']) * 100
                slic_impact = sum(c['importance_raw'] for c in results['slic']['concepts']) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("SAM Impact", f"{sam_impact:.1f}%")
                c2.metric("SLIC Impact", f"{slic_impact:.1f}%")
                c3.metric("Difference", f"{sam_impact - slic_impact:+.1f}%")
    
    else:
        with col2:
            st.markdown("""
            ### 👈 Upload an image to begin
            
            **Pipeline:**
            ```
            Image → ResNet-50 → Grad-CAM → SAM/SLIC → CLIP → Blur Validation
            ```
            
            **What makes this different?**
            
            Traditional XAI shows *where* the model looks.
            We show *what* concepts matter and *prove* it causally.
            """)


if __name__ == "__main__":
    main()
