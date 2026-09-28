import html
import random
import numpy as np
import torch
import streamlit as st
from mmsdk import mmdatasdk

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.mosei_dataset import MOSEIDataset
from src.data.split import create_video_splits


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

BASE_PATH = "C:/SRUJAN/multimodal_ai/data/processed/aligned_mark1"

DATA_FILES = {
    "glove_vectors": f"{BASE_PATH}/glove_vectors_aligned.csd",
    "COVAREP": f"{BASE_PATH}/COVAREP_aligned.csd",
    "FACET 4.2": f"{BASE_PATH}/FACET42_aligned.csd",
    "All Labels": f"{BASE_PATH}/labels_aligned.csd",
}

CHECKPOINT_PATH = "C:/SRUJAN/multimodal_ai/checkpoints/mark1_best.pt"

EMOTIONS = [
    "Happy",
    "Sad",
    "Anger",
    "Surprise",
    "Disgust",
    "Fear"
]

EMOTION_ICONS = {
    "Happy": "😊",
    "Sad": "😢",
    "Anger": "😠",
    "Surprise": "😲",
    "Disgust": "🤢",
    "Fear": "😨"
}

EMOTION_COLORS = {
    "Happy": "#10b981",    # Emerald
    "Sad": "#3b82f6",      # Blue
    "Anger": "#ef4444",    # Red
    "Surprise": "#8b5cf6", # Purple
    "Disgust": "#f59e0b",  # Amber
    "Fear": "#6366f1"      # Indigo
}

# Recorded Mark 1 Test Benchmark Metrics
BENCHMARK_METRICS = {
    "macro_f1": 0.2822,
    "micro_f1": 0.4603,
    "macro_precision": 0.4522,
    "macro_recall": 0.2679,
    "best_epoch": 2,
    "best_val_loss": 0.4901,
}

PER_EMOTION_RESULTS = [
    {"Emotion": "Happy", "Precision": 0.69, "Recall": 0.71, "F1-Score": 0.70, "Support": 773},
    {"Emotion": "Sad", "Precision": 0.37, "Recall": 0.47, "F1-Score": 0.42, "Support": 407},
    {"Emotion": "Anger", "Precision": 0.67, "Recall": 0.11, "F1-Score": 0.19, "Support": 417},
    {"Emotion": "Surprise", "Precision": 0.00, "Recall": 0.00, "F1-Score": 0.00, "Support": 177},
    {"Emotion": "Disgust", "Precision": 0.47, "Recall": 0.32, "F1-Score": 0.38, "Support": 300},
    {"Emotion": "Fear", "Precision": 0.50, "Recall": 0.01, "F1-Score": 0.01, "Support": 171},
]

MLFLOW_INFO = {
    "tracking_uri": "sqlite:///C:/SRUJAN/multimodal_ai/mlflow.db",
    "experiment_name": "Mark1-Multimodal-Emotion",
    "run_name": "Mark1_Baseline",
    "run_id": "27b85ba426b648538a68a2852d59ae7c",
}

SEED = 42


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Mark 1 — Multimodal Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Base container styling */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1350px;
}

/* Header badge styling */
.badge-mark1 {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 9999px;
    margin-bottom: 0.5rem;
}

.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
    color: var(--text-color);
}

.main-subtitle {
    font-size: 1.05rem;
    color: #8892b0;
    margin-bottom: 1.2rem;
}

/* Custom cards */
.prediction-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    height: 100%;
}

.ground-truth-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    height: 100%;
}

.modality-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.modality-box h4 {
    margin-top: 0;
    margin-bottom: 0.35rem;
    font-size: 0.95rem;
    font-weight: 600;
}

.modality-stat {
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0.2rem 0;
}

.modality-desc {
    font-size: 0.78rem;
    color: #94a3b8;
}

/* Ground truth tag */
.gt-tag {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    border-radius: 6px;
    font-size: 0.88rem;
    font-weight: 600;
}

.gt-tag-empty {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    background-color: rgba(148, 163, 184, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #94a3b8;
    border-radius: 6px;
    font-size: 0.88rem;
}

/* Probability item */
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
}

.prob-badge-gt {
    font-size: 0.68rem;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-weight: 600;
    margin-left: 0.4rem;
}

.prob-badge-top {
    font-size: 0.68rem;
    background: rgba(59, 130, 246, 0.25);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.4);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-weight: 600;
    margin-left: 0.4rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# DEVICE & CACHED RESOURCE LOADERS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_dataset():
    """Load preprocessed CMU-MOSEI dataset computational sequences."""
    try:
        dataset = mmdatasdk.mmdataset(DATA_FILES, BASE_PATH)
        return dataset
    except Exception as exc:
        st.error(f"Error loading CMU-MOSEI dataset: {exc}")
        raise exc


@st.cache_resource(show_spinner=False)
def load_model():
    """Load Mark 1 Multimodal Emotion Model weights from checkpoint."""
    try:
        model = MultimodalEmotionModel()
        model = model.to(device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint
    except Exception as exc:
        st.error(f"Error loading model checkpoint: {exc}")
        raise exc


@st.cache_resource(show_spinner=False)
def prepare_test_dataset(_dataset):
    """
    Reconstruct the exact Mark 1 video-level test split.
    The leading underscore in _dataset prevents hashing the unhashable mmdataset object.
    """
    try:
        sequences = _dataset.computational_sequences

        text_ids = set(sequences["glove_vectors"].data.keys())
        audio_ids = set(sequences["COVAREP"].data.keys())
        vision_ids = set(sequences["FACET 4.2"].data.keys())
        label_ids = set(sequences["All Labels"].data.keys())

        common_ids = sorted(text_ids & audio_ids & vision_ids & label_ids)

        _, _, test_ids = create_video_splits(
            common_ids,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=SEED
        )

        test_dataset = MOSEIDataset(_dataset, test_ids)
        return test_dataset, test_ids
    except Exception as exc:
        st.error(f"Error preparing test dataset split: {exc}")
        raise exc


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

with st.spinner("Initializing Mark 1 Multimodal AI pipeline & CMU-MOSEI dataset..."):
    dataset = load_dataset()
    model, checkpoint = load_model()
    test_dataset, test_segment_ids = prepare_test_dataset(dataset)


# Initialize session state for sample selection and analysis trigger
if "selected_sample_idx" not in st.session_state:
    st.session_state.selected_sample_idx = 0

if "has_run_analysis" not in st.session_state:
    st.session_state.has_run_analysis = True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<span class="badge-mark1">Architecture</span>', unsafe_allow_html=True)
    st.markdown("### Mark 1 Specifications")

    st.markdown(
        """
        **Model Architecture:**
        - **Text:** 300-d GloVe $\\rightarrow$ BiLSTM $\\rightarrow$ 128-d
        - **Audio:** 74-d COVAREP $\\rightarrow$ BiLSTM $\\rightarrow$ 128-d
        - **Vision:** 35-d FACET 4.2 $\\rightarrow$ BiLSTM $\\rightarrow$ 128-d
        - **Fusion:** Concat (384-d) $\\rightarrow$ MLP (256-d) $\\rightarrow$ 6 outputs
        """
    )

    st.markdown("---")
    st.markdown("### Runtime Information")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Device", str(device).upper())
        st.metric("Emotions", "6 classes")
    with col_sb2:
        st.metric("Best Epoch", int(checkpoint.get("epoch", 2)))
        st.metric("Test Samples", f"{len(test_dataset):,}")

    st.markdown("---")
    st.markdown("### MLflow Tracking")
    st.caption("Mark 1 experiment tracking logged via MLflow:")
    st.code(
        f"Experiment: {MLFLOW_INFO['experiment_name']}\n"
        f"Run Name:   {MLFLOW_INFO['run_name']}\n"
        f"Backend:    SQLite (mlflow.db)\n"
        f"Run ID:     {MLFLOW_INFO['run_id'][:12]}...",
        language="yaml"
    )

    st.markdown("---")
    st.caption(
        "Multimodal AI Pipeline • Mark 1 Baseline\n"
        "Trained on CMU-MOSEI Multimodal Sentiment & Emotion Dataset."
    )


# ============================================================
# 1. HEADER SECTION
# ============================================================

header_col, badge_col = st.columns([4, 1])

with header_col:
    st.markdown('<span class="badge-mark1">Mark 1</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Multimodal Emotion AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Text + Audio + Vision → Emotion Recognition</div>',
        unsafe_allow_html=True
    )

with badge_col:
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 0.5rem;">
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.82rem; font-weight: 600;">
                ● Active Checkpoint (Epoch {checkpoint.get('epoch', 2)})
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# ============================================================
# 6. SAMPLE SELECTOR SECTION
# ============================================================

st.markdown("### 🔬 Test Sample Selection")
st.caption(
    "Select an unseen test-set segment from CMU-MOSEI. "
    "You can search by Segment ID or pick a random sample."
)

selector_col1, selector_col2, selector_col3 = st.columns([5, 2, 2])

# Build selectbox options formatted with sample index and segment ID
segment_options = [
    f"[{i:04d}] {seg_id}"
    for i, seg_id in enumerate(test_segment_ids)
]

def on_select_change():
    chosen_str = st.session_state.sample_selector_box
    idx = int(chosen_str.split("]")[0].replace("[", ""))
    st.session_state.selected_sample_idx = idx
    st.session_state.has_run_analysis = True

def pick_random_sample():
    rng = random.Random()
    st.session_state.selected_sample_idx = rng.randint(0, len(test_dataset) - 1)
    st.session_state.has_run_analysis = True

with selector_col1:
    selected_option = st.selectbox(
        "Search Test Segment ID",
        options=segment_options,
        index=st.session_state.selected_sample_idx,
        key="sample_selector_box",
        on_change=on_select_change,
        label_visibility="collapsed"
    )

with selector_col2:
    if st.button("🎲 Random Sample", use_container_width=True):
        pick_random_sample()
        st.rerun()

with selector_col3:
    current_idx = st.session_state.selected_sample_idx
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("◀ Prev", use_container_width=True, disabled=(current_idx <= 0)):
            st.session_state.selected_sample_idx = max(0, current_idx - 1)
            st.session_state.has_run_analysis = True
            st.rerun()
    with nav_col2:
        if st.button("Next ▶", use_container_width=True, disabled=(current_idx >= len(test_dataset) - 1)):
            st.session_state.selected_sample_idx = min(len(test_dataset) - 1, current_idx + 1)
            st.session_state.has_run_analysis = True
            st.rerun()

# Fetch active sample
active_idx = st.session_state.selected_sample_idx
sample = test_dataset[active_idx]
active_segment_id = sample["segment_id"]


# ============================================================
# 5. MULTIMODAL INPUT SECTION
# ============================================================

st.markdown("#### 📦 Multimodal Inputs")
st.caption(
    "ℹ️ *This Mark 1 UI currently operates on preprocessed CMU-MOSEI features "
    "(GloVe word vectors, COVAREP acoustic features, and FACET 4.2 facial features), not raw user-uploaded modalities.*"
)

mod_col1, mod_col2, mod_col3 = st.columns(3)

with mod_col1:
    st.markdown(
        f"""
        <div class="modality-box">
            <h4>💬 TEXT</h4>
            <div class="modality-stat">{sample['text'].shape[0]} <span style="font-size:0.9rem; font-weight:normal; color:#94a3b8;">frames</span></div>
            <div class="modality-desc">GloVe Word Vectors (300 dimensions)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Tensor shape: [{sample['text'].shape[0]}, 300]</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_col2:
    st.markdown(
        f"""
        <div class="modality-box">
            <h4>🎙️ AUDIO</h4>
            <div class="modality-stat">{sample['audio'].shape[0]} <span style="font-size:0.9rem; font-weight:normal; color:#94a3b8;">frames</span></div>
            <div class="modality-desc">COVAREP Features (74 dimensions, stride 10)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Tensor shape: [{sample['audio'].shape[0]}, 74]</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_col3:
    st.markdown(
        f"""
        <div class="modality-box">
            <h4>👁️ VISION</h4>
            <div class="modality-stat">{sample['vision'].shape[0]} <span style="font-size:0.9rem; font-weight:normal; color:#94a3b8;">frames</span></div>
            <div class="modality-desc">FACET 4.2 Features (35 dimensions, stride 5)</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Tensor shape: [{sample['vision'].shape[0]}, 35]</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# Action button
col_btn_space1, col_btn, col_btn_space2 = st.columns([1, 2, 1])
with col_btn:
    analyze_clicked = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    if analyze_clicked:
        st.session_state.has_run_analysis = True


# ============================================================
# MULTIMODAL INFERENCE & DISPLAY
# ============================================================

if st.session_state.has_run_analysis:
    with st.spinner("Running multimodal inference..."):
        # Prepare model input tensors
        text_tensor = sample["text"].unsqueeze(0).to(device)
        audio_tensor = sample["audio"].unsqueeze(0).to(device)
        vision_tensor = sample["vision"].unsqueeze(0).to(device)
        ground_truth_labels = sample["labels"].cpu().numpy().astype(int)

        # Run model inference
        with torch.no_grad():
            logits = model(text_tensor, audio_tensor, vision_tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        # Prediction computations
        best_idx = int(np.argmax(probabilities))
        predicted_emotion = EMOTIONS[best_idx]
        predicted_score = float(probabilities[best_idx])
        predicted_icon = EMOTION_ICONS.get(predicted_emotion, "🎭")
        predicted_color = EMOTION_COLORS.get(predicted_emotion, "#3b82f6")

        # Ground truth emotions
        actual_emotions = [
            emotion for emotion, label in zip(EMOTIONS, ground_truth_labels) if label == 1
        ]
        is_top_match = bool(ground_truth_labels[best_idx] == 1)

    # ========================================================
    # 2 & 4. MAIN PREDICTION & GROUND TRUTH
    # ========================================================

    st.markdown("### 🎯 Prediction & Ground Truth")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                    <span style="font-size:0.8rem; font-weight:700; color:#93c5fd; letter-spacing:0.06em; text-transform:uppercase;">
                        MODEL PREDICTION
                    </span>
                    <span style="background:rgba(59,130,246,0.2); color:#93c5fd; padding:0.2rem 0.55rem; border-radius:9999px; font-size:0.75rem; font-weight:600;">
                        Mark 1 Multimodal
                    </span>
                </div>
                <div style="display:flex; align-items:center; gap:0.9rem; margin: 0.8rem 0;">
                    <span style="font-size:2.8rem; line-height:1;">{predicted_icon}</span>
                    <div>
                        <div style="font-size:1.85rem; font-weight:800; color:{predicted_color}; letter-spacing:-0.01em;">
                            {predicted_emotion.upper()}
                        </div>
                        <div style="font-size:0.88rem; color:#94a3b8;">
                            Strongest predicted emotion
                        </div>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.92rem; color:#cbd5e1;">Model Score:</span>
                    <span style="font-size:1.35rem; font-weight:700; color:#ffffff;">{predicted_score * 100:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        # Build HTML tags for ground truth emotions
        if actual_emotions:
            gt_tags_html = "".join([
                f'<span class="gt-tag">{EMOTION_ICONS.get(e, "")} {e}</span>'
                for e in actual_emotions
            ])
        else:
            gt_tags_html = '<span class="gt-tag-empty">No positive emotion label annotated</span>'

        match_badge_html = (
            '<div style="margin-top:0.9rem; padding:0.6rem 0.8rem; background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:8px; color:#34d399; font-size:0.85rem; font-weight:600;">'
            f'✓ Top prediction ({predicted_emotion}) matches an annotated ground-truth emotion'
            '</div>'
            if is_top_match else
            '<div style="margin-top:0.9rem; padding:0.6rem 0.8rem; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); border-radius:8px; color:#fbbf24; font-size:0.85rem; font-weight:500;">'
            f'Top prediction ({predicted_emotion}) does not match an annotated ground-truth emotion'
            '</div>'
        )

        st.markdown(
            f"""
            <div class="ground-truth-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.6rem;">
                    <span style="font-size:0.8rem; font-weight:700; color:#34d399; letter-spacing:0.06em; text-transform:uppercase;">
                        GROUND TRUTH
                    </span>
                    <span style="background:rgba(16,185,129,0.15); color:#34d399; padding:0.2rem 0.55rem; border-radius:9999px; font-size:0.75rem; font-weight:600;">
                        CMU-MOSEI Label
                    </span>
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; margin-bottom:0.7rem;">
                    Actual CMU-MOSEI emotion label(s) for <code>{html.escape(active_segment_id)}</code>:
                </div>
                <div style="min-height: 2.4rem; display: flex; flex-wrap: wrap; align-items: center;">
                    {gt_tags_html}
                </div>
                {match_badge_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # 3. EMOTION PROBABILITY VISUALIZATION
    # ========================================================

    st.markdown("### 📊 Emotion Probability Visualization")
    st.caption("Multimodal sigmoid output probabilities across all six emotions:")

    prob_col1, prob_col2 = st.columns(2)

    for i, emotion in enumerate(EMOTIONS):
        score = float(probabilities[i])
        is_gt = bool(ground_truth_labels[i] == 1)
        is_top = (i == best_idx)
        icon = EMOTION_ICONS[emotion]
        color = EMOTION_COLORS[emotion]

        target_col = prob_col1 if i < 3 else prob_col2

        with target_col:
            badges = ""
            if is_top:
                badges += '<span class="prob-badge-top">TOP PREDICTION</span>'
            if is_gt:
                badges += '<span class="prob-badge-gt">GROUND TRUTH</span>'

            st.markdown(
                f"""
                <div class="prob-label-row">
                    <div>
                        <span style="font-weight:600; font-size:0.95rem;">{icon} {emotion}</span>
                        {badges}
                    </div>
                    <div style="font-weight:700; font-size:0.92rem; color:{color if is_top else '#cbd5e1'};">
                        {score * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(score)
            st.write("")


# ============================================================
# 7, 8, 9. MODEL INFO, EVALUATION & MLFLOW
# ============================================================

st.markdown("---")
st.markdown("### 📑 Model Details & Performance")

tab_model, tab_eval, tab_mlflow = st.tabs([
    "🧠 Model Information",
    "📈 Mark 1 Evaluation",
    "🔬 MLflow Experiment Tracking"
])

with tab_model:
    st.markdown("#### Model Architecture & Specifications")
    st.caption("Mark 1 multimodal emotion baseline configuration:")

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown(
            """
            - **Architecture:** BiLSTM Encoders + Multimodal Fusion MLP
            - **Modalities:** Text + Audio + Vision
            - **Dataset:** CMU-MOSEI
            - **Emotions:** 6 (Happy, Sad, Anger, Surprise, Disgust, Fear)
            - **Best epoch:** 2
            """
        )

    with spec_col2:
        st.markdown(
            """
            - **Text Encoder:** 300-d GloVe &rarr; BiLSTM &rarr; 128-d embedding
            - **Audio Encoder:** 74-d COVAREP (stride 10) &rarr; BiLSTM &rarr; 128-d embedding
            - **Vision Encoder:** 35-d FACET 4.2 (stride 5) &rarr; BiLSTM &rarr; 128-d embedding
            - **Fusion MLP:** Concat ($128 \\times 3 = 384$) &rarr; Linear(256) + ReLU + Dropout(0.2) &rarr; Linear(6)
            - **Loss Function:** Multi-label BCEWithLogitsLoss
            """
        )

with tab_eval:
    st.markdown("#### Compact Mark 1 Performance")
    st.caption(
        "Evaluated on 1,498 test segments using multi-label decision threshold $\\tau = 0.5$."
    )

    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
    with ev_col1:
        st.metric("Macro F1", f"{BENCHMARK_METRICS['macro_f1']:.4f}")
    with ev_col2:
        st.metric("Micro F1", f"{BENCHMARK_METRICS['micro_f1']:.4f}")
    with ev_col3:
        st.metric("Macro Precision", f"{BENCHMARK_METRICS['macro_precision']:.4f}")
    with ev_col4:
        st.metric("Macro Recall", f"{BENCHMARK_METRICS['macro_recall']:.4f}")

    st.markdown("##### Per-Emotion Classification Report")
    st.dataframe(
        PER_EMOTION_RESULTS,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "💡 **Baseline Performance Assessment:** "
        "The model demonstrates strong performance on dominant classes like **Happy** (Precision: 0.69, Recall: 0.71, F1: 0.70) "
        "and moderate performance on **Sad** (F1: 0.42) and **Disgust** (F1: 0.38). "
        "Underrepresented emotions in CMU-MOSEI (such as Surprise and Fear) suffer from high class imbalance and exhibit low recall in this baseline."
    )

with tab_mlflow:
    st.markdown("#### MLflow Experiment Tracking")
    st.caption("Experiment tracking and metrics logging are recorded in the local MLflow SQLite database:")

    st.markdown(
        f"""
        - **Tracking URI:** `{MLFLOW_INFO['tracking_uri']}`
        - **Experiment:** `{MLFLOW_INFO['experiment_name']}`
        - **Logged Run Name:** `{MLFLOW_INFO['run_name']}`
        - **Run ID:** `{MLFLOW_INFO['run_id']}`
        - **Best Validation Loss:** `{BENCHMARK_METRICS['best_val_loss']}`
        """
    )
    st.code(
        f"# Load run via MLflow Python API:\n"
        f"import mlflow\n"
        f"mlflow.set_tracking_uri('{MLFLOW_INFO['tracking_uri']}')\n"
        f"run = mlflow.get_run('{MLFLOW_INFO['run_id']}')\n"
        f"print(run.data.metrics)",
        language="python"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.84rem; padding: 0.5rem 0;">
        Multimodal Emotion AI • Mark 1 • CMU-MOSEI | Text + Audio + Vision → Emotion Recognition
    </div>
    """,
    unsafe_allow_html=True
)