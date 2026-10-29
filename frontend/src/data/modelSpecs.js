export const EMOTIONS = [
  'Happy',
  'Sad',
  'Anger',
  'Surprise',
  'Disgust',
  'Fear',
];

/* ============================================================
   EMOTION PALETTE — Mark 3 Editorial Design Tokens
   Single source of truth for all emotion colours.
   These are design tokens, not model parameters.
   ============================================================ */
export const EMOTION_PALETTE = {
  Happy:    { hex: '#E8B931', rgb: '232,185,49'  },
  Sad:      { hex: '#548CE8', rgb: '84,140,232'  },
  Anger:    { hex: '#E45B52', rgb: '228,91,82'   },
  Surprise: { hex: '#B39AF5', rgb: '179,154,245' },
  Disgust:  { hex: '#82A96C', rgb: '130,169,108' },
  Fear:     { hex: '#8176E8', rgb: '129,118,232' },
};

export const EMOTION_META = {
  Happy: {
    color: EMOTION_PALETTE.Happy.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Happy.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Happy.rgb}, 0.25)`,
    description: 'Positive affect, joyful or satisfied expressions',
  },
  Sad: {
    color: EMOTION_PALETTE.Sad.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Sad.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Sad.rgb}, 0.25)`,
    description: 'Sorrow, grief, or melancholic tone',
  },
  Anger: {
    color: EMOTION_PALETTE.Anger.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Anger.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Anger.rgb}, 0.25)`,
    description: 'Frustration, irritation, or hostile acoustic cues',
  },
  Surprise: {
    color: EMOTION_PALETTE.Surprise.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Surprise.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Surprise.rgb}, 0.25)`,
    description: 'Unexpected astonishment or startled vocal inflection',
  },
  Disgust: {
    color: EMOTION_PALETTE.Disgust.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Disgust.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Disgust.rgb}, 0.25)`,
    description: 'Revulsion, distaste, or aversion reactions',
  },
  Fear: {
    color: EMOTION_PALETTE.Fear.hex,
    lightColor: `rgba(${EMOTION_PALETTE.Fear.rgb}, 0.12)`,
    borderColor: `rgba(${EMOTION_PALETTE.Fear.rgb}, 0.25)`,
    description: 'Apprehension, distress, or anxious prosody',
  },
};

export const MARK2_BENCHMARK = {
  microF1: 0.5316,
  macroF1: 0.4693,
  subsetAccuracy: 0.1900,
  perEmotionF1: {
    Happy: 0.7592,
    Sad: 0.4545,
    Anger: 0.4850,
    Surprise: 0.3056,
    Disgust: 0.5594,
    Fear: 0.2523,
  },
};

export const MARK3_BENCHMARK = {
  microF1: 0.5220,
  macroF1: 0.4629,
  subsetAccuracy: 0.2179,
  perEmotionF1: {
    Happy: 0.7233,
    Sad: 0.4627,
    Anger: 0.5244,
    Surprise: 0.2965,
    Disgust: 0.5051,
    Fear: 0.2655,
  },
  ensembleSize: 5,
  seeds: [42, 123, 456, 789, 2026],
  candidate: 'split_at_gap',
  thresholds: {
    Happy: 0.5,
    Sad: 0.5,
    Anger: 0.5,
    Surprise: 0.5,
    Disgust: 0.5,
    Fear: 0.5,
  },
};

export const MARK2_5_SPECS = {
  checkpoint: 'checkpoints/mark2_5_fast_best.pt',
  bestEpoch: 3,
  valMicroF1: 0.5280966767,
  methodology: 'Selective fine-tuning with frozen backbone representations',
};

export const ARCHITECTURE_DETAILS = {
  mark1: {
    title: 'Mark 1 (Active Runtime Baseline)',
    textEncoder: 'GloVe Word Vectors (300-d) → BiLSTM → 128-d',
    audioEncoder: 'COVAREP Features (74-d, stride 10) → BiLSTM → 128-d',
    visionEncoder: 'FACET 4.2 Features (35-d, stride 5) → BiLSTM → 128-d',
    fusion: 'Modality Concatenation (384-d) → MLP (256-d)',
    output: '6 emotion logits with BCEWithLogitsLoss',
    task: 'Multi-label emotion classification',
    checkpoint: 'checkpoints/mark1_best.pt (Epoch 2, Val Loss: 0.4901)',
  },
  mark2: {
    title: 'Mark 2 & Mark 2.5 Architecture',
    textEncoder: 'DeBERTa-v3-base',
    audioEncoder: 'Wav2Vec2-base-960h',
    visionEncoder: 'ViT-base-patch16-224-in21k',
    projections: 'Each modality linearly projected to 256 dimensions',
    fusion: '2-Layer Multimodal Transformer with [CLS], [TEXT], [AUDIO], [VIDEO] tokens',
    output: '256-D fused representation → Classifier → 6 emotion logits',
    loss: 'BCEWithLogitsLoss (Multi-label)',
    training: 'Selective fine-tuning (Mark 2.5 checkpoint: checkpoints/mark2_5_fast_best.pt)',
  },
  mark3: {
    title: 'Mark 3 (Temporal Multimodal Fusion Ensemble)',
    textEncoder: 'DeBERTa-v3-base (768-d, mask-aware mean pooling)',
    audioEncoder: 'Wav2Vec2-base-960h (768-d, FFmpeg 16kHz mono, mean pooling)',
    visionEncoder: 'ViT-base-patch16-224-in21k (768-d, uniform 8-frame CLS mean)',
    temporal: 'Causal window of up to 8 segments with controlled split_at_gap suffix',
    fusion: 'Modality gating + Relative temporal position embeddings + Gap projections + 2-Layer Transformer',
    ensemble: '5-Seed Ensemble (Seeds 42, 123, 456, 789, 2026) with fixed 0.5 thresholds',
    output: 'Ensemble averaged probabilities with independent sigmoid decisions at 0.5',
  },
};
