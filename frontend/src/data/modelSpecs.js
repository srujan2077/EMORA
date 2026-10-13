export const EMOTIONS = [
  'Happy',
  'Sad',
  'Anger',
  'Surprise',
  'Disgust',
  'Fear',
];

export const EMOTION_META = {
  Happy: {
    color: '#10b981',
    lightColor: 'rgba(16, 185, 129, 0.1)',
    borderColor: 'rgba(16, 185, 129, 0.25)',
    description: 'Positive affect, joyful or satisfied expressions',
  },
  Sad: {
    color: '#60a5fa',
    lightColor: 'rgba(96, 165, 250, 0.1)',
    borderColor: 'rgba(96, 165, 250, 0.25)',
    description: 'Sorrow, grief, or melancholic tone',
  },
  Anger: {
    color: '#f87171',
    lightColor: 'rgba(248, 113, 113, 0.1)',
    borderColor: 'rgba(248, 113, 113, 0.25)',
    description: 'Frustration, irritation, or hostile acoustic cues',
  },
  Surprise: {
    color: '#c084fc',
    lightColor: 'rgba(192, 132, 252, 0.1)',
    borderColor: 'rgba(192, 132, 252, 0.25)',
    description: 'Unexpected astonishment or startled vocal inflection',
  },
  Disgust: {
    color: '#fbbf24',
    lightColor: 'rgba(251, 191, 36, 0.1)',
    borderColor: 'rgba(251, 191, 36, 0.25)',
    description: 'Revulsion, distaste, or aversion reactions',
  },
  Fear: {
    color: '#818cf8',
    lightColor: 'rgba(129, 140, 248, 0.1)',
    borderColor: 'rgba(129, 140, 248, 0.25)',
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
};
