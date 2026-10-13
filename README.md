# EMORA

### Multimodal AI Emotion Recognition Pipeline with MLOps

EMORA is a multimodal artificial intelligence system designed to recognize human emotions by combining **text, audio, and visual information**.

The project currently focuses on **Mark 1**, which establishes the core multimodal emotion-recognition pipeline using the CMU-MOSEI dataset, deep learning, experiment tracking, and an interactive inference interface.

---

## Project Overview

Human emotions are expressed through multiple signals such as words, voice characteristics, and facial expressions. Systems relying on only one modality may miss important emotional cues.

EMORA addresses this by processing three modalities independently and combining their learned representations before emotion classification.

```text
                 CMU-MOSEI
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Text         Audio        Vision
        │            │            │
     GloVe         COVAREP      FACET 4.2
        │            │            │
     BiLSTM        BiLSTM       BiLSTM
        │            │            │
      128-D        128-D        128-D
        └────────────┼────────────┘
                     ▼
              Feature Fusion
                  384 → 256
                     │
                     ▼
            Emotion Classifier
                     │
                     ▼
              6 Emotion Outputs
