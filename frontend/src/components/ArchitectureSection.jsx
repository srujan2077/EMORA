import React, { useState } from 'react';
import { EMOTIONS, MARK3_BENCHMARK } from '../data/modelSpecs';

export default function ArchitectureSection({ defaultOpen = true }) {
  const [activeNode, setActiveNode] = useState('fusion');

  const nodesInfo = {
    text: {
      name: 'DeBERTa-v3-base',
      dim: '768-D',
      desc: 'Disentangled attention mechanism encoding lexical syntax, semantic valence, and sentiment intent from utterance transcripts.',
      specs: ['12 Layers', '768 Hidden Dim', '12 Attention Heads', 'Mask-aware Mean Pooling'],
    },
    audio: {
      name: 'Wav2Vec2-base-960h',
      dim: '768-D',
      desc: 'Self-supervised convolutional + transformer stack extracting pitch contours, speaking rate, energy variance, and acoustic prosody.',
      specs: ['16 kHz Mono Audio', 'CNN Feature Extractor', '12 Transformer Blocks', 'Temporal Mean Pooling'],
    },
    vision: {
      name: 'ViT-base-patch16-224-in21k',
      dim: '768-D',
      desc: 'Vision Transformer trained on ImageNet-21k capturing dynamic facial action units, eyebrow furrowing, and micro-expressions.',
      specs: ['16x16 Patch Size', '224x224 Resolution', 'Uniform 8-Frame Sequence', '[CLS] Token Representation'],
    },
    fusion: {
      name: 'Multimodal Transformer Fusion',
      dim: '256-D Core',
      desc: 'Projects each 768-D modality vector to 256-D, prepends modality tokens ([TEXT], [AUDIO], [VIDEO], [CLS]), and learns cross-modal attention.',
      specs: ['Linear + LayerNorm Projections', '2-Layer Transformer', '8 Multi-Head Attention', 'Relative Position Bias'],
    },
    temporal: {
      name: 'Causal Temporal Ensemble (Mark 3)',
      dim: '5-Seed Ensemble',
      desc: 'Accumulates causal conversational context over up to 8 dialogue segments with controlled split-at-gap suffixing and fixed 0.5 decision thresholds.',
      specs: ['5 Independent Training Seeds', 'Averaged Probabilities', 'Calibrated Sigmoid Cutoffs', 'BCEWithLogitsLoss'],
    },
  };

  const selected = nodesInfo[activeNode] || nodesInfo.fusion;

  return (
    <section
      style={{
        background: '#111111',
        color: '#F4F2EC',
        padding: 'clamp(4rem, 8vh, 7rem) 0',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div className="editorial-container">
        {/* Section Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            borderBottom: '1px solid rgba(244, 242, 236, 0.15)',
            paddingBottom: '1rem',
            marginBottom: 'clamp(2.5rem, 5vh, 4rem)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'rgba(244, 242, 236, 0.5)',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ color: '#F4F2EC', fontWeight: 600 }}>[03/04]</span>
            <span>SYSTEM ARCHITECTURE // MARK 03</span>
          </div>
          <span>NEURAL TRANSFORMER ENSEMBLE PIPELINE</span>
        </div>

        {/* Title */}
        <div style={{ maxWidth: '780px', marginBottom: '3.5rem' }}>
          <h2
            style={{
              fontSize: 'clamp(2rem, 4.5vw, 3.75rem)',
              fontWeight: 700,
              letterSpacing: '-0.03em',
              lineHeight: 1.05,
              margin: '0 0 1.25rem 0',
              textTransform: 'uppercase',
            }}
          >
            Three Disparate Modalities.
            <br />
            One Fused Affective Manifold.
          </h2>
          <p
            style={{
              fontSize: '0.9375rem',
              color: 'rgba(244, 242, 236, 0.7)',
              lineHeight: 1.7,
              margin: 0,
            }}
          >
            Raw linguistic text, continuous vocal prosody, and facial video patches are independently encoded by specialized transformer backbones, linearly mapped into a joint 256-D subspace, and fused through cross-attention.
          </p>
        </div>

        {/* Interactive Pipeline Diagram (SVG) */}
        <div
          style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(244, 242, 236, 0.15)',
            padding: 'clamp(1.5rem, 3vw, 2.5rem)',
            marginBottom: '3rem',
            position: 'relative',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'rgba(244, 242, 236, 0.4)',
              marginBottom: '1.5rem',
            }}
          >
            INTERACTIVE PIPELINE TOPOLOGY — SELECT NODE TO INSPECT TENSOR GEOMETRY
          </div>

          {/* Pipeline Nodes Row */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '1rem',
              marginBottom: '2rem',
            }}
          >
            {[
              { id: 'text', title: '1. TEXT ENCODER', model: 'DeBERTa-v3', dim: '768-D' },
              { id: 'audio', title: '2. AUDIO ENCODER', model: 'Wav2Vec2', dim: '768-D' },
              { id: 'vision', title: '3. VISION ENCODER', model: 'ViT-base', dim: '768-D' },
              { id: 'fusion', title: '4. FUSION CORE', model: '2-Layer Trans.', dim: '256-D' },
              { id: 'temporal', title: '5. SOTA ENSEMBLE', model: '5-Seed Causal', dim: '6 Sigmoids' },
            ].map((node) => {
              const isSelected = activeNode === node.id;
              return (
                <button
                  key={node.id}
                  type="button"
                  onClick={() => setActiveNode(node.id)}
                  style={{
                    background: isSelected ? 'rgba(244, 242, 236, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                    border: isSelected ? '1px solid #F4F2EC' : '1px solid rgba(244, 242, 236, 0.15)',
                    padding: '1.25rem 1rem',
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    position: 'relative',
                  }}
                >
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.5625rem',
                      letterSpacing: '0.08em',
                      color: isSelected ? '#F4F2EC' : 'rgba(244, 242, 236, 0.5)',
                      marginBottom: '0.4rem',
                    }}
                  >
                    {node.title}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.9375rem',
                      fontWeight: 600,
                      color: '#F4F2EC',
                      marginBottom: '0.35rem',
                    }}
                  >
                    {node.model}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.625rem',
                      color: 'rgba(244, 242, 236, 0.6)',
                    }}
                  >
                    {node.dim}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Node Detail Readout */}
          <div
            style={{
              borderTop: '1px solid rgba(244, 242, 236, 0.12)',
              paddingTop: '1.5rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(12, 1fr)',
              gap: '2rem',
              alignItems: 'baseline',
            }}
          >
            <div style={{ gridColumn: 'span 7' }}>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.625rem',
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  color: 'rgba(244, 242, 236, 0.5)',
                  marginBottom: '0.5rem',
                }}
              >
                Subsystem Overview // {selected.name} ({selected.dim})
              </div>
              <p
                style={{
                  fontSize: '0.875rem',
                  lineHeight: 1.7,
                  color: '#F4F2EC',
                  margin: 0,
                }}
              >
                {selected.desc}
              </p>
            </div>

            <div style={{ gridColumn: 'span 5' }}>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.625rem',
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  color: 'rgba(244, 242, 236, 0.5)',
                  marginBottom: '0.75rem',
                }}
              >
                Technical Parameters
              </div>
              <ul
                style={{
                  listStyle: 'none',
                  padding: 0,
                  margin: 0,
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6875rem',
                  color: 'rgba(244, 242, 236, 0.8)',
                  lineHeight: 1.9,
                }}
              >
                {selected.specs.map((spec, i) => (
                  <li key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ color: 'rgba(244, 242, 236, 0.4)' }}>—</span>
                    <span>{spec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Output Emotion Space — Strictly Monochrome Architecture Tags */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '1.5rem',
            borderTop: '1px solid rgba(244, 242, 236, 0.15)',
            paddingTop: '1.75rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            letterSpacing: '0.06em',
          }}
        >
          <span style={{ color: 'rgba(244, 242, 236, 0.5)' }}>
            TARGET EMOTION CHANNELS (INDEPENDENT SIGMOIDS):
          </span>

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            {EMOTIONS.map((emo) => (
              <span
                key={emo}
                style={{
                  padding: '0.2rem 0.6rem',
                  border: '1px solid rgba(244, 242, 236, 0.25)',
                  color: '#F4F2EC',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                }}
              >
                {emo} (τ = {MARK3_BENCHMARK.thresholds[emo] ? (MARK3_BENCHMARK.thresholds[emo] * 100).toFixed(0) : '50'}%)
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
