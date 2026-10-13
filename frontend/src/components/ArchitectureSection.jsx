import React, { useState } from 'react';
import { EMOTIONS, EMOTION_META, ARCHITECTURE_DETAILS } from '../data/modelSpecs';

export default function ArchitectureSection({ defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const specs = [
    { label: 'Text', value: 'DeBERTa-v3-base', detail: '768-D contextual embeddings' },
    { label: 'Audio', value: 'Wav2Vec2-base-960h', detail: '768-D acoustic representations' },
    { label: 'Vision', value: 'ViT-base-patch16-224-in21k', detail: '768-D visual patch embeddings' },
    { label: 'Projection', value: '256-D per modality', detail: 'Linear + LayerNorm' },
    { label: 'Fusion', value: '2-Layer Transformer', detail: '[CLS], [TEXT], [AUDIO], [VIDEO] tokens' },
    { label: 'Output', value: '6-label sigmoid classifier', detail: 'BCEWithLogitsLoss' },
    { label: 'Dataset', value: 'CMU-MOSEI', detail: '22,856 utterances' },
  ];

  return (
    <div className="editorial-container">
      {/* Collapsible trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%',
          padding: '1rem 0',
          borderBottom: '1px solid var(--border-faint)',
        }}
      >
        <span
          style={{
            fontSize: 'clamp(1rem, 1.8vw, 1.25rem)',
            fontWeight: 500,
            color: 'var(--text-primary)',
            letterSpacing: '-0.01em',
          }}
        >
          Model Architecture
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            transition: 'color 0.2s ease',
          }}
        >
          {isOpen ? 'Hide' : 'Show'}
        </span>
      </button>

      {/* Expandable content */}
      {isOpen && (
        <div
          className="animate-fade-up"
          style={{
            paddingTop: '2.5rem',
            paddingBottom: '1rem',
          }}
        >
          {/* Architecture specs — clean list */}
          <div style={{ maxWidth: '560px' }}>
            {specs.map((spec, i) => (
              <div
                key={spec.label}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  padding: '0.875rem 0',
                  borderBottom: i < specs.length - 1 ? '1px solid var(--border-faint)' : 'none',
                }}
              >
                <span
                  style={{
                    fontSize: '0.8125rem',
                    color: 'var(--text-muted)',
                    minWidth: '80px',
                  }}
                >
                  {spec.label}
                </span>
                <div style={{ textAlign: 'right' }}>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8125rem',
                      color: 'var(--text-primary)',
                      fontWeight: 500,
                    }}
                  >
                    {spec.value}
                  </span>
                  <br />
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.5625rem',
                      color: 'var(--text-faint)',
                      letterSpacing: '0.02em',
                    }}
                  >
                    {spec.detail}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Pipeline flow — editorial, minimal */}
          <div
            style={{
              marginTop: '3rem',
              padding: '2rem 0',
              borderTop: '1px solid var(--border-faint)',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                letterSpacing: '0.2em',
                textTransform: 'uppercase',
                color: 'var(--text-faint)',
                marginBottom: '2rem',
              }}
            >
              Pipeline Flow
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                color: 'var(--text-secondary)',
              }}
            >
              {[
                'Input',
                '→',
                'DeBERTa · Wav2Vec2 · ViT',
                '→',
                '256-D Projection',
                '→',
                'Transformer Fusion',
                '→',
                '6 Emotions',
              ].map((item, i) => (
                <span
                  key={i}
                  style={{
                    color: item === '→' ? 'var(--text-faint)' : 'var(--text-secondary)',
                  }}
                >
                  {item}
                </span>
              ))}
            </div>
          </div>

          {/* Emotion outputs */}
          <div
            style={{
              marginTop: '1rem',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '1.25rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.06em',
            }}
          >
            {EMOTIONS.map((emotion) => {
              const meta = EMOTION_META[emotion];
              return (
                <span key={emotion} style={{ color: meta.color }}>
                  {emotion}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
