import React, { useState } from 'react';
import {
  MARK2_BENCHMARK,
  MARK3_BENCHMARK,
  EMOTIONS,
} from '../data/modelSpecs';

export default function BenchmarksSection() {
  const [selectedBenchmark, setSelectedBenchmark] = useState('mark3');

  const current = selectedBenchmark === 'mark3' ? MARK3_BENCHMARK : MARK2_BENCHMARK;

  return (
    <section
      style={{
        padding: 'clamp(4rem, 8vh, 7rem) 0',
        borderTop: '1px solid var(--divider)',
      }}
    >
      <div className="editorial-container">
        {/* Section Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            borderBottom: '1px solid var(--divider)',
            paddingBottom: '1rem',
            marginBottom: 'clamp(2.5rem, 5vh, 4rem)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--text-muted)',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>[04/04]</span>
            <span>EMPIRICAL BENCHMARK // CMU-MOSEI</span>
          </div>
          <span>22,856 ALIGNED TEST UTTERANCES · HELD-OUT SPLIT</span>
        </div>

        {/* Title + Model Switcher */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-end',
            flexWrap: 'wrap',
            gap: '1.5rem',
            marginBottom: '3rem',
          }}
        >
          <div style={{ maxWidth: '640px' }}>
            <h2
              style={{
                fontSize: 'clamp(2rem, 4.2vw, 3.5rem)',
                fontWeight: 700,
                letterSpacing: '-0.03em',
                lineHeight: 1.05,
                margin: '0 0 1rem 0',
                textTransform: 'uppercase',
              }}
            >
              Rigorous Evaluation on Standardized Split.
            </h2>
            <p
              style={{
                fontSize: '0.9375rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.7,
                margin: 0,
              }}
            >
              CMU-MOSEI multi-label emotion recognition evaluation across six affective classes. All metrics computed under strict standardized data splits with fixed 0.5 decision thresholds.
            </p>
          </div>

          {/* Model Switcher */}
          <div
            style={{
              display: 'inline-flex',
              border: '1px solid var(--divider)',
              background: 'var(--surface)',
            }}
          >
            <button
              type="button"
              onClick={() => setSelectedBenchmark('mark3')}
              style={{
                padding: '0.5rem 1rem',
                fontSize: '0.6875rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: selectedBenchmark === 'mark3' ? 600 : 400,
                border: 'none',
                background: selectedBenchmark === 'mark3' ? 'var(--text-primary)' : 'transparent',
                color: selectedBenchmark === 'mark3' ? 'var(--canvas)' : 'var(--text-muted)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Mark 3 (5-Seed Ensemble)
            </button>
            <button
              type="button"
              onClick={() => setSelectedBenchmark('mark2')}
              style={{
                padding: '0.5rem 1rem',
                fontSize: '0.6875rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: selectedBenchmark === 'mark2' ? 600 : 400,
                border: 'none',
                borderLeft: '1px solid var(--divider)',
                background: selectedBenchmark === 'mark2' ? 'var(--text-primary)' : 'transparent',
                color: selectedBenchmark === 'mark2' ? 'var(--canvas)' : 'var(--text-muted)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Mark 2 (Single Transformer)
            </button>
          </div>
        </div>

        {/* Top High-Impact Metrics */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1.5rem',
            marginBottom: '3.5rem',
          }}
        >
          {[
            { label: 'MICRO-F1 (OVERALL)', value: current.microF1, note: 'Primary multi-label classification benchmark' },
            { label: 'MACRO-F1 (CLASS AVG)', value: current.macroF1, note: 'Unweighted mean across all 6 affective classes' },
            { label: 'SUBSET ACCURACY', value: current.subsetAccuracy, note: 'Exact multi-label match across full vector' },
          ].map((metric) => (
            <div
              key={metric.label}
              style={{
                border: '1px solid var(--divider)',
                background: 'var(--surface)',
                padding: '1.75rem',
              }}
            >
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.5625rem',
                  letterSpacing: '0.12em',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  marginBottom: '0.5rem',
                }}
              >
                {metric.label}
              </div>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 'clamp(2.5rem, 4.5vw, 3.5rem)',
                  fontWeight: 700,
                  letterSpacing: '-0.04em',
                  lineHeight: 1,
                  color: 'var(--text-primary)',
                  marginBottom: '0.75rem',
                }}
              >
                {metric.value.toFixed(4)}
              </div>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.625rem',
                  color: 'var(--text-faint)',
                  lineHeight: 1.5,
                }}
              >
                {metric.note}
              </div>
            </div>
          ))}
        </div>

        {/* Per-Emotion Class Breakdown (Editorial Table) */}
        <div
          style={{
            border: '1px solid var(--divider)',
            background: 'var(--surface)',
            padding: 'clamp(1.5rem, 3vw, 2.5rem)',
            marginBottom: '3.5rem',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              borderBottom: '1px solid var(--divider-light)',
              paddingBottom: '0.75rem',
              marginBottom: '1.5rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
            }}
          >
            <span>Class-Wise Performance Breakdown ({selectedBenchmark.toUpperCase()})</span>
            <span>Threshold Calibration / Metric F1</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {EMOTIONS.map((emotion) => {
              const f1 = current.perEmotionF1[emotion];
              const percentage = (f1 * 100).toFixed(1);
              const threshold = current.thresholds ? current.thresholds[emotion] : 0.5;

              return (
                <div
                  key={emotion}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '120px 1fr 80px',
                    alignItems: 'center',
                    gap: '1.5rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span
                      style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: 'var(--text-primary)',
                      }}
                    />
                    <span
                      style={{
                        fontSize: '0.875rem',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                      }}
                    >
                      {emotion}
                    </span>
                  </div>

                  {/* Neutral Monochrome Bar */}
                  <div
                    style={{
                      height: '4px',
                      background: 'var(--divider-light)',
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${f1 * 100}%`,
                        background: 'var(--text-primary)',
                        opacity: 0.85,
                        transition: 'width 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
                      }}
                    />
                  </div>

                  <div
                    style={{
                      textAlign: 'right',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8125rem',
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                    }}
                  >
                    {percentage}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Architecture Evolution Grid */}
        <div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
              marginBottom: '1.5rem',
            }}
          >
            Architectural Evolution Across Mark Generations
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: '1.5rem',
            }}
          >
            {[
              {
                gen: 'MARK 01 // BASELINE',
                sub: 'BiLSTM Stack (Legacy CSD)',
                items: ['GloVe Word Vectors (300-D)', 'COVAREP Acoustic (74-D)', 'FACET Visual (35-D)', 'Concatenation → Dense MLP'],
                f1: '0.4901 (Val Loss)',
              },
              {
                gen: 'MARK 02 // TRANSFORMER',
                sub: 'Deep Multimodal Foundation',
                items: ['DeBERTa-v3-base (768-D)', 'Wav2Vec2-base-960h (768-D)', 'ViT-base-patch16 (768-D)', '2-Layer Multimodal Self-Attention'],
                f1: '0.5316 Micro-F1',
              },
              {
                gen: 'MARK 03 // SOTA ENSEMBLE',
                sub: 'Causal Windows + 5-Seed Ensemble',
                items: ['Causal Temporal Dialogue Windowing', 'Controlled split_at_gap Policy', '5-Seed Ensemble Averaging', 'Validation-Calibrated Thresholds'],
                f1: '0.5333 Micro-F1 (SOTA)',
              },
            ].map((col) => (
              <div
                key={col.gen}
                style={{
                  border: '1px solid var(--divider)',
                  background: 'var(--surface)',
                  padding: '1.5rem',
                }}
              >
                <div
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.5625rem',
                    letterSpacing: '0.1em',
                    color: 'var(--text-muted)',
                    marginBottom: '0.35rem',
                  }}
                >
                  {col.gen}
                </div>
                <div
                  style={{
                    fontSize: '1rem',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: '1rem',
                  }}
                >
                  {col.sub}
                </div>

                <ul
                  style={{
                    listStyle: 'none',
                    padding: 0,
                    margin: '0 0 1.25rem 0',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.6875rem',
                    lineHeight: 1.8,
                    color: 'var(--text-secondary)',
                  }}
                >
                  {col.items.map((item, i) => (
                    <li key={i}>— {item}</li>
                  ))}
                </ul>

                <div
                  style={{
                    borderTop: '1px solid var(--divider-light)',
                    paddingTop: '0.75rem',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.6875rem',
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                  }}
                >
                  {col.f1}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
