import React from 'react';
import {
  MARK2_BENCHMARK,
  EMOTIONS,
  EMOTION_META,
} from '../data/modelSpecs';

export default function BenchmarksSection() {
  return (
    <div className="editorial-container animate-fade-up">
      {/* Hero heading */}
      <div style={{ marginBottom: 'clamp(3rem, 6vh, 5rem)' }}>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.5625rem',
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            color: 'var(--text-faint)',
            marginBottom: '1.5rem',
          }}
        >
          Official Benchmark
        </div>

        <h2
          style={{
            fontSize: 'clamp(1.5rem, 3vw, 2.5rem)',
            fontWeight: 600,
            letterSpacing: '-0.025em',
            color: 'var(--text-primary)',
            marginBottom: '1rem',
          }}
        >
          CMU-MOSEI Test Evaluation
        </h2>

        <p
          style={{
            fontSize: '0.9375rem',
            color: 'var(--text-secondary)',
            maxWidth: '520px',
            lineHeight: 1.7,
          }}
        >
          Multi-label emotion recognition metrics on the held-out test split.
          22,856 utterances across six affective dimensions.
        </p>
      </div>

      {/* Top-level metrics — large numbers */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '2rem',
          marginBottom: 'clamp(3rem, 6vh, 5rem)',
          maxWidth: '640px',
        }}
      >
        {[
          { label: 'Micro-F1', value: MARK2_BENCHMARK.microF1 },
          { label: 'Macro-F1', value: MARK2_BENCHMARK.macroF1 },
          { label: 'Subset Acc', value: MARK2_BENCHMARK.subsetAccuracy },
        ].map((metric) => (
          <div key={metric.label}>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--text-faint)',
                marginBottom: '0.5rem',
              }}
            >
              {metric.label}
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'clamp(1.75rem, 3vw, 2.5rem)',
                fontWeight: 600,
                color: 'var(--text-primary)',
                letterSpacing: '-0.02em',
              }}
            >
              {metric.value.toFixed(4)}
            </div>
          </div>
        ))}
      </div>

      {/* Divider */}
      <div className="editorial-divider" style={{ marginBottom: 'clamp(2.5rem, 5vh, 4rem)' }} />

      {/* Per-Emotion Breakdown — editorial table */}
      <div>
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
          Per-Emotion F1 Score
        </div>

        <div style={{ maxWidth: '560px' }}>
          {EMOTIONS.map((emotion, i) => {
            const f1 = MARK2_BENCHMARK.perEmotionF1[emotion];
            const meta = EMOTION_META[emotion];
            const percentage = (f1 * 100).toFixed(1);

            return (
              <div
                key={emotion}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '1rem 0',
                  borderBottom: i < EMOTIONS.length - 1 ? '1px solid var(--border-faint)' : 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      background: meta.color,
                      display: 'inline-block',
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: '0.875rem',
                      color: 'var(--text-primary)',
                      fontWeight: 500,
                      minWidth: '72px',
                    }}
                  >
                    {emotion}
                  </span>
                </div>

                {/* Bar */}
                <div
                  style={{
                    flex: 1,
                    maxWidth: '240px',
                    height: '2px',
                    background: 'var(--border-faint)',
                    margin: '0 1.5rem',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    className="animate-bar-grow"
                    style={{
                      height: '100%',
                      width: `${f1 * 100}%`,
                      background: meta.color,
                      opacity: 0.7,
                    }}
                  />
                </div>

                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.8125rem',
                    color: 'var(--text-secondary)',
                    fontWeight: 500,
                    minWidth: '50px',
                    textAlign: 'right',
                  }}
                >
                  {percentage}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Divider */}
      <div className="editorial-divider" style={{ margin: 'clamp(2.5rem, 5vh, 4rem) 0' }} />

      {/* Architecture Evolution — text only, no cards */}
      <div style={{ maxWidth: '640px' }}>
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
          Architecture Evolution
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '3rem',
          }}
        >
          {/* Mark 1 */}
          <div>
            <div
              style={{
                fontSize: '0.8125rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                marginBottom: '1rem',
              }}
            >
              Mark 1 — Baseline
            </div>
            <ul
              style={{
                listStyle: 'none',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                lineHeight: 2.2,
                color: 'var(--text-muted)',
              }}
            >
              <li>GloVe + BiLSTM</li>
              <li>COVAREP + BiLSTM</li>
              <li>FACET + BiLSTM</li>
              <li>Concat → MLP</li>
            </ul>
          </div>

          {/* Mark 2 */}
          <div>
            <div
              style={{
                fontSize: '0.8125rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginBottom: '1rem',
              }}
            >
              Mark 2 — Transformer
            </div>
            <ul
              style={{
                listStyle: 'none',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                lineHeight: 2.2,
                color: 'var(--text-secondary)',
              }}
            >
              <li>DeBERTa-v3-base</li>
              <li>Wav2Vec2-base-960h</li>
              <li>ViT-base-patch16-224</li>
              <li>2-Layer Self-Attention</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
