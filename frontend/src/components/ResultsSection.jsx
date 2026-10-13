import React, { useEffect, useState } from 'react';
import { EMOTIONS, EMOTION_META } from '../data/modelSpecs';

export default function ResultsSection({ prediction }) {
  if (!prediction || !prediction.scores) return null;

  const {
    scores = {},
    predicted_emotions = [],
    emotions_detail = {},
    modalities = {},
    model = 'Mark 2',
    segment_id,
  } = prediction;

  // Find dominant emotion
  let dominantEmotion = 'Happy';
  let dominantScore = 0;

  Object.entries(scores).forEach(([emotion, score]) => {
    if (score > dominantScore) {
      dominantScore = score;
      dominantEmotion = emotion;
    }
  });

  const dominantMeta = EMOTION_META[dominantEmotion] || {
    color: '#7877c6',
    description: 'Predicted emotional affect',
  };

  const dominantPercentage = (dominantScore * 100).toFixed(1);

  // Active modalities
  const textActive = modalities?.text?.processed ?? false;
  const audioActive = modalities?.audio?.processed ?? false;
  const videoActive = modalities?.vision?.processed ?? false;

  return (
    <>
      {/* ═══════ DOMINANT RESULT — The strongest visual moment ═══════ */}
      <section
        className="editorial-section animate-fade-up"
        style={{
          paddingTop: 'clamp(4rem, 10vh, 7rem)',
          paddingBottom: 'clamp(3rem, 6vh, 5rem)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Background glow — very subtle */}
        <div
          style={{
            position: 'absolute',
            top: '20%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '500px',
            height: '300px',
            background: `radial-gradient(ellipse, ${dominantMeta.color}08 0%, transparent 70%)`,
            pointerEvents: 'none',
          }}
        />

        <div className="editorial-container">
          {/* Overline */}
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
            Analysis Result
            {segment_id && segment_id !== 'raw-input' && (
              <span style={{ marginLeft: '1rem', color: 'var(--text-muted)' }}>
                · {segment_id}
              </span>
            )}
          </div>

          {/* Dominant Emotion — LARGE */}
          <div style={{ marginBottom: '1rem' }}>
            <h2
              className="animate-fade-up"
              style={{
                fontSize: 'clamp(3rem, 8vw, 7rem)',
                fontWeight: 700,
                letterSpacing: '-0.03em',
                lineHeight: 0.9,
                color: dominantMeta.color,
                margin: 0,
              }}
            >
              {dominantEmotion.toUpperCase()}
            </h2>
          </div>

          {/* Percentage — visually powerful */}
          <AnimatedNumber value={dominantPercentage} color={dominantMeta.color} />

          {/* Subtle descriptor */}
          <p
            style={{
              marginTop: '1.5rem',
              fontSize: '0.875rem',
              color: 'var(--text-secondary)',
              maxWidth: '360px',
              lineHeight: 1.7,
            }}
          >
            Dominant emotional signal
          </p>

          {/* Multi-label note */}
          {predicted_emotions.length > 0 && (
            <div
              style={{
                marginTop: '1rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.625rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.02em',
              }}
            >
              {predicted_emotions.length} active signal{predicted_emotions.length > 1 ? 's' : ''} above threshold ·{' '}
              <span style={{ color: 'var(--text-secondary)' }}>
                {predicted_emotions.join(', ')}
              </span>
            </div>
          )}
        </div>
      </section>

      {/* Divider */}
      <div className="editorial-container">
        <div className="editorial-divider" />
      </div>

      {/* ═══════ EMOTION DISTRIBUTION ═══════ */}
      <section className="editorial-section">
        <div className="editorial-container">
          {/* Section heading */}
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              color: 'var(--text-faint)',
              marginBottom: '3rem',
            }}
          >
            Emotion Distribution
          </div>

          {/* Editorial horizontal bars */}
          <div className="stagger-children" style={{ maxWidth: '640px' }}>
            {EMOTIONS.map((emotion) => {
              const score = scores[emotion] !== undefined ? scores[emotion] : 0;
              const emotionKey = emotion.toLowerCase();
              const detail = emotions_detail[emotionKey];
              const threshold = detail?.threshold ?? 0.5;
              const percentage = (score * 100).toFixed(1);
              const isDominant = emotion === dominantEmotion;
              const isDetected = detail?.detected ?? (score >= threshold);
              const meta = EMOTION_META[emotion] || { color: '#7877c6' };

              return (
                <div
                  key={emotion}
                  style={{
                    marginBottom: '1.75rem',
                    position: 'relative',
                  }}
                >
                  {/* Label row */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'baseline',
                      justifyContent: 'space-between',
                      marginBottom: '0.5rem',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
                      <span
                        style={{
                          fontSize: '0.8125rem',
                          fontWeight: isDominant ? 600 : 400,
                          color: isDominant ? 'var(--text-primary)' : 'var(--text-secondary)',
                          letterSpacing: '0.01em',
                          minWidth: '72px',
                        }}
                      >
                        {emotion}
                      </span>

                      {isDominant && (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.5625rem',
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                            color: meta.color,
                          }}
                        >
                          Dominant
                        </span>
                      )}

                      {isDetected && !isDominant && (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.5625rem',
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                            color: 'var(--text-muted)',
                          }}
                        >
                          Active
                        </span>
                      )}
                    </div>

                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: isDominant ? '0.9375rem' : '0.8125rem',
                        fontWeight: isDominant ? 600 : 400,
                        color: isDominant ? 'var(--text-primary)' : 'var(--text-secondary)',
                      }}
                    >
                      {percentage}%
                    </span>
                  </div>

                  {/* Bar track */}
                  <div
                    style={{
                      width: '100%',
                      height: isDominant ? '3px' : '2px',
                      background: 'var(--border-faint)',
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Threshold marker */}
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${Math.min(99, threshold * 100)}%`,
                        width: '1px',
                        background: 'rgba(255,255,255,0.08)',
                        zIndex: 1,
                      }}
                    />

                    {/* Fill */}
                    <div
                      className="animate-bar-grow"
                      style={{
                        height: '100%',
                        width: `${Math.min(100, Math.max(0.5, score * 100))}%`,
                        background: isDominant
                          ? meta.color
                          : isDetected
                          ? meta.color
                          : 'rgba(255,255,255,0.1)',
                        opacity: isDominant ? 1 : isDetected ? 0.7 : 0.35,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Multi-label explanation — one subtle line */}
          <div
            style={{
              marginTop: '2rem',
              fontSize: '0.75rem',
              color: 'var(--text-faint)',
              fontStyle: 'italic',
              maxWidth: '480px',
            }}
          >
            Multiple emotional signals may be active simultaneously. Independent sigmoid classifiers per emotion.
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="editorial-container">
        <div className="editorial-divider" />
      </div>

      {/* ═══════ PROCESSING STATUS — minimal ═══════ */}
      <section className="editorial-section" style={{ paddingTop: '2rem', paddingBottom: '2rem' }}>
        <div className="editorial-container">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '2rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.06em',
              color: 'var(--text-faint)',
            }}
          >
            <span style={{ color: textActive ? 'var(--text-secondary)' : 'var(--text-faint)' }}>
              Text {textActive ? '✓' : '—'}
            </span>
            <span style={{ color: audioActive ? 'var(--text-secondary)' : 'var(--text-faint)' }}>
              Audio {audioActive ? '✓' : '—'}
            </span>
            <span style={{ color: videoActive ? 'var(--text-secondary)' : 'var(--text-faint)' }}>
              Video {videoActive ? '✓' : '—'}
            </span>
            <span style={{ color: 'var(--accent)' }}>
              Fusion ✓
            </span>

            <span style={{ marginLeft: 'auto', color: 'var(--text-faint)', fontSize: '0.5625rem' }}>
              {model}
            </span>
          </div>
        </div>
      </section>
    </>
  );
}

/* ============================================================
   ANIMATED NUMBER — large percentage with count-up feel
   ============================================================ */
function AnimatedNumber({ value, color }) {
  const [displayValue, setDisplayValue] = useState('0.0');

  useEffect(() => {
    const target = parseFloat(value);
    const duration = 800;
    const startTime = performance.now();

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = (target * eased).toFixed(1);
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value]);

  return (
    <div
      className="animate-count-up"
      style={{
        fontSize: 'clamp(3.5rem, 9vw, 8rem)',
        fontWeight: 700,
        fontFamily: 'var(--font-mono)',
        letterSpacing: '-0.04em',
        lineHeight: 0.9,
        color: 'var(--text-primary)',
      }}
    >
      {displayValue}
      <span
        style={{
          fontSize: '0.35em',
          fontWeight: 400,
          color: 'var(--text-muted)',
          marginLeft: '0.1em',
        }}
      >
        %
      </span>
    </div>
  );
}
