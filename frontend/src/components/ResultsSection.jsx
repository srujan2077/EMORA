import React, { useEffect, useState } from 'react';
import { EMOTIONS, EMOTION_META, EMOTION_PALETTE } from '../data/modelSpecs';
import EmotionCanvas from './EmotionCanvas';

/* ============================================================
   RESULTS SECTION — Mark 3 Editorial Design

   VISUAL CONCEPT:
   Before analysis: Quiet, minimal, monochrome.
   After analysis: Emotion-driven color bursts to life!

   1. Oversized Dominant Emotion headline in assigned emotion colour
   2. The Emotion Canvas as visual centerpiece
   3. Six-Emotion Distribution bars with matching colours
   4. Processing & model metadata

   All values bound to genuine API predictions. Never normalized.
   ============================================================ */

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

  // Identify dominant emotion from actual scores
  let dominantEmotion = 'Happy';
  let dominantScore = 0;

  Object.entries(scores).forEach(([emotion, score]) => {
    if (score > dominantScore) {
      dominantScore = score;
      dominantEmotion = emotion;
    }
  });

  const dominantMeta = EMOTION_META[dominantEmotion] || {
    color: EMOTION_PALETTE[dominantEmotion]?.hex || '#111111',
    description: 'Dominant affective signal',
  };

  const dominantPercentage = (dominantScore * 100).toFixed(1);

  // Active modalities flags
  const textActive = modalities?.text?.processed ?? false;
  const audioActive = modalities?.audio?.processed ?? false;
  const videoActive = modalities?.vision?.processed ?? false;

  return (
    <>
      {/* ═══════ 01. DOMINANT EMOTION HEADLINE — VISIBLE COLOUR ═══════ */}
      <section
        className="editorial-section animate-fade-up"
        style={{
          paddingTop: 'clamp(2.5rem, 6vh, 4.5rem)',
          paddingBottom: 'clamp(1.5rem, 3vh, 2.5rem)',
        }}
      >
        <div className="editorial-container">
          {/* Overline */}
          <div
            className="text-overline"
            style={{
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
            }}
          >
            <span>Emotion Detected</span>
            <span style={{ color: 'var(--divider)' }}>/</span>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
              Primary Signal
            </span>
            {segment_id && segment_id !== 'raw-input' && (
              <>
                <span style={{ color: 'var(--divider)' }}>/</span>
                <span style={{ color: 'var(--text-faint)' }}>{segment_id}</span>
              </>
            )}
          </div>

          {/* Large, bold editorial headline using its ASSIGNED EMOTION COLOUR */}
          <h2
            style={{
              fontSize: 'clamp(3.5rem, 9.5vw, 8rem)',
              fontWeight: 800,
              letterSpacing: '-0.04em',
              lineHeight: 0.86,
              color: dominantMeta.color, // ASSIGNED EMOTION COLOUR
              margin: 0,
              marginBottom: '1rem',
              textTransform: 'uppercase',
              textShadow: `0 8px 32px rgba(0,0,0,0.06)`,
            }}
          >
            {dominantEmotion}.
          </h2>

          {/* Percentage & badge row */}
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '1.25rem',
              flexWrap: 'wrap',
              marginBottom: '1.25rem',
            }}
          >
            <AnimatedNumber value={dominantPercentage} color="var(--text-primary)" />

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6875rem',
                  fontWeight: 600,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  background: 'var(--text-primary)',
                  color: 'var(--canvas)',
                  padding: '0.3rem 0.75rem',
                  borderRadius: '0px',
                }}
              >
                Dominant Affect
              </span>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.625rem',
                  color: 'var(--text-muted)',
                  letterSpacing: '0.04em',
                }}
              >
                Sigmoid Score
              </span>
            </div>
          </div>

          {/* Active signals above threshold */}
          {predicted_emotions.length > 0 && (
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                color: 'var(--text-secondary)',
                letterSpacing: '0.02em',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                flexWrap: 'wrap',
              }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Detected signals:</span>
              {predicted_emotions.map((emo) => {
                const isDom = emo.toLowerCase() === dominantEmotion.toLowerCase();
                return (
                  <span
                    key={emo}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      padding: '0.2rem 0.6rem',
                      border: '1px solid var(--divider)',
                      background: isDom ? 'var(--text-primary)' : 'var(--surface)',
                      color: isDom ? 'var(--canvas)' : 'var(--text-primary)',
                      fontWeight: isDom ? 600 : 500,
                      fontSize: '0.625rem',
                      letterSpacing: '0.04em',
                      textTransform: 'uppercase',
                    }}
                  >
                    {isDom ? `★ ${emo}` : emo}
                  </span>
                );
              })}
            </div>
          )}

          {/* Accent rule — editorial monochrome */}
          <div
            style={{
              width: '48px',
              height: '1px',
              background: 'var(--text-primary)',
              marginTop: '1.75rem',
            }}
            aria-hidden="true"
          />
        </div>
      </section>

      {/* Divider */}
      <div className="editorial-container">
        <div className="editorial-divider" />
      </div>

      {/* ═══════ 02. EMOTION CANVAS — THE VISUAL CENTREPIECE ═══════ */}
      <section className="editorial-section animate-fade-in" style={{ paddingBottom: '2.5rem' }}>
        <div className="editorial-container">
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              marginBottom: '1.25rem',
            }}
          >
            <div className="text-overline">
              Emotional Landscape // Generative Projection
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                letterSpacing: '0.08em',
                color: 'var(--text-muted)',
                textTransform: 'uppercase',
              }}
            >
              Live Tensor Visualization
            </div>
          </div>

          {/* Centerpiece Emotion Canvas */}
          <EmotionCanvas
            scores={scores}
            dominantEmotion={dominantEmotion}
            detectedEmotions={predicted_emotions}
          />

          {/* Canvas legend — every emotion with its exact matching color */}
          <div
            style={{
              marginTop: '1.25rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: '0.75rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.04em',
              padding: '0.875rem 1rem',
              background: 'var(--surface)',
              border: '1px solid var(--divider)',
            }}
          >
            {EMOTIONS.map((emotion) => {
              const meta = EMOTION_META[emotion] || { color: '#111' };
              const score = scores[emotion] !== undefined ? scores[emotion] : 0;
              const isDominant = emotion.toLowerCase() === dominantEmotion.toLowerCase();
              const isDetected = predicted_emotions.some(
                (d) => d.toLowerCase() === emotion.toLowerCase()
              );

              return (
                <div
                  key={emotion}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '0.5rem',
                  }}
                >
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      fontWeight: isDominant ? 600 : 400,
                      color: isDominant ? 'var(--text-primary)' : 'var(--text-secondary)',
                    }}
                  >
                    <span
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: meta.color,
                        display: 'inline-block',
                        flexShrink: 0,
                      }}
                      aria-hidden="true"
                    />
                    <span>{emotion}</span>
                  </span>

                  <span
                    style={{
                      fontWeight: isDominant || isDetected ? 600 : 400,
                      color: isDominant || isDetected ? meta.color : 'var(--text-muted)',
                    }}
                  >
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="editorial-container">
        <div className="editorial-divider" />
      </div>

      {/* ═══════ 03. EMOTION DISTRIBUTION — MATCHING PROGRESS BARS ═══════ */}
      <section className="editorial-section">
        <div className="editorial-container">
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              marginBottom: '2rem',
            }}
          >
            <div className="text-overline">
              Emotion Distribution // Multi-Label Probabilities
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.04em',
              }}
            >
              Vertical bars indicate fixed 0.5 decision thresholds
            </div>
          </div>

          {/* Refined editorial distribution bars */}
          <div className="stagger-children" style={{ maxWidth: '680px' }}>
            {EMOTIONS.map((emotion) => {
              const score = scores[emotion] !== undefined ? scores[emotion] : 0;
              const emotionKey = emotion.toLowerCase();
              const detail = emotions_detail[emotionKey];
              const threshold = detail?.threshold ?? 0.5;
              const percentage = (score * 100).toFixed(1);
              const isDominant = emotion.toLowerCase() === dominantEmotion.toLowerCase();
              const isDetected = detail?.detected ?? (score >= threshold);
              const meta = EMOTION_META[emotion] || { color: '#111' };

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
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span
                        style={{
                          fontSize: '0.875rem',
                          fontWeight: isDominant ? 700 : isDetected ? 600 : 400,
                          color: 'var(--text-primary)',
                          letterSpacing: '0.01em',
                          minWidth: '72px',
                        }}
                      >
                        {emotion}
                      </span>

                      {/* Monochrome Status Badges */}
                      {isDominant && (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.5625rem',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            background: 'var(--text-primary)',
                            color: 'var(--canvas)',
                            padding: '0.12rem 0.5rem',
                            fontWeight: 600,
                          }}
                          aria-label="Dominant signal"
                        >
                          Dominant
                        </span>
                      )}

                      {isDetected && !isDominant && (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.5625rem',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            border: '1px solid var(--text-primary)',
                            color: 'var(--text-primary)',
                            padding: '0.1rem 0.45rem',
                            fontWeight: 500,
                          }}
                          aria-label="Active signal above threshold"
                        >
                          Active
                        </span>
                      )}

                      {!isDetected && !isDominant && (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.5625rem',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            color: 'var(--text-faint)',
                          }}
                        >
                          Sub-threshold
                        </span>
                      )}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.625rem',
                          color: 'var(--text-faint)',
                        }}
                      >
                        (τ = {(threshold * 100).toFixed(0)}%)
                      </span>

                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: isDominant ? '0.9375rem' : '0.8125rem',
                          fontWeight: isDominant ? 700 : isDetected ? 600 : 500,
                          color: meta.color, // ASSIGNED EMOTION COLOUR
                        }}
                        aria-label={`${emotion} probability: ${percentage} percent`}
                      >
                        {percentage}%
                      </span>
                    </div>
                  </div>

                  {/* Bar track — EVERY emotion uses its assigned matching emotion colour */}
                  <div
                    style={{
                      width: '100%',
                      height: isDominant ? '5px' : isDetected ? '4px' : '3px',
                      background: 'var(--divider-light)',
                      position: 'relative',
                      overflow: 'hidden',
                      borderRadius: '1px',
                    }}
                    role="meter"
                    aria-label={`${emotion} probability`}
                    aria-valuenow={parseFloat(percentage)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    {/* Calibrated threshold marker */}
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${Math.min(99, threshold * 100)}%`,
                        width: '1.5px',
                        background: 'rgba(255, 255, 255, 0.4)',
                        zIndex: 2,
                      }}
                      title={`Decision threshold: ${(threshold * 100).toFixed(1)}%`}
                      aria-hidden="true"
                    />

                    {/* Matching Emotion Colour Progress Fill */}
                    <div
                      className="animate-bar-grow"
                      style={{
                        height: '100%',
                        width: `${Math.min(100, Math.max(0.75, score * 100))}%`,
                        backgroundColor: meta.color, // MATCHING COLOUR FOR EVERY EMOTION
                        opacity: isDominant ? 1 : isDetected ? 0.9 : 0.35,
                        transition: 'width 1.2s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.4s ease',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Multi-label explanation */}
          <p
            style={{
              marginTop: '2rem',
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              fontStyle: 'italic',
              maxWidth: '520px',
              lineHeight: 1.6,
            }}
          >
            Multiple emotional signals can be active simultaneously. The model employs independent sigmoid classifiers per emotion with a fixed 0.5 decision threshold.
          </p>
        </div>
      </section>

      {/* Divider */}
      <div className="editorial-container">
        <div className="editorial-divider" />
      </div>

      {/* ═══════ 04. PROCESSING STATUS & MODEL METADATA ═══════ */}
      <section className="editorial-section" style={{ paddingTop: '1.5rem', paddingBottom: '1.5rem' }}>
        <div className="editorial-container">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.06em',
              color: 'var(--text-faint)',
              flexWrap: 'wrap',
              gap: '1.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
              <span style={{ color: textActive ? 'var(--text-primary)' : 'var(--text-faint)' }}>
                Text: {textActive ? 'DeBERTa-v3 ✓' : '—'}
              </span>
              <span style={{ color: audioActive ? 'var(--text-primary)' : 'var(--text-faint)' }}>
                Audio: {audioActive ? 'Wav2Vec2 ✓' : '—'}
              </span>
              <span style={{ color: videoActive ? 'var(--text-primary)' : 'var(--text-faint)' }}>
                Video: {videoActive ? 'ViT-base ✓' : '—'}
              </span>
              <span style={{ color: 'var(--text-primary)' }}>
                Fusion: Transformer ✓
              </span>
            </div>

            <div style={{ color: 'var(--text-muted)', fontSize: '0.625rem' }}>
              Architecture: <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{model}</span>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

/* ============================================================
   ANIMATED NUMBER — large percentage count-up
   ============================================================ */
function AnimatedNumber({ value, color = 'var(--text-primary)' }) {
  const [displayValue, setDisplayValue] = useState('0.0');
  const reducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    if (reducedMotion) {
      setDisplayValue(value);
      return;
    }

    const target = parseFloat(value);
    const duration = 800;
    const startTime = performance.now();

    let rafId;
    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = (target * eased).toFixed(1);
      setDisplayValue(current);

      if (progress < 1) {
        rafId = requestAnimationFrame(animate);
      }
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [value, reducedMotion]);

  return (
    <div
      className="animate-count-up"
      style={{
        fontSize: 'clamp(3rem, 8vw, 6.5rem)',
        fontWeight: 800,
        fontFamily: 'var(--font-mono)',
        letterSpacing: '-0.04em',
        lineHeight: 0.9,
        color: color,
      }}
      aria-live="polite"
    >
      {displayValue}
      <span
        style={{
          fontSize: '0.3em',
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
