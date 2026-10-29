import React, { useMemo, useRef, useEffect, useState } from 'react';
import { EMOTIONS, EMOTION_PALETTE } from '../data/modelSpecs';

/* ============================================================
   EMOTION CANVAS — Signature Mark 3 Visual Feature

   Transforms the actual six-emotion prediction vector into a
   vivid, breathing, abstract data-art composition.

   BEFORE ANALYSIS: Pure monochrome baseline.
   AFTER ANALYSIS: The predicted emotions bring the canvas to life.

   Uses genuine API response values. Never normalizes, fabricates,
   or modifies sigmoid probabilities.
   ============================================================ */

// Calibrated field focal coordinates across the 16:9 editorial composition
const FIELD_CONFIGS = [
  { cx: 0.28, cy: 0.32, baseSize: 0.42, delay: '0s' },    // Happy — top left
  { cx: 0.72, cy: 0.68, baseSize: 0.40, delay: '1.2s' },  // Sad — bottom right
  { cx: 0.68, cy: 0.28, baseSize: 0.38, delay: '0.6s' },  // Anger — top right
  { cx: 0.30, cy: 0.72, baseSize: 0.36, delay: '1.8s' },  // Surprise — bottom left
  { cx: 0.82, cy: 0.46, baseSize: 0.32, delay: '2.4s' },  // Disgust — mid right
  { cx: 0.48, cy: 0.50, baseSize: 0.44, delay: '0.3s' },  // Fear — centre
];

export default function EmotionCanvas({ scores, dominantEmotion = null, detectedEmotions = [] }) {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const hasScores = Boolean(scores && Object.keys(scores).length > 0);

  // ResizeObserver for dynamic responsiveness
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Compute fields from actual model prediction scores
  const fields = useMemo(() => {
    if (!hasScores) return [];

    // Identify dominant emotion if not passed directly
    let domKey = dominantEmotion;
    let maxVal = -1;
    if (!domKey) {
      EMOTIONS.forEach((emo) => {
        const val = scores[emo] ?? scores[emo.toLowerCase()] ?? 0;
        if (val > maxVal) {
          maxVal = val;
          domKey = emo;
        }
      });
    }

    return EMOTIONS.map((emotion, i) => {
      const rawScore = scores[emotion] ?? scores[emotion.toLowerCase()] ?? 0;
      const config = FIELD_CONFIGS[i];
      const palette = EMOTION_PALETTE[emotion];
      const isDominant = emotion.toLowerCase() === (domKey || '').toLowerCase();
      const isDetected = detectedEmotions.some(
        (d) => d.toLowerCase() === emotion.toLowerCase()
      ) || (isDominant && rawScore > 0);

      // Radial field size proportional to prediction probability
      const sizeMultiplier = isDominant
        ? 0.55 + rawScore * 0.95
        : isDetected
        ? 0.35 + rawScore * 0.75
        : 0.18 + rawScore * 0.55;

      // Opacity scaling: dominant & detected are vivid, non-zero sub-threshold are gentle atmospheric tones
      const opacity = rawScore < 0.005
        ? 0
        : isDominant
        ? Math.min(0.96, 0.75 + rawScore * 0.22)
        : isDetected
        ? Math.min(0.85, 0.55 + rawScore * 0.3)
        : Math.min(0.45, 0.15 + rawScore * 0.35);

      return {
        emotion,
        color: palette.hex,
        cx: config.cx,
        cy: config.cy,
        delay: config.delay,
        size: config.baseSize * sizeMultiplier,
        opacity,
        isDominant,
        isDetected,
        score: rawScore,
      };
    });
  }, [scores, hasScores, dominantEmotion, detectedEmotions]);

  return (
    <div
      ref={containerRef}
      className="emotion-canvas-container"
      role="img"
      aria-label={
        hasScores
          ? `Emotion Canvas visualization: ${EMOTIONS.map(
              (e) =>
                `${e} ${((scores[e] ?? scores[e.toLowerCase()] ?? 0) * 100).toFixed(1)}%`
            ).join(', ')}`
          : 'Emotion Canvas — awaiting analysis'
      }
    >
      {/* Subtle cartographic grid overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          pointerEvents: 'none',
          zIndex: 4,
        }}
        aria-hidden="true"
      />

      {/* Axis crosshair markers */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '32px',
          height: '1px',
          background: 'rgba(255,255,255,0.18)',
          transform: 'translate(-50%, -50%)',
          zIndex: 5,
          pointerEvents: 'none',
        }}
        aria-hidden="true"
      />
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '1px',
          height: '32px',
          background: 'rgba(255,255,255,0.18)',
          transform: 'translate(-50%, -50%)',
          zIndex: 5,
          pointerEvents: 'none',
        }}
        aria-hidden="true"
      />

      {hasScores ? (
        <>
          {/* Active Emotion Color Fields */}
          {fields.map((field) => {
            const px = field.cx * 100;
            const py = field.cy * 100;
            const baseDim = Math.min(dimensions.width || 640, (dimensions.height || 400) * 1.5);
            const size = Math.max(40, field.size * baseDim);

            return (
              <div
                key={field.emotion}
                className="emotion-canvas-field"
                style={{
                  left: `${px}%`,
                  top: `${py}%`,
                  width: `${size}px`,
                  height: `${size}px`,
                  background: `radial-gradient(circle at 50% 50%, ${field.color} 0%, ${field.color}E6 28%, ${field.color}88 55%, ${field.color}22 75%, transparent 100%)`,
                  opacity: field.opacity,
                  filter: field.isDominant ? 'blur(36px)' : 'blur(28px)',
                  mixBlendMode: 'screen',
                  transform: `translate(-50%, -50%) scale(${field.isDominant ? 1.08 : 1.0})`,
                  animationDelay: field.delay,
                  zIndex: field.isDominant ? 3 : field.isDetected ? 2 : 1,
                }}
                aria-hidden="true"
              />
            );
          })}

          {/* Focal Pins & Micro-Labels for Active/Dominant Signals */}
          {fields
            .filter((f) => (f.isDetected || f.isDominant) && f.score > 0.05)
            .map((field) => {
              const px = field.cx * 100;
              const py = field.cy * 100;
              const percentage = (field.score * 100).toFixed(1);

              return (
                <div
                  key={`pin-${field.emotion}`}
                  style={{
                    position: 'absolute',
                    left: `${px}%`,
                    top: `${py}%`,
                    transform: 'translate(-50%, -50%)',
                    zIndex: 6,
                    pointerEvents: 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '4px',
                    transition: 'all 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
                  }}
                  aria-hidden="true"
                >
                  {/* Concentric coordinate marker */}
                  <div
                    style={{
                      width: field.isDominant ? '14px' : '10px',
                      height: field.isDominant ? '14px' : '10px',
                      borderRadius: '50%',
                      border: `1.5px solid ${field.color}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: '#0A0A0A',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.7)',
                    }}
                  >
                    <div
                      style={{
                        width: field.isDominant ? '6px' : '4px',
                        height: field.isDominant ? '6px' : '4px',
                        borderRadius: '50%',
                        backgroundColor: field.color,
                      }}
                    />
                  </div>

                  {/* Micro label */}
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.625rem',
                      fontWeight: 600,
                      color: '#FFFFFF',
                      background: 'rgba(10, 10, 10, 0.94)',
                      backdropFilter: 'blur(4px)',
                      padding: '1px 6px',
                      borderRadius: '2px',
                      border: '1px solid var(--divider)',
                      letterSpacing: '0.04em',
                      whiteSpace: 'nowrap',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.7)',
                    }}
                  >
                    {field.emotion} <span style={{ color: field.color }}>{percentage}%</span>
                  </div>
                </div>
              );
            })}
        </>
      ) : (
        /* Empty State — Strict Monochrome Editorial Baseline */
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 5,
          }}
        >
          {/* Concentric hairline guide rings */}
          <div
            style={{
              position: 'absolute',
              width: '240px',
              height: '240px',
              borderRadius: '50%',
              border: '1px solid var(--divider)',
              opacity: 0.65,
            }}
            aria-hidden="true"
          />
          <div
            style={{
              position: 'absolute',
              width: '150px',
              height: '150px',
              borderRadius: '50%',
              border: '1px dashed var(--divider)',
              opacity: 0.5,
            }}
            aria-hidden="true"
          />
          <div
            style={{
              position: 'absolute',
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              border: '1px solid var(--divider)',
              opacity: 0.4,
            }}
            aria-hidden="true"
          />

          <div
            style={{
              position: 'relative',
              textAlign: 'center',
              maxWidth: '300px',
              padding: '0 1rem',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                color: 'var(--text-muted)',
                marginBottom: '0.75rem',
              }}
            >
              Baseline State · Monochrome
            </div>
            <p
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                letterSpacing: '0.04em',
                color: 'var(--text-secondary)',
                lineHeight: 1.6,
                margin: 0,
              }}
            >
              Run an analysis to reveal the emotional landscape.
            </p>
          </div>
        </div>
      )}

      {/* Technical corner annotations */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '16px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.5rem',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: 'var(--text-faint)',
          zIndex: 6,
          pointerEvents: 'none',
        }}
        aria-hidden="true"
      >
        EMORA // Affective Field
      </div>

      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          right: '16px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.5rem',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: hasScores ? 'var(--text-secondary)' : 'var(--text-faint)',
          zIndex: 6,
          pointerEvents: 'none',
        }}
        aria-hidden="true"
      >
        {hasScores ? 'Sigmoid Tensor · 6 Dimensions' : 'State: Awaiting Inference'}
      </div>
    </div>
  );
}
