import React, { useState, useEffect } from 'react';

export default function PageEntrance({ onComplete }) {
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(true);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    // Check if user prefers reduced motion or already loaded in session
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      setVisible(false);
      onComplete?.();
      return;
    }

    // Fast, crisp progress counter: 0 -> 100% in 500ms
    const start = performance.now();
    const duration = 520;

    const interval = setInterval(() => {
      const elapsed = performance.now() - start;
      const p = Math.min(100, Math.round((elapsed / duration) * 100));
      setProgress(p);

      if (p >= 100) {
        clearInterval(interval);
        setRevealed(true);
        setTimeout(() => {
          setVisible(false);
          onComplete?.();
        }, 350);
      }
    }, 20);

    return () => clearInterval(interval);
  }, [onComplete]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: '#111111',
        color: '#F4F2EC',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: 'clamp(2rem, 5vw, 4rem)',
        pointerEvents: revealed ? 'none' : 'auto',
        transform: revealed ? 'translateY(-100%)' : 'translateY(0)',
        transition: 'transform 0.65s cubic-bezier(0.77, 0, 0.175, 1)',
        willChange: 'transform',
      }}
      aria-hidden="true"
    >
      {/* Top row metadata */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.625rem',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: 'rgba(244, 242, 236, 0.6)',
        }}
      >
        <span>EMORA · RESEARCH INSTRUMENTATION</span>
        <span>MARK 03 // MULTIMODAL INTELLIGENCE</span>
      </div>

      {/* Center oversized wordmark reveal */}
      <div style={{ overflow: 'hidden' }}>
        <h1
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: 'clamp(3rem, 12vw, 10rem)',
            fontWeight: 800,
            letterSpacing: '-0.04em',
            lineHeight: 0.88,
            margin: 0,
            textTransform: 'uppercase',
            color: '#F4F2EC',
            transform: revealed ? 'translateY(-30%)' : 'translateY(0)',
            transition: 'transform 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          EMORA
        </h1>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 'clamp(0.75rem, 1.5vw, 1.125rem)',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: 'rgba(244, 242, 236, 0.5)',
            marginTop: '0.75rem',
          }}
        >
          Affective Tensor Synthesis
        </div>
      </div>

      {/* Bottom telemetry and rapid percentage ticker */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          borderTop: '1px solid rgba(244, 242, 236, 0.15)',
          paddingTop: '1.25rem',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.6875rem',
          letterSpacing: '0.08em',
        }}
      >
        <span style={{ color: 'rgba(244, 242, 236, 0.6)' }}>
          INITIALIZING ENCODERS: DeBERTa · Wav2Vec2 · ViT
        </span>
        <span
          style={{
            fontSize: 'clamp(1.5rem, 3vw, 2.5rem)',
            fontWeight: 700,
            color: '#F4F2EC',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {String(progress).padStart(2, '0')}%
        </span>
      </div>
    </div>
  );
}
