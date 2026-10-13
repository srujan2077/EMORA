import React from 'react';

export default function HeroSection() {
  return (
    <section
      style={{
        padding: 'clamp(5rem, 16vh, 10rem) 0 clamp(3rem, 8vh, 5rem)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Subtle gradient accent — barely visible */}
      <div
        style={{
          position: 'absolute',
          top: '-30%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '600px',
          height: '400px',
          background: 'radial-gradient(ellipse, rgba(120, 119, 198, 0.04) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      <div className="editorial-container">
        <div className="animate-fade-up" style={{ maxWidth: '720px' }}>
          {/* Overline */}
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
              marginBottom: '1.5rem',
            }}
          >
            Multimodal Emotion Intelligence
          </div>

          {/* Display Title */}
          <h1 className="text-display" style={{ marginBottom: '2rem' }}>
            <span style={{ display: 'block', color: 'var(--text-primary)' }}>EMORA</span>
            <span
              style={{
                display: 'block',
                color: 'var(--text-muted)',
                fontSize: '0.45em',
                fontWeight: 400,
                letterSpacing: '0.08em',
                marginTop: '0.5rem',
              }}
            >
              MARK 02
            </span>
          </h1>

          {/* Description */}
          <p
            style={{
              fontSize: 'clamp(0.9rem, 1.3vw, 1.125rem)',
              lineHeight: 1.8,
              color: 'var(--text-secondary)',
              maxWidth: '480px',
            }}
          >
            Understand emotion through language, voice,
            and visual expression. Three transformer encoders.
            One fused prediction.
          </p>
        </div>
      </div>
    </section>
  );
}
