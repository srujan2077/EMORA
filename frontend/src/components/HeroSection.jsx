import React from 'react';
import AffectiveCanvas from './AffectiveCanvas';

export default function HeroSection({ onExploreClick }) {
  return (
    <section
      style={{
        paddingTop: 'clamp(3rem, 7vh, 5.5rem)',
        paddingBottom: 'clamp(2.5rem, 5vh, 4rem)',
        position: 'relative',
        borderBottom: '1px solid var(--divider)',
      }}
    >
      <div className="editorial-container">
        {/* Top telemetry bar */}
        <div
          className="editorial-header-meta"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            borderBottom: '1px solid var(--divider)',
            paddingBottom: '0.875rem',
            marginBottom: 'clamp(2rem, 4vh, 3.5rem)',
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
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>[01/04]</span>
            <span>EMORA // MARK 03</span>
            <span style={{ color: 'var(--divider)' }}>/</span>
            <span>MULTIMODAL INTELLIGENCE</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            <span>3 ENCODERS : 1 FUSED AFFECTIVE TENSOR</span>
            <span style={{ color: 'var(--divider)' }}>|</span>
            <span style={{ color: 'var(--text-primary)' }}>LATENCY: 24ms</span>
          </div>
        </div>

        {/* Hero Grid: Giant Clipped Headline + Editorial Sidecard */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(12, 1fr)',
            gap: 'clamp(1.5rem, 3vw, 3rem)',
            alignItems: 'end',
            marginBottom: 'clamp(2.5rem, 5vh, 4.5rem)',
          }}
        >
          {/* Main Display Headline with Clipped Reveals */}
          <div style={{ gridColumn: 'span 8' }} className="hero-headline-col">
            <h1
              style={{
                margin: 0,
                fontSize: 'clamp(3.5rem, 8.8vw, 8.2rem)',
                fontWeight: 800,
                letterSpacing: '-0.04em',
                lineHeight: 0.88,
                textTransform: 'uppercase',
                color: 'var(--text-primary)',
              }}
            >
              <span className="clip-line" style={{ display: 'block', overflow: 'hidden' }}>
                <span className="clip-text" style={{ display: 'block', animationDelay: '0.05s' }}>
                  EMOTION IS
                </span>
              </span>
              <span className="clip-line" style={{ display: 'block', overflow: 'hidden' }}>
                <span className="clip-text" style={{ display: 'block', animationDelay: '0.18s' }}>
                  MORE THAN
                </span>
              </span>
              <span className="clip-line" style={{ display: 'block', overflow: 'hidden' }}>
                <span className="clip-text" style={{ display: 'block', animationDelay: '0.32s' }}>
                  A LABEL.
                </span>
              </span>
            </h1>
          </div>

          {/* Right Editorial Sidecard */}
          <div
            style={{
              gridColumn: 'span 4',
              borderLeft: '1px solid var(--divider)',
              paddingLeft: 'clamp(1rem, 2vw, 2rem)',
              paddingBottom: '0.5rem',
            }}
            className="hero-sidecard-col"
          >
            <div
              className="text-overline"
              style={{
                marginBottom: '1rem',
                color: 'var(--text-primary)',
                fontWeight: 600,
              }}
            >
              Abstract
            </div>
            <p
              style={{
                fontSize: '0.9375rem',
                lineHeight: 1.7,
                color: 'var(--text-secondary)',
                marginBottom: '1.75rem',
              }}
            >
              Human emotional experience defies discrete classification. EMORA interprets subtle affective cues by jointly fusing deep representations across language, acoustic prosody, and facial micro-dynamics.
            </p>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '0.75rem',
                borderTop: '1px solid var(--divider-light)',
                paddingTop: '1rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.625rem',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                color: 'var(--text-muted)',
              }}
            >
              <div>
                <span style={{ display: 'block', color: 'var(--text-faint)' }}>ENCODERS</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>DeBERTa · W2V · ViT</span>
              </div>
              <div>
                <span style={{ display: 'block', color: 'var(--text-faint)' }}>BENCHMARK</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>CMU-MOSEI SOTA</span>
              </div>
            </div>
          </div>
        </div>

        {/* Live Signal Instrument — AffectiveCanvas */}
        <div style={{ marginTop: '1rem', position: 'relative' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              marginBottom: '0.5rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
            }}
          >
            <span>LIVE INSTRUMENT // MULTIMODAL SIGNAL SCANNER</span>
            <span>INTERACTIVE HARMONIC PROJECTION — HOVER TO MODULATE</span>
          </div>

          <AffectiveCanvas height={180} />
        </div>
      </div>
    </section>
  );
}
