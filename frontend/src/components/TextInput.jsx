import React from 'react';

const SAMPLES = [
  {
    label: 'Joy & Caution',
    text: 'I felt absolutely thrilled and overjoyed by the unexpected outcome, but also somewhat cautious about what might happen next.',
  },
  {
    label: 'Frustration',
    text: 'I cannot believe this is happening again after we specifically spent months agreeing on the protocol and safety parameters.',
  },
  {
    label: 'Melancholy',
    text: 'There was a heavy, quiet stillness across the hall as the final archival results were read aloud to the team.',
  },
];

export default function TextInput({ textValue, setTextValue }) {
  const charCount = textValue.length;
  const wordCount = textValue.trim() ? textValue.trim().split(/\s+/).length : 0;
  const estimatedTokens = Math.round(wordCount * 1.33);

  return (
    <div
      style={{
        border: '1px solid var(--divider)',
        background: 'var(--surface)',
        padding: 'clamp(1.25rem, 2.5vw, 2rem)',
        position: 'relative',
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          borderBottom: '1px solid var(--divider-light)',
          paddingBottom: '0.75rem',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              fontWeight: 600,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'var(--text-primary)',
            }}
          >
            [MODALITY 01 // LANGUAGE CONSOLE]
          </span>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.08em',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
            }}
          >
            DeBERTa-v3-base · 768-D
          </span>
        </div>

        {/* Sample chips */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              color: 'var(--text-faint)',
              textTransform: 'uppercase',
            }}
          >
            Sample:
          </span>
          {SAMPLES.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => setTextValue(sample.text)}
              style={{
                background: textValue === sample.text ? 'var(--text-primary)' : 'transparent',
                color: textValue === sample.text ? 'var(--canvas)' : 'var(--text-secondary)',
                border: '1px solid var(--divider)',
                padding: '0.2rem 0.55rem',
                fontSize: '0.5625rem',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>

      {/* Large editorial textarea */}
      <textarea
        rows={4}
        value={textValue}
        onChange={(e) => setTextValue(e.target.value)}
        placeholder="Enter transcript or utterance text to decode affective semantics…"
        style={{
          width: '100%',
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-sans)',
          fontSize: 'clamp(1rem, 1.4vw, 1.1875rem)',
          lineHeight: 1.8,
          resize: 'vertical',
          minHeight: '110px',
        }}
        aria-label="Transcript text input for emotion analysis"
      />

      {/* Footer telemetry */}
      <div
        style={{
          marginTop: '1rem',
          borderTop: '1px solid var(--divider-light)',
          paddingTop: '0.75rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.625rem',
          letterSpacing: '0.06em',
          color: 'var(--text-muted)',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span>{charCount} CHARACTERS</span>
          <span style={{ color: 'var(--divider)' }}>/</span>
          <span>{wordCount} WORDS</span>
          <span style={{ color: 'var(--divider)' }}>/</span>
          <span>~{estimatedTokens} SUBWORD TOKENS</span>
        </div>

        {charCount > 0 && (
          <button
            type="button"
            onClick={() => setTextValue('')}
            className="btn-text"
            style={{ fontSize: '0.625rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}
          >
            [Clear Buffer]
          </button>
        )}
      </div>
    </div>
  );
}
