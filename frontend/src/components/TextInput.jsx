import React from 'react';

export default function TextInput({ textValue, setTextValue }) {
  const charCount = textValue.length;
  const wordCount = textValue.trim() ? textValue.trim().split(/\s+/).length : 0;

  const handleLoadSample = () => {
    setTextValue(
      'I felt absolutely thrilled and overjoyed by the unexpected outcome, but also somewhat cautious about what might happen next.'
    );
  };

  return (
    <div>
      {/* Section label */}
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          marginBottom: '1.5rem',
        }}
      >
        <div>
          <h2
            style={{
              fontSize: 'clamp(1.25rem, 2.5vw, 2rem)',
              fontWeight: 600,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              marginBottom: '0.35rem',
            }}
          >
            Language
          </h2>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'var(--text-faint)',
            }}
          >
            DeBERTa-v3-base
          </span>
        </div>
      </div>

      {/* Large editorial textarea */}
      <textarea
        rows={4}
        value={textValue}
        onChange={(e) => setTextValue(e.target.value)}
        placeholder="Enter transcript or utterance text…"
        className="textarea-editorial"
      />

      {/* Metadata + actions row */}
      <div
        style={{
          marginTop: '0.75rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.04em',
          }}
        >
          {charCount} characters · {wordCount} words
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button onClick={handleLoadSample} className="btn-text">
            Load sample
          </button>
          {charCount > 0 && (
            <button onClick={() => setTextValue('')} className="btn-text" style={{ color: '#f87171' }}>
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
