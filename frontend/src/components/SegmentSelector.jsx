import React, { useState, useRef, useEffect } from 'react';

export default function SegmentSelector({
  samples = [],
  selectedSegment,
  onSelectSegment,
  onEvaluateSegment,
  loading,
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const dropdownRef = useRef(null);

  const filteredSamples = samples.filter((id) =>
    id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleRandomPick = () => {
    if (!samples.length) return;
    const randomIndex = Math.floor(Math.random() * samples.length);
    onSelectSegment(samples[randomIndex]);
  };

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setSearchOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div>
      {samples.length > 0 ? (
        <>
          {/* Section label with segment count */}
          <div className="text-overline" style={{ marginBottom: '1.25rem' }}>
            CMU-MOSEI Test Set · {samples.length.toLocaleString()} Aligned Segments Available
          </div>

          {/* Search + controls row */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
              flexWrap: 'wrap',
            }}
          >
            {/* Search input with combobox */}
            <div ref={dropdownRef} style={{ flex: 1, minWidth: '240px', position: 'relative' }}>
              <input
                type="text"
                placeholder="Search benchmark segment ID…"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setSearchOpen(true);
                }}
                onFocus={() => setSearchOpen(true)}
                className="input-editorial"
                style={{ width: '100%' }}
                aria-label="Search CMU-MOSEI segment ID"
                aria-expanded={searchOpen && !!searchTerm}
                role="combobox"
              />

              {/* Autocomplete dropdown */}
              {searchOpen && searchTerm && (
                <div
                  role="listbox"
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    marginTop: '0.25rem',
                    maxHeight: '240px',
                    overflowY: 'auto',
                    background: 'var(--surface)',
                    border: '1px solid var(--divider)',
                    zIndex: 50,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
                  }}
                >
                  {filteredSamples.length === 0 ? (
                    <div
                      style={{
                        padding: '1rem',
                        color: 'var(--text-muted)',
                        textAlign: 'center',
                        fontSize: '0.6875rem',
                      }}
                    >
                      No matching segment ID
                    </div>
                  ) : (
                    filteredSamples.slice(0, 30).map((id) => (
                      <button
                        key={id}
                        role="option"
                        aria-selected={selectedSegment === id}
                        onClick={() => {
                          onSelectSegment(id);
                          setSearchOpen(false);
                          setSearchTerm('');
                        }}
                        style={{
                          display: 'block',
                          width: '100%',
                          textAlign: 'left',
                          padding: '0.5rem 0.75rem',
                          background: selectedSegment === id ? 'rgba(17,17,17,0.05)' : 'transparent',
                          color: selectedSegment === id ? 'var(--text-primary)' : 'var(--text-secondary)',
                          border: 'none',
                          cursor: 'pointer',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.75rem',
                          transition: 'background 0.15s ease',
                        }}
                        onMouseEnter={(e) => {
                          if (selectedSegment !== id) {
                            e.target.style.background = 'rgba(17,17,17,0.03)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (selectedSegment !== id) {
                            e.target.style.background = 'transparent';
                          }
                        }}
                      >
                        {id}
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Random Pick Button */}
            <button
              onClick={handleRandomPick}
              disabled={loading}
              className="btn-ghost"
              aria-label="Select random segment"
            >
              Random
            </button>

            {/* Evaluate Segment Button */}
            {onEvaluateSegment && selectedSegment && (
              <button
                onClick={() => onEvaluateSegment(selectedSegment)}
                disabled={loading}
                className="btn-ghost"
                style={{
                  color: 'var(--text-primary)',
                  borderColor: 'var(--text-primary)',
                }}
              >
                Evaluate Segment
              </button>
            )}
          </div>

          {/* Active segment indicator */}
          {selectedSegment && (
            <div
              style={{
                marginTop: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                color: 'var(--text-faint)',
              }}
            >
              Active: <span style={{ color: 'var(--text-secondary)' }}>{selectedSegment}</span>
            </div>
          )}
        </>
      ) : (
        /* Transparent Editorial Notice when backend Mark 1 CSD dataset is not loaded */
        <div style={{ marginBottom: '1rem' }}>
          <div
            className="text-overline"
            style={{
              marginBottom: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <span>CMU-MOSEI Benchmark · Aligned Dataset Split</span>
            <span style={{ color: 'var(--text-muted)' }}>0 Pre-Aligned Segments Hosted</span>
          </div>

          <div
            style={{
              padding: '1.25rem 1.5rem',
              border: '1px solid var(--divider)',
              background: 'var(--surface)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '0.75rem',
                marginBottom: '0.5rem',
              }}
            >
              <span
                style={{
                  fontSize: '0.8125rem',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  letterSpacing: '-0.01em',
                }}
              >
                Mark 1 Aligned CSD Dataset Unavailable
              </span>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.5625rem',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  padding: '2px 8px',
                  border: '1px solid var(--divider)',
                  color: 'var(--text-muted)',
                }}
              >
                COVAREP Offline
              </span>
            </div>

            <p
              style={{
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.6,
                margin: 0,
                maxWidth: '680px',
              }}
            >
              The legacy Mark 1 dataset endpoint (<code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem' }}>/samples</code>) requires <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6875rem' }}>COVAREP_aligned.csd</code> which is not hosted on the server. Full multimodal inference using the Mark 2 and Mark 3 transformer pipelines (DeBERTa-v3, Wav2Vec2, and ViT) is fully operational via the Language, Audio, and Video inputs below.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
