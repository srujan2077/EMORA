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
      {/* Section label */}
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
        CMU-MOSEI Test Set · {samples.length.toLocaleString()} segments
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
        {/* Search input */}
        <div ref={dropdownRef} style={{ flex: 1, minWidth: '240px', position: 'relative' }}>
          <input
            type="text"
            placeholder="Search segment ID…"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setSearchOpen(true);
            }}
            onFocus={() => setSearchOpen(true)}
            className="input-editorial"
            style={{ width: '100%' }}
          />

          {/* Autocomplete dropdown */}
          {searchOpen && searchTerm && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                marginTop: '0.25rem',
                maxHeight: '240px',
                overflowY: 'auto',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-subtle)',
                zIndex: 50,
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
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
                  No matching segment
                </div>
              ) : (
                filteredSamples.slice(0, 30).map((id) => (
                  <button
                    key={id}
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
                      background: selectedSegment === id ? 'var(--accent-dim)' : 'transparent',
                      color: selectedSegment === id ? 'var(--accent-bright)' : 'var(--text-secondary)',
                      border: 'none',
                      cursor: 'pointer',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.75rem',
                      transition: 'background 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      if (selectedSegment !== id) {
                        e.target.style.background = 'rgba(255,255,255,0.03)';
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

        {/* Random */}
        <button
          onClick={handleRandomPick}
          disabled={loading || !samples.length}
          className="btn-ghost"
        >
          Random
        </button>

        {/* Evaluate Segment */}
        {onEvaluateSegment && selectedSegment && (
          <button
            onClick={() => onEvaluateSegment(selectedSegment)}
            disabled={loading}
            className="btn-ghost"
            style={{
              color: 'var(--accent)',
              borderColor: 'rgba(120, 119, 198, 0.2)',
            }}
          >
            Evaluate Segment
          </button>
        )}
      </div>

      {/* Active segment ID — subtle display */}
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
    </div>
  );
}
