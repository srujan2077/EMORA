import React from 'react';

export default function Header({
  backendHealth,
  activeTab,
  setActiveTab,
  selectedModel,
  setSelectedModel,
}) {
  const isOnline = backendHealth?.status === 'ok';
  const device = backendHealth?.device ? backendHealth.device.toUpperCase() : 'CPU';

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'rgba(0, 0, 0, 0.88)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--divider)',
      }}
    >
      <div
        className="editorial-container"
        style={{
          height: '3.75rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Left: Brand Identity */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.875rem' }}>
          <a
            href="/"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('analysis');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            style={{
              fontWeight: 800,
              fontSize: '1.0625rem',
              letterSpacing: '0.12em',
              color: 'var(--text-primary)',
              textTransform: 'uppercase',
              textDecoration: 'none',
              cursor: 'pointer',
            }}
          >
            EMORA
          </a>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.14em',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
            }}
          >
            MK-03 // SOTA
          </span>
        </div>

        {/* Center: Navigation Tabs */}
        <nav
          style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          role="navigation"
          aria-label="Main navigation"
        >
          {[
            { id: 'analysis', label: 'Analysis Console' },
            { id: 'architecture', label: 'Architecture' },
            { id: 'benchmarks', label: 'Benchmarks' },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                aria-current={isActive ? 'page' : undefined}
                className="nav-btn"
                style={{
                  background: isActive ? 'var(--text-primary)' : 'transparent',
                  border: 'none',
                  padding: '0.45rem 0.95rem',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6875rem',
                  fontWeight: isActive ? 600 : 400,
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                  color: isActive ? 'var(--canvas)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                  position: 'relative',
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Right: Telemetry & Model Quick-Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          {/* Model toggle */}
          {setSelectedModel && (
            <div
              style={{
                display: 'inline-flex',
                border: '1px solid var(--divider)',
                background: 'var(--surface)',
              }}
            >
              <button
                type="button"
                onClick={() => setSelectedModel('mark3')}
                style={{
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.5625rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: selectedModel === 'mark3' ? 600 : 400,
                  border: 'none',
                  background: selectedModel === 'mark3' ? 'var(--text-primary)' : 'transparent',
                  color: selectedModel === 'mark3' ? 'var(--canvas)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  letterSpacing: '0.02em',
                }}
                title="Mark 3: 5-Seed Multimodal Ensemble with Causal Windowing"
              >
                MK-3
              </button>
              <button
                type="button"
                onClick={() => setSelectedModel('mark2')}
                style={{
                  padding: '0.25rem 0.6rem',
                  fontSize: '0.5625rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: selectedModel === 'mark2' ? 600 : 400,
                  border: 'none',
                  borderLeft: '1px solid var(--divider)',
                  background: selectedModel === 'mark2' ? 'var(--text-primary)' : 'transparent',
                  color: selectedModel === 'mark2' ? 'var(--canvas)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  letterSpacing: '0.02em',
                }}
                title="Mark 2: Single Multimodal Transformer Baseline"
              >
                MK-2
              </button>
            </div>
          )}

          {/* API Status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
            }}
            role="status"
            aria-live="polite"
            aria-label={isOnline ? `API online on ${device}` : 'API offline'}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: isOnline ? 'var(--text-primary)' : 'transparent',
                border: '1px solid var(--text-primary)',
                display: 'inline-block',
                flexShrink: 0,
              }}
              aria-hidden="true"
            />
            <span>
              {isOnline ? `API [${device}]` : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
