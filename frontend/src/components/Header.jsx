import React from 'react';

export default function Header({ backendHealth, activeTab, setActiveTab }) {
  const isOnline = backendHealth?.status === 'ok';

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'rgba(5, 5, 6, 0.85)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border-faint)',
      }}
    >
      <div
        className="editorial-container"
        style={{
          height: '3.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Left: Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <span
            style={{
              fontWeight: 700,
              fontSize: '0.875rem',
              letterSpacing: '0.12em',
              color: 'var(--text-primary)',
            }}
          >
            EMORA
          </span>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              letterSpacing: '0.1em',
              color: 'var(--text-muted)',
            }}
          >
            MARK 02
          </span>
        </div>

        {/* Center: Navigation */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          {['analysis', 'architecture', 'benchmarks'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                background: 'none',
                border: 'none',
                padding: '0.4rem 0.875rem',
                fontFamily: 'var(--font-sans)',
                fontSize: '0.6875rem',
                fontWeight: 500,
                letterSpacing: '0.04em',
                textTransform: 'capitalize',
                color: activeTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
                cursor: 'pointer',
                transition: 'color 0.2s ease',
                position: 'relative',
              }}
              onMouseEnter={(e) => {
                if (activeTab !== tab) e.target.style.color = 'var(--text-secondary)';
              }}
              onMouseLeave={(e) => {
                if (activeTab !== tab) e.target.style.color = 'var(--text-muted)';
              }}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              {activeTab === tab && (
                <span
                  style={{
                    position: 'absolute',
                    bottom: '-0.5rem',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '16px',
                    height: '1px',
                    background: 'var(--accent)',
                  }}
                />
              )}
            </button>
          ))}
        </nav>

        {/* Right: API Status — tiny dot only */}
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
        >
          <span
            style={{
              width: '5px',
              height: '5px',
              borderRadius: '50%',
              background: isOnline ? '#4ade80' : '#f87171',
              display: 'inline-block',
              flexShrink: 0,
            }}
          />
          <span style={{ color: isOnline ? 'var(--text-muted)' : '#f87171' }}>
            {isOnline ? 'API' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
