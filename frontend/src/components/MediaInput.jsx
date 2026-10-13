import React, { useRef, useState } from 'react';

export default function MediaInput({
  type,       // 'audio' | 'video'
  file,
  setFile,
  label,
  modelName,
  accept,
  sampleUrl,
  sampleName,
  sampleMime,
}) {
  const [mediaUrl, setMediaUrl] = useState(null);
  const [duration, setDuration] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const processFile = (f) => {
    if (f) {
      setFile(f);
      const url = URL.createObjectURL(f);
      setMediaUrl(url);
    }
  };

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    processFile(f);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer?.files?.[0];
    if (f) processFile(f);
  };

  const handleRemove = () => {
    setFile(null);
    setMediaUrl(null);
    setDuration(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch(sampleUrl);
      if (res.ok) {
        const blob = await res.blob();
        const f = new File([blob], sampleName, { type: sampleMime });
        processFile(f);
      }
    } catch (err) {
      console.warn(`Could not load sample ${type}:`, err);
    }
  };

  const handleLoadedMetadata = (e) => {
    const dur = e.currentTarget.duration;
    if (dur && !isNaN(dur)) {
      setDuration(dur.toFixed(1));
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return '';
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div>
      {/* Section heading */}
      <h3
        style={{
          fontSize: 'clamp(1rem, 1.8vw, 1.375rem)',
          fontWeight: 600,
          letterSpacing: '-0.015em',
          color: 'var(--text-primary)',
          marginBottom: '0.3rem',
        }}
      >
        {label}
      </h3>
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.5625rem',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'var(--text-faint)',
          display: 'block',
          marginBottom: '1.25rem',
        }}
      >
        {modelName}
      </span>

      {/* Content */}
      {mediaUrl ? (
        /* File loaded state */
        <div>
          {type === 'audio' && (
            <audio
              controls
              src={mediaUrl}
              onLoadedMetadata={handleLoadedMetadata}
              style={{
                width: '100%',
                height: '36px',
                marginBottom: '0.75rem',
                filter: 'invert(1) hue-rotate(180deg)',
                opacity: 0.7,
                transition: 'opacity 0.2s ease',
              }}
              onMouseEnter={(e) => (e.target.style.opacity = '1')}
              onMouseLeave={(e) => (e.target.style.opacity = '0.7')}
            />
          )}

          {type === 'video' && (
            <video
              controls
              src={mediaUrl}
              onLoadedMetadata={handleLoadedMetadata}
              style={{
                width: '100%',
                height: '180px',
                objectFit: 'contain',
                background: '#000',
                marginBottom: '0.75rem',
                transition: 'transform 0.2s ease',
              }}
              onMouseEnter={(e) => (e.target.style.transform = 'scale(1.01)')}
              onMouseLeave={(e) => (e.target.style.transform = 'scale(1)')}
            />
          )}

          {/* File info */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.625rem',
              color: 'var(--text-muted)',
            }}
          >
            <span>
              {file?.name}
              {duration && ` · ${duration}s`}
              {file?.size && ` · ${formatSize(file.size)}`}
            </span>
            <button onClick={handleRemove} className="btn-text" style={{ color: '#f87171', fontSize: '0.625rem' }}>
              Remove
            </button>
          </div>
        </div>
      ) : (
        /* Upload state — subtle, integrated */
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className="upload-area-editorial"
          style={{
            padding: '2rem 1.5rem',
            borderLeft: isDragging ? '2px solid var(--accent)' : '2px solid var(--border-subtle)',
            cursor: 'pointer',
            background: isDragging ? 'var(--accent-dim)' : 'transparent',
          }}
          onMouseEnter={(e) => {
            if (!isDragging) {
              e.currentTarget.style.borderLeftColor = 'var(--border-medium)';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.01)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isDragging) {
              e.currentTarget.style.borderLeftColor = 'var(--border-subtle)';
              e.currentTarget.style.background = 'transparent';
            }
          }}
        >
          <div
            className="icon-container"
            style={{
              fontSize: '0.8125rem',
              color: 'var(--text-secondary)',
              marginBottom: '0.375rem',
            }}
          >
            + Add {type}
          </div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              color: 'var(--text-faint)',
              letterSpacing: '0.02em',
            }}
          >
            {type === 'audio' ? '.wav, .mp3, .ogg · 16kHz mono recommended' : '.mp4, .webm, .mov · facial expression cues'}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept={accept}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      )}

      {/* Load sample link */}
      {!mediaUrl && (
        <div style={{ marginTop: '0.5rem' }}>
          <button onClick={handleLoadSample} className="btn-text" style={{ fontSize: '0.625rem' }}>
            Load sample {type}
          </button>
        </div>
      )}
    </div>
  );
}
