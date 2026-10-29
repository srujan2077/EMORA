import React, { useRef, useState, useEffect } from 'react';

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

  useEffect(() => {
    if (!file) {
      setMediaUrl(null);
      setDuration(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setMediaUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
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
    if (f) setFile(f);
  };

  const handleRemove = () => {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch(sampleUrl);
      if (res.ok) {
        const blob = await res.blob();
        const f = new File([blob], sampleName, { type: sampleMime });
        setFile(f);
      }
    } catch (err) {
      console.warn(`Could not load sample ${type}:`, err);
    }
  };

  const handleLoadedMetadata = (e) => {
    const dur = e.currentTarget.duration;
    if (dur && !isNaN(dur)) {
      setDuration(dur.toFixed(2));
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return '';
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const channelNumber = type === 'audio' ? '02' : '03';

  return (
    <div
      style={{
        border: '1px solid var(--divider)',
        background: 'var(--surface)',
        padding: 'clamp(1.25rem, 2vw, 1.75rem)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minHeight: '260px',
        position: 'relative',
        transition: 'border-color 0.2s ease',
      }}
      className={isDragging ? 'drag-active' : ''}
    >
      {/* Modality Header */}
      <div>
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            borderBottom: '1px solid var(--divider-light)',
            paddingBottom: '0.625rem',
            marginBottom: '1rem',
          }}
        >
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
            [MODALITY {channelNumber} // {label.toUpperCase()}]
          </span>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.5625rem',
              letterSpacing: '0.06em',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
            }}
          >
            {modelName}
          </span>
        </div>

        {/* Content Area */}
        {mediaUrl ? (
          /* File Loaded State */
          <div>
            {type === 'audio' && (
              <div
                style={{
                  background: 'var(--canvas)',
                  border: '1px solid var(--divider)',
                  padding: '1rem',
                  marginBottom: '1rem',
                }}
              >
                {/* Simulated Audio Waveform Graphic */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                    height: '38px',
                    marginBottom: '0.75rem',
                  }}
                  aria-hidden="true"
                >
                  {[30, 45, 60, 25, 80, 95, 70, 40, 85, 100, 75, 50, 35, 65, 90, 40, 20, 55, 75, 90, 60, 30, 70, 85, 45, 25, 60, 80, 50, 35].map((val, idx) => (
                    <div
                      key={idx}
                      style={{
                        flex: 1,
                        height: `${val}%`,
                        background: 'var(--text-primary)',
                        opacity: 0.35 + (val / 100) * 0.5,
                        borderRadius: '1px',
                      }}
                    />
                  ))}
                </div>

                <audio
                  controls
                  src={mediaUrl}
                  onLoadedMetadata={handleLoadedMetadata}
                  style={{ width: '100%', height: '32px' }}
                  aria-label={`${label} file audio player`}
                />
              </div>
            )}

            {type === 'video' && (
              <div
                style={{
                  position: 'relative',
                  background: '#111111',
                  border: '1px solid var(--divider)',
                  marginBottom: '1rem',
                  overflow: 'hidden',
                }}
              >
                <video
                  controls
                  src={mediaUrl}
                  onLoadedMetadata={handleLoadedMetadata}
                  style={{
                    width: '100%',
                    height: '160px',
                    objectFit: 'contain',
                    display: 'block',
                  }}
                  aria-label={`${label} file video player`}
                />
              </div>
            )}

            {/* File telemetry */}
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.625rem',
                letterSpacing: '0.04em',
                color: 'var(--text-secondary)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderTop: '1px solid var(--divider-light)',
                paddingTop: '0.625rem',
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '75%' }}>
                {file?.name} {duration && `(${duration}s)`} {file?.size && `· ${formatSize(file.size)}`}
              </span>
              <button
                type="button"
                onClick={handleRemove}
                className="btn-text"
                style={{ fontSize: '0.5625rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}
                aria-label={`Remove ${label} file`}
              >
                [Remove]
              </button>
            </div>
          </div>
        ) : (
          /* Empty / Drop Zone */
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className="upload-dropzone"
            role="button"
            tabIndex={0}
            aria-label={`Upload ${type} file for emotional analysis`}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            style={{
              padding: '2.25rem 1.25rem',
              border: isDragging ? '1px solid var(--text-primary)' : '1px dashed var(--divider)',
              background: isDragging ? 'rgba(17, 17, 17, 0.04)' : 'var(--canvas)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                letterSpacing: '0.04em',
                marginBottom: '0.35rem',
              }}
            >
              + ATTACH {type.toUpperCase()} SIGNAL
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.5625rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.04em',
              }}
            >
              {type === 'audio' ? '.WAV / .MP3 (16 kHz MONO)' : '.MP4 / .WEBM (FACIAL CUES)'}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={accept}
              onChange={handleFileChange}
              style={{ display: 'none' }}
              aria-hidden="true"
            />
          </div>
        )}
      </div>

      {/* Preset sample load link */}
      {!mediaUrl && (
        <div style={{ marginTop: '0.75rem', textAlign: 'right' }}>
          <button
            type="button"
            onClick={handleLoadSample}
            className="btn-text"
            style={{ fontSize: '0.5625rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}
          >
            [Load Preset {type}]
          </button>
        </div>
      )}
    </div>
  );
}
