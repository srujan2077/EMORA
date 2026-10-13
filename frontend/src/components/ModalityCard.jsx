import React, { useRef, useState } from 'react';
import {
  FileText,
  Volume2,
  Video,
  UploadCloud,
  Check,
  RotateCcw,
  Sparkles,
  Clock,
  Trash2,
} from 'lucide-react';

/* ============================================================
   TEXT MODALITY CARD
   ============================================================ */
export function TextModalityCard({
  textValue,
  setTextValue,
  onAnalyze,
  loading,
  processingStatus,
  selectedSegment,
}) {
  const charCount = textValue.length;
  const wordCount = textValue.trim() ? textValue.trim().split(/\s+/).length : 0;

  const handleClear = () => setTextValue('');

  const handleLoadSampleText = () => {
    setTextValue(
      `I felt absolutely thrilled and overjoyed by the unexpected outcome, but also somewhat cautious about what might happen next.`
    );
  };

  const isProcessed = processingStatus === 'completed' && Boolean(textValue.trim());

  return (
    <div className="research-card p-4 sm:p-5 flex flex-col justify-between h-full">
      <div>
        {/* Card Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-white/5 border border-[var(--border-subtle)] flex items-center justify-center text-indigo-400">
              <FileText className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight text-white font-sans">
                TEXT
              </h3>
              <p className="text-[11px] text-[var(--text-muted)]">Language & semantics</p>
            </div>
          </div>

          <span className="text-[11px] font-mono text-[var(--text-muted)]">
            {loading ? (
              <span className="text-amber-400">Analyzing...</span>
            ) : isProcessed ? (
              <span className="text-emerald-400 flex items-center gap-1">
                <Check className="w-3 h-3" /> Processed
              </span>
            ) : (
              <span>Ready</span>
            )}
          </span>
        </div>

        {/* Text Area */}
        <div className="relative">
          <textarea
            rows={5}
            value={textValue}
            onChange={(e) => setTextValue(e.target.value)}
            placeholder="Type utterance transcript, or paste speech text for emotion analysis..."
            className="w-full bg-[var(--bg-input)] border border-[var(--border-subtle)] focus:border-[var(--border-focus)] rounded-lg p-3 text-xs text-white placeholder-[var(--text-muted)] focus:outline-none resize-none transition-colors font-sans leading-relaxed"
          />
        </div>

        {/* Text Metrics & Subtle Actions */}
        <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] font-mono mt-2">
          <span>
            {charCount} characters · {wordCount} words
          </span>
          <div className="flex items-center gap-2.5">
            <button
              onClick={handleLoadSampleText}
              className="text-[11px] text-[var(--text-secondary)] hover:text-white transition-colors"
            >
              Sample text
            </button>
            {charCount > 0 && (
              <button
                onClick={handleClear}
                className="text-[11px] text-[var(--text-muted)] hover:text-rose-400 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Card Footer: Model Spec & Modality Action */}
      <div className="mt-4 pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between gap-2">
        <span className="text-[11px] font-mono text-[var(--text-secondary)]">
          DeBERTa-v3-base
        </span>
        <button
          onClick={onAnalyze}
          disabled={loading || !textValue.trim()}
          className="btn-secondary-action text-xs py-1 px-2.5"
          title="Analyze text modality independently"
        >
          <Sparkles className="w-3 h-3 text-indigo-400" />
          <span>Analyze Text</span>
        </button>
      </div>
    </div>
  );
}

/* ============================================================
   AUDIO MODALITY CARD
   ============================================================ */
export function AudioModalityCard({
  audioFile,
  setAudioFile,
  onAnalyze,
  loading,
  processingStatus,
}) {
  const [duration, setDuration] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleAudioLoadedMetadata = (e) => {
    const dur = e.currentTarget.duration;
    if (dur && !isNaN(dur)) {
      setDuration(dur.toFixed(2));
    }
  };

  const processAudio = (file) => {
    if (file) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    processAudio(file);
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
    const file = e.dataTransfer?.files?.[0];
    if (file && (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|ogg|flac|m4a)$/i))) {
      processAudio(file);
    }
  };

  const handleRemove = () => {
    setAudioFile(null);
    setAudioUrl(null);
    setDuration(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch('/samples/sample_audio.wav');
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], 'sample_audio.wav', { type: 'audio/wav' });
        processAudio(file);
      }
    } catch (err) {
      console.warn('Could not load sample audio:', err);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const isProcessed = processingStatus === 'completed' && Boolean(audioFile);

  return (
    <div className="research-card p-4 sm:p-5 flex flex-col justify-between h-full">
      <div>
        {/* Card Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-white/5 border border-[var(--border-subtle)] flex items-center justify-center text-sky-400">
              <Volume2 className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight text-white font-sans">
                AUDIO
              </h3>
              <p className="text-[11px] text-[var(--text-muted)]">Neural acoustic features</p>
            </div>
          </div>

          <span className="text-[11px] font-mono text-[var(--text-muted)]">
            {loading ? (
              <span className="text-amber-400">Analyzing...</span>
            ) : isProcessed ? (
              <span className="text-emerald-400 flex items-center gap-1">
                <Check className="w-3 h-3" /> Processed
              </span>
            ) : (
              <span>Ready</span>
            )}
          </span>
        </div>

        {/* Content Area: Audio Player or Upload Box */}
        {audioUrl ? (
          <div className="bg-[var(--bg-input)] p-3 rounded-lg border border-[var(--border-subtle)] space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-white truncate max-w-[170px]" title={audioFile?.name}>
                {audioFile?.name}
              </span>
              <span className="text-[var(--text-muted)] text-[11px]">
                {formatFileSize(audioFile?.size)}
              </span>
            </div>

            <audio
              controls
              src={audioUrl}
              onLoadedMetadata={handleAudioLoadedMetadata}
              className="w-full h-8"
            />

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] font-mono pt-1">
              <span>{duration ? `${duration}s duration` : 'Audio track loaded'}</span>
              <button
                onClick={handleRemove}
                className="text-rose-400 hover:text-rose-300 flex items-center gap-1 text-[11px]"
              >
                <Trash2 className="w-3 h-3" />
                <span>Remove</span>
              </button>
            </div>
          </div>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
              isDragging
                ? 'border-indigo-400 bg-indigo-500/10'
                : 'border-[var(--border-subtle)] hover:border-[var(--border-default)] bg-[var(--bg-input)]'
            }`}
          >
            <UploadCloud className="w-6 h-6 text-[var(--text-muted)] mx-auto mb-1.5" />
            <p className="text-xs text-[var(--text-secondary)] font-medium">
              Upload or drag audio
            </p>
            <p className="text-[10px] text-[var(--text-muted)] font-mono mt-0.5">
              .wav, .mp3, .ogg (16kHz mono recommended)
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        )}

        {/* Subtle Sample Loader */}
        {!audioUrl && (
          <div className="flex justify-end mt-2">
            <button
              onClick={handleLoadSample}
              className="text-[11px] font-mono text-[var(--text-secondary)] hover:text-white transition-colors"
            >
              Load sample audio
            </button>
          </div>
        )}
      </div>

      {/* Card Footer: Model Spec & Modality Action */}
      <div className="mt-4 pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between gap-2">
        <span className="text-[11px] font-mono text-[var(--text-secondary)]">
          Wav2Vec2-base-960h
        </span>
        <button
          onClick={onAnalyze}
          disabled={loading || !audioFile}
          className="btn-secondary-action text-xs py-1 px-2.5"
          title="Analyze audio modality independently"
        >
          <Sparkles className="w-3 h-3 text-sky-400" />
          <span>Analyze Audio</span>
        </button>
      </div>
    </div>
  );
}

/* ============================================================
   VIDEO MODALITY CARD
   ============================================================ */
export function VideoModalityCard({
  videoFile,
  setVideoFile,
  onAnalyze,
  loading,
  processingStatus,
}) {
  const [duration, setDuration] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleVideoLoadedMetadata = (e) => {
    const dur = e.currentTarget.duration;
    if (dur && !isNaN(dur)) {
      setDuration(dur.toFixed(2));
    }
  };

  const processVideo = (file) => {
    if (file) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    processVideo(file);
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
    const file = e.dataTransfer?.files?.[0];
    if (file && (file.type.startsWith('video/') || file.name.match(/\.(mp4|webm|mov|mkv)$/i))) {
      processVideo(file);
    }
  };

  const handleRemove = () => {
    setVideoFile(null);
    setVideoUrl(null);
    setDuration(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch('/samples/sample_video.mp4');
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], 'sample_video.mp4', { type: 'video/mp4' });
        processVideo(file);
      }
    } catch (err) {
      console.warn('Could not load sample video:', err);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const isProcessed = processingStatus === 'completed' && Boolean(videoFile);

  return (
    <div className="research-card p-4 sm:p-5 flex flex-col justify-between h-full">
      <div>
        {/* Card Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-white/5 border border-[var(--border-subtle)] flex items-center justify-center text-purple-400">
              <Video className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight text-white font-sans">
                VIDEO
              </h3>
              <p className="text-[11px] text-[var(--text-muted)]">Visual frame analysis</p>
            </div>
          </div>

          <span className="text-[11px] font-mono text-[var(--text-muted)]">
            {loading ? (
              <span className="text-amber-400">Analyzing...</span>
            ) : isProcessed ? (
              <span className="text-emerald-400 flex items-center gap-1">
                <Check className="w-3 h-3" /> Processed
              </span>
            ) : (
              <span>Ready</span>
            )}
          </span>
        </div>

        {/* Content Area: Video Preview or Upload Box */}
        {videoUrl ? (
          <div className="bg-[var(--bg-input)] p-2.5 rounded-lg border border-[var(--border-subtle)] space-y-2">
            <video
              controls
              src={videoUrl}
              onLoadedMetadata={handleVideoLoadedMetadata}
              className="w-full h-28 rounded bg-black object-contain"
            />

            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-white truncate max-w-[170px]" title={videoFile?.name}>
                {videoFile?.name}
              </span>
              <span className="text-[var(--text-muted)] text-[11px]">
                {formatFileSize(videoFile?.size)}
              </span>
            </div>

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] font-mono pt-0.5">
              <span>{duration ? `${duration}s duration` : 'Video loaded'}</span>
              <button
                onClick={handleRemove}
                className="text-rose-400 hover:text-rose-300 flex items-center gap-1 text-[11px]"
              >
                <Trash2 className="w-3 h-3" />
                <span>Remove</span>
              </button>
            </div>
          </div>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
              isDragging
                ? 'border-indigo-400 bg-indigo-500/10'
                : 'border-[var(--border-subtle)] hover:border-[var(--border-default)] bg-[var(--bg-input)]'
            }`}
          >
            <UploadCloud className="w-6 h-6 text-[var(--text-muted)] mx-auto mb-1.5" />
            <p className="text-xs text-[var(--text-secondary)] font-medium">
              Upload or drag video
            </p>
            <p className="text-[10px] text-[var(--text-muted)] font-mono mt-0.5">
              .mp4, .webm, .mov (facial expression cues)
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        )}

        {/* Subtle Sample Loader */}
        {!videoUrl && (
          <div className="flex justify-end mt-2">
            <button
              onClick={handleLoadSample}
              className="text-[11px] font-mono text-[var(--text-secondary)] hover:text-white transition-colors"
            >
              Load sample video
            </button>
          </div>
        )}
      </div>

      {/* Card Footer: Model Spec & Modality Action */}
      <div className="mt-4 pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between gap-2">
        <span className="text-[11px] font-mono text-[var(--text-secondary)]">
          ViT-base-patch16-224-in21k
        </span>
        <button
          onClick={onAnalyze}
          disabled={loading || !videoFile}
          className="btn-secondary-action text-xs py-1 px-2.5"
          title="Analyze video modality independently"
        >
          <Sparkles className="w-3 h-3 text-purple-400" />
          <span>Analyze Video</span>
        </button>
      </div>
    </div>
  );
}
