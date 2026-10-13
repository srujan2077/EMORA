import React from 'react';
import { Sparkles, Info, Check } from 'lucide-react';
import { EMOTIONS, EMOTION_META } from '../data/modelSpecs';

export default function ResultsPanel({ prediction }) {
  if (!prediction || !prediction.scores) return null;

  const {
    scores = {},
    predicted_emotions = [],
    emotions_detail = {},
    modalities = {},
    model = 'Mark 2',
    segment_id,
  } = prediction;

  // Find dominant emotion
  let dominantEmotion = 'Happy';
  let dominantScore = 0;

  Object.entries(scores).forEach(([emotion, score]) => {
    if (score > dominantScore) {
      dominantScore = score;
      dominantEmotion = emotion;
    }
  });

  const dominantMeta = EMOTION_META[dominantEmotion] || {
    color: '#6366f1',
    description: 'Predicted emotional affect',
  };

  const dominantPercentage = (dominantScore * 100).toFixed(1);

  // Determine active modalities
  const textActive = modalities?.text?.processed ?? false;
  const audioActive = modalities?.audio?.processed ?? false;
  const videoActive = modalities?.vision?.processed ?? false;

  return (
    <div className="research-card-elevated p-5 sm:p-7 space-y-6 animate-fade-in border border-[var(--border-default)]">
      {/* Top Meta Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border-subtle)]">
        <span className="text-[11px] font-mono uppercase tracking-wider text-[var(--text-muted)]">
          ANALYSIS RESULT
        </span>

        <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-muted)]">
          <span>Inference: <strong className="text-white font-medium">{model}</strong></span>
          {segment_id && segment_id !== 'raw-input' && (
            <>
              <span>•</span>
              <span>Target: <code className="text-indigo-300">{segment_id}</code></span>
            </>
          )}
        </div>
      </div>

      {/* Hero: Dominant Emotion & Multi-label Context */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
        {/* Dominant Emotion Block */}
        <div className="md:col-span-6 space-y-1.5">
          <div className="text-[11px] font-mono uppercase text-[var(--text-secondary)] tracking-wide">
            Dominant Emotion
          </div>

          <div className="flex items-baseline gap-3">
            <h2
              className="text-3xl sm:text-4xl font-bold tracking-tight font-sans"
              style={{ color: dominantMeta.color }}
            >
              {dominantEmotion.toUpperCase()}
            </h2>
            <span className="text-2xl sm:text-3xl font-mono font-semibold text-white">
              {dominantPercentage}%
            </span>
          </div>

          <p className="text-xs text-[var(--text-secondary)]">
            {dominantMeta.description}
          </p>
        </div>

        {/* Multi-Label Explanation & Context */}
        <div className="md:col-span-6 bg-[var(--bg-app)]/70 p-3.5 rounded-lg border border-[var(--border-subtle)] space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-medium text-white font-sans">
            <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span>Multi-label Emotion Classification</span>
          </div>

          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
            Multiple emotions may be active simultaneously. The model applies independent sigmoid units rather than a single mutually-exclusive softmax.
          </p>

          <div className="text-[11px] font-mono text-[var(--text-muted)] pt-1 border-t border-[var(--border-subtle)]">
            {predicted_emotions.length > 0 ? (
              <span className="text-emerald-400">
                Active signals ({predicted_emotions.length}): {predicted_emotions.join(', ')}
              </span>
            ) : (
              <span>Sub-threshold across all dimensions; displaying highest probability signal.</span>
            )}
          </div>
        </div>
      </div>

      {/* Horizontal Probability Bars */}
      <div className="space-y-3 pt-2">
        <div className="text-[11px] font-mono uppercase tracking-wider text-[var(--text-muted)] mb-2">
          Affective Probability Distribution
        </div>

        <div className="space-y-2.5">
          {EMOTIONS.map((emotion) => {
            const score = scores[emotion] !== undefined ? scores[emotion] : 0;
            const emotionKey = emotion.toLowerCase();
            const detail = emotions_detail[emotionKey];
            const threshold = detail?.threshold ?? 0.5;
            const percentage = (score * 100).toFixed(1);
            const isDominant = emotion === dominantEmotion;
            const isDetected = detail?.detected ?? (score >= threshold);
            const meta = EMOTION_META[emotion] || { color: '#6366f1' };

            return (
              <div
                key={emotion}
                className={`p-2.5 rounded-lg transition-colors ${
                  isDominant
                    ? 'bg-white/[0.04] border border-white/10'
                    : 'bg-transparent border border-transparent'
                }`}
              >
                {/* Row Header */}
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-medium ${
                        isDominant ? 'text-white font-semibold' : 'text-[var(--text-secondary)]'
                      }`}
                    >
                      {emotion}
                    </span>

                    {isDominant && (
                      <span className="text-[10px] font-mono font-medium px-1.5 py-0.2 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/25">
                        Dominant
                      </span>
                    )}

                    {isDetected && !isDominant && (
                      <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded border border-emerald-500/20">
                        Active
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2.5 font-mono text-xs">
                    <span className="text-white font-medium">{percentage}%</span>
                    <span className="text-[var(--text-muted)] text-[11px]">
                      ({score.toFixed(4)})
                    </span>
                    <span className="text-[10px] text-[var(--text-subtle)] hidden sm:inline">
                      τ = {threshold.toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Progress Track */}
                <div className="relative w-full h-2 bg-white/5 rounded-full overflow-hidden">
                  {/* Decision Threshold Marker Line */}
                  <div
                    className="absolute top-0 bottom-0 w-[1.5px] bg-white/20 z-10"
                    style={{ left: `${Math.min(99, threshold * 100)}%` }}
                    title={`Decision threshold: τ = ${threshold}`}
                  />

                  {/* Probability Fill Bar */}
                  <div
                    className="h-full rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: `${Math.min(100, Math.max(1, score * 100))}%`,
                      backgroundColor: isDominant
                        ? meta.color
                        : isDetected
                        ? meta.color
                        : '#334155',
                      opacity: isDominant ? 1 : isDetected ? 0.85 : 0.45,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Compact Pipeline Status Row */}
      <div className="pt-4 border-t border-[var(--border-subtle)] flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[var(--text-muted)]">
        <div className="flex items-center gap-2">
          <span>Pipeline:</span>
          <span className="flex items-center gap-1">
            <span className={textActive ? 'text-emerald-400' : 'text-[var(--text-subtle)]'}>
              Text {textActive ? '✓' : '—'}
            </span>
            <span className="text-[var(--text-subtle)]">──</span>
            <span className={audioActive ? 'text-emerald-400' : 'text-[var(--text-subtle)]'}>
              Audio {audioActive ? '✓' : '—'}
            </span>
            <span className="text-[var(--text-subtle)]">──</span>
            <span className={videoActive ? 'text-emerald-400' : 'text-[var(--text-subtle)]'}>
              Video {videoActive ? '✓' : '—'}
            </span>
            <span className="text-[var(--text-subtle)]">──</span>
            <span className="text-indigo-400">
              Transformer Fusion ✓
            </span>
          </span>
        </div>

        <div className="text-[11px] text-[var(--text-subtle)]">
          BCEWithLogitsLoss · 6-Class Sigmoid
        </div>
      </div>
    </div>
  );
}
