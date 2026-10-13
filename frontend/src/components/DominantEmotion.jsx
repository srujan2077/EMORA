import React from 'react';
import { Sparkles, AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';
import { EMOTION_META } from '../data/modelSpecs';

export default function DominantEmotion({ prediction, threshold = 0.5 }) {
  if (!prediction || !prediction.scores) return null;

  const { scores, predicted_emotions = [] } = prediction;

  // Identify highest scoring emotion
  let dominantEmotion = 'Happy';
  let dominantScore = 0;

  Object.entries(scores).forEach(([emotion, score]) => {
    if (score > dominantScore) {
      dominantScore = score;
      dominantEmotion = emotion;
    }
  });

  const meta = EMOTION_META[dominantEmotion] || {
    color: '#6366f1',
    description: 'Predicted emotional affect',
  };

  const percentage = (dominantScore * 100).toFixed(1);
  const isAboveThreshold = dominantScore >= threshold;

  return (
    <div className="card-elevated p-5 sm:p-6 border border-[var(--border-subtle)] bg-[var(--bg-card)] relative overflow-hidden">
      {/* Subtle background glow */}
      <div
        className="absolute -top-16 -right-16 w-44 h-44 rounded-full blur-3xl opacity-20 pointer-events-none"
        style={{ backgroundColor: meta.color }}
      />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-400 font-mono">
              DOMINANT EMOTION
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-[var(--text-muted)] border border-[var(--border-subtle)] font-mono">
              Highest Probability Signal
            </span>
          </div>

          <div className="flex items-baseline gap-3">
            <h2
              className="text-3xl sm:text-4xl font-extrabold tracking-tight font-mono uppercase"
              style={{ color: meta.color }}
            >
              {dominantEmotion}
            </h2>
            <span className="text-2xl sm:text-3xl font-extrabold text-white font-mono">
              {percentage}%
            </span>
          </div>

          <p className="text-xs text-[var(--text-secondary)] mt-1 max-w-lg">
            {meta.description}
          </p>
        </div>

        {/* Multi-label status badge */}
        <div className="bg-[var(--bg-primary)]/80 border border-[var(--border-subtle)] rounded-xl p-3 sm:max-w-xs shrink-0">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-white mb-1">
            <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
            <span>Multi-Label Context</span>
          </div>
          <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">
            {predicted_emotions.length > 0 ? (
              <>
                <span className="text-emerald-400 font-bold font-mono">
                  {predicted_emotions.length}
                </span>{' '}
                emotion{predicted_emotions.length > 1 ? 's' : ''} currently exceed the{' '}
                <span className="font-mono text-white">optimized</span> decision
                thresholds: <span className="text-white font-mono">{predicted_emotions.join(', ')}</span>.
              </>
            ) : (
              <>
                No emotion reached the <span className="font-mono text-white">optimized</span>{' '}
                thresholds. Showing highest probability signal.
              </>
            )}
          </p>
        </div>
      </div>

      {/* Honest multi-label disclaimer note */}
      <div className="mt-4 pt-3 border-t border-[var(--border-subtle)] flex items-start gap-2 text-[11px] text-[var(--text-muted)]">
        <AlertCircle className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0 mt-0.5" />
        <span>
          <strong>Architecture Note:</strong> EMORA employs multi-label sigmoid classifiers.
          The dominant emotion represents the highest confidence score, but emotions in real speech
          and video are not mutually exclusive.
        </span>
      </div>
    </div>
  );
}
