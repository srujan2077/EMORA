import React from 'react';
import { BarChart3, CheckCircle2, CircleDot, Tag } from 'lucide-react';
import { EMOTIONS, EMOTION_META } from '../data/modelSpecs';

export default function EmotionProfile({ prediction, threshold = 0.5 }) {
  if (!prediction || !prediction.scores) return null;

  const { scores = {}, predicted_emotions = [], ground_truth = [] } = prediction;

  return (
    <div className="card-elevated p-5 sm:p-6 border border-[var(--border-subtle)] bg-[var(--bg-card)]">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              Emotion Profile
            </h3>
          </div>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">
            Sigmoid probability activations across all six core affective dimensions.
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/80 inline-block" />
            <span>Active (≥ {threshold * 100}%)</span>
          </div>
          <div className="flex items-center gap-1.5 text-[var(--text-muted)]">
            <span className="w-2.5 h-2.5 rounded-sm bg-white/20 inline-block" />
            <span>Sub-threshold</span>
          </div>
        </div>
      </div>

      {/* Grid of horizontal probability bars */}
      <div className="space-y-4">
        {EMOTIONS.map((emotion) => {
          const score = scores[emotion] !== undefined ? scores[emotion] : 0;
          const emotionKey = emotion.toLowerCase();
          const perEmotionThreshold = prediction.emotions_detail?.[emotionKey]?.threshold ?? threshold;
          const percentage = (score * 100).toFixed(1);
          const isPredicted = score >= perEmotionThreshold;
          const isGroundTruth = ground_truth.includes(emotion);
          const meta = EMOTION_META[emotion] || { color: '#6366f1' };

          return (
            <div key={emotion} className="space-y-1.5">
              {/* Row title and badges */}
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white tracking-wide">{emotion}</span>

                  {isPredicted ? (
                    <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                      <CheckCircle2 className="w-2.5 h-2.5" />
                      PREDICTED
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-[var(--text-muted)] border border-[var(--border-subtle)]">
                      NOT PREDICTED
                    </span>
                  )}

                  {isGroundTruth && (
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 flex items-center gap-1">
                      <Tag className="w-2.5 h-2.5" />
                      GROUND TRUTH
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 font-mono">
                  <span className="text-white font-bold">{percentage}%</span>
                  <span className="text-[var(--text-muted)] text-[10px]">
                    ({score.toFixed(4)})
                  </span>
                </div>
              </div>

              {/* Progress bar container */}
              <div className="relative w-full h-3 bg-[var(--bg-primary)] rounded-full overflow-hidden border border-[var(--border-subtle)]">
                {/* 50% Threshold marker line */}
                <div
                  className="absolute top-0 bottom-0 w-[2px] bg-white/30 z-10"
                  style={{ left: `${perEmotionThreshold * 100}%` }}
                  title={`Decision Threshold (τ = ${perEmotionThreshold})`}
                />

                {/* Animated colored bar */}
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.min(100, Math.max(1, score * 100))}%`,
                    backgroundColor: isPredicted ? meta.color : '#475569',
                    boxShadow: isPredicted ? `0 0 12px ${meta.color}50` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Threshold Guide */}
      <div className="mt-5 pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between text-[11px] text-[var(--text-muted)] font-mono">
        <span>0%</span>
        <span className="text-white/70">
          {prediction.emotions_detail ? 'Fixed 0.5 decision thresholds' : `Decision Threshold: τ = ${threshold.toFixed(2)}`}
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}
