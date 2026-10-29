import React from 'react';
import { Award, CheckCircle2, TrendingUp, BarChart2 } from 'lucide-react';
import {
  MARK2_BENCHMARK,
  MARK2_5_SPECS,
  EMOTIONS,
  EMOTION_META,
  ARCHITECTURE_DETAILS,
} from '../data/modelSpecs';

export default function BenchmarksView() {
  return (
    <div className="research-card-elevated p-6 sm:p-8 space-y-6 animate-fade-in border border-[var(--border-default)]">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Award className="w-4 h-4 text-amber-400" />
          <h2 className="text-base sm:text-lg font-bold text-white tracking-tight font-sans">
            CMU-MOSEI Official Benchmark Evaluation
          </h2>
        </div>
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
          Quantitative multi-label emotion recognition metrics on the held-out test split of the CMU Multimodal Opinion Sentiment and Emotion Intensity (MOSEI) dataset.
        </p>
      </div>

      {/* Top Level Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-center">
        <div className="p-4 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)] space-y-1">
          <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">Test Micro-F1</div>
          <div className="text-2xl font-bold text-white">
            {MARK2_BENCHMARK.microF1.toFixed(4)}
          </div>
          <div className="text-[10px] text-emerald-400">Harmonic mean across instances</div>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)] space-y-1">
          <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">Test Macro-F1</div>
          <div className="text-2xl font-bold text-white">
            {MARK2_BENCHMARK.macroF1.toFixed(4)}
          </div>
          <div className="text-[10px] text-indigo-400">Unweighted class average</div>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)] space-y-1">
          <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">Subset Accuracy</div>
          <div className="text-2xl font-bold text-white">
            {MARK2_BENCHMARK.subsetAccuracy.toFixed(4)}
          </div>
          <div className="text-[10px] text-purple-400">Strict exact multi-label match</div>
        </div>
      </div>

      {/* Per-Emotion Breakdown Table */}
      <div className="p-4 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-3">
        <div className="text-xs font-semibold text-white uppercase font-mono tracking-wider">
          Class-Wise Affect Performance Breakdown
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono text-left">
            <thead>
              <tr className="border-b border-[var(--border-subtle)] text-[var(--text-muted)] text-[11px]">
                <th className="pb-2.5 font-medium">Emotion</th>
                <th className="pb-2.5 font-medium">Test F1</th>
                <th className="pb-2.5 font-medium">Score Relative Bar</th>
                <th className="pb-2.5 font-medium hidden sm:table-cell">Affect Profile</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {EMOTIONS.map((emotion) => {
                const f1 = MARK2_BENCHMARK.perEmotionF1[emotion];
                const meta = EMOTION_META[emotion];
                return (
                  <tr key={emotion}>
                    <td className="py-2.5 text-white font-medium flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full inline-block"
                        style={{ backgroundColor: meta.color }}
                      />
                      <span>{emotion}</span>
                    </td>
                    <td className="py-2.5 text-white font-medium">
                      {(f1 * 100).toFixed(1)}% <span className="text-[10px] text-[var(--text-muted)]">({f1.toFixed(4)})</span>
                    </td>
                    <td className="py-2.5">
                      <div className="w-32 sm:w-48 h-1.5 rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${f1 * 100}%`,
                            backgroundColor: meta.color,
                          }}
                        />
                      </div>
                    </td>
                    <td className="py-2.5 text-[11px] text-[var(--text-muted)] hidden sm:table-cell font-sans">
                      {meta.description}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Evolution Comparison Table */}
      <div className="p-4 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-3">
        <div className="text-xs font-semibold text-white uppercase font-mono tracking-wider">
          Architecture Evolution: Mark 1 → Mark 2 → Mark 3
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-sans">
          <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)] space-y-2">
            <div className="font-mono text-xs font-semibold text-zinc-400">
              Mark 1: Baseline Architecture
            </div>
            <ul className="space-y-1 text-[11px] text-[var(--text-secondary)] font-mono">
              <li>• Text: GloVe + BiLSTM</li>
              <li>• Audio: COVAREP + BiLSTM</li>
              <li>• Vision: FACET + BiLSTM</li>
              <li>• Fusion: Concat (384-d) + MLP</li>
              <li>• Checkpoint: mark1_best.pt</li>
            </ul>
          </div>

          <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)] space-y-2">
            <div className="font-mono text-xs font-semibold text-indigo-300">
              Mark 2: Deep Transformer Fusion
            </div>
            <ul className="space-y-1 text-[11px] text-[var(--text-secondary)] font-mono">
              <li>• Text: DeBERTa-v3-base</li>
              <li>• Audio: Wav2Vec2-base-960h</li>
              <li>• Vision: ViT-base-patch16-224</li>
              <li>• Fusion: 2-Layer Transformer</li>
              <li>• Checkpoint: mark2_best.pt</li>
            </ul>
          </div>

          <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-emerald-500/40 space-y-2">
            <div className="font-mono text-xs font-semibold text-emerald-400">
              Mark 3: Temporal SOTA Ensemble
            </div>
            <ul className="space-y-1 text-[11px] text-[var(--text-secondary)] font-mono">
              <li>• Encoders: DeBERTa + Wav2Vec2 + ViT</li>
              <li>• Temporal: Causal window (T ≤ 8)</li>
              <li>• Candidate: Controlled split_at_gap</li>
              <li>• Ensemble: 5 seeds (42..2026)</li>
              <li>• Thresholds: Fixed 0.5 provisional policy</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
