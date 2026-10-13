import React, { useState } from 'react';
import { ChevronDown, Cpu, Award } from 'lucide-react';
import {
  MARK2_BENCHMARK,
  MARK2_5_SPECS,
  EMOTIONS,
  EMOTION_META,
} from '../data/modelSpecs';

export default function ModelDetails() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="research-card overflow-hidden">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-white/5 border border-[var(--border-subtle)] flex items-center justify-center text-indigo-400">
            <Cpu className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs sm:text-sm font-semibold text-white tracking-wide font-sans">
                Model Details & Architecture Specifications
              </h3>
              <span className="text-[10px] font-mono text-[var(--text-muted)] bg-white/5 px-1.5 py-0.5 rounded border border-white/5">
                Mark 2
              </span>
            </div>
            <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
              DeBERTa-v3, Wav2Vec2, ViT encoders, cross-modal transformer fusion, and benchmark metrics
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-[var(--text-muted)]">
          <span className="hidden sm:inline">{isOpen ? 'Hide details' : 'Show details'}</span>
          <ChevronDown
            className={`w-4 h-4 transition-transform duration-200 ${
              isOpen ? 'rotate-180' : ''
            }`}
          />
        </div>
      </button>

      {/* Collapsible Content */}
      {isOpen && (
        <div className="p-4 sm:p-5 border-t border-[var(--border-subtle)] space-y-5 bg-[var(--bg-app)]/40 text-xs">
          {/* Architecture Spec Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-2">
              <div className="text-[11px] font-mono font-medium text-indigo-300 uppercase tracking-wider pb-1.5 border-b border-[var(--border-subtle)]">
                Encoder & Projection Architecture
              </div>
              <ul className="space-y-1.5 text-[11px]">
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Text Encoder:</span>
                  <span className="font-mono text-white font-medium">DeBERTa-v3-base</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Audio Encoder:</span>
                  <span className="font-mono text-white font-medium">Wav2Vec2-base-960h</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Vision Encoder:</span>
                  <span className="font-mono text-white font-medium">ViT-base-patch16-224-in21k</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Projection Space:</span>
                  <span className="font-mono text-indigo-300">256-D per modality</span>
                </li>
              </ul>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-2">
              <div className="text-[11px] font-mono font-medium text-indigo-300 uppercase tracking-wider pb-1.5 border-b border-[var(--border-subtle)]">
                Fusion & Classifier Specifications
              </div>
              <ul className="space-y-1.5 text-[11px]">
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Fusion Network:</span>
                  <span className="font-mono text-white">2-Layer Transformer ([CLS], [T], [A], [V])</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Classification Task:</span>
                  <span className="font-mono text-white">6-label sigmoid classifier</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Loss Function:</span>
                  <span className="font-mono text-white">BCEWithLogitsLoss</span>
                </li>
                <li className="flex justify-between py-0.5">
                  <span className="text-[var(--text-muted)]">Benchmark Dataset:</span>
                  <span className="font-mono text-white">CMU-MOSEI (22,856 utterances)</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Official Benchmark Results */}
          <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-3">
            <div className="flex items-center justify-between pb-1.5 border-b border-[var(--border-subtle)]">
              <div className="flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-[11px] font-mono font-medium text-white uppercase tracking-wider">
                  CMU-MOSEI Test Set Benchmark Results
                </span>
              </div>
              <span className="text-[10px] font-mono text-[var(--text-muted)]">
                Checkpoint: checkpoints/mark2_best.pt
              </span>
            </div>

            {/* Top Metrics Row */}
            <div className="grid grid-cols-3 gap-2 font-mono text-center">
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)]">
                <div className="text-[10px] text-[var(--text-muted)] uppercase">Micro-F1</div>
                <div className="text-sm font-semibold text-white">
                  {MARK2_BENCHMARK.microF1.toFixed(4)}
                </div>
              </div>
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)]">
                <div className="text-[10px] text-[var(--text-muted)] uppercase">Macro-F1</div>
                <div className="text-sm font-semibold text-white">
                  {MARK2_BENCHMARK.macroF1.toFixed(4)}
                </div>
              </div>
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)]">
                <div className="text-[10px] text-[var(--text-muted)] uppercase">Subset Accuracy</div>
                <div className="text-sm font-semibold text-white">
                  {MARK2_BENCHMARK.subsetAccuracy.toFixed(4)}
                </div>
              </div>
            </div>

            {/* Per-Emotion F1 Breakdown */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 pt-1 font-mono text-xs">
              {EMOTIONS.map((emotion) => {
                const f1 = MARK2_BENCHMARK.perEmotionF1[emotion];
                const meta = EMOTION_META[emotion];
                return (
                  <div
                    key={emotion}
                    className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)] flex flex-col items-center justify-center gap-0.5"
                  >
                    <span className="text-[10px] text-[var(--text-muted)]">{emotion}</span>
                    <span className="font-medium" style={{ color: meta.color }}>
                      {(f1 * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
