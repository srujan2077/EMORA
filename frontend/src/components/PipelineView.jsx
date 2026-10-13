import React from 'react';
import { FileText, Volume2, Video, Cpu, ArrowDown, Check } from 'lucide-react';
import { EMOTIONS, EMOTION_META } from '../data/modelSpecs';

export default function PipelineView() {
  return (
    <div className="research-card-elevated p-6 sm:p-8 space-y-6 animate-fade-in border border-[var(--border-default)]">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h2 className="text-base sm:text-lg font-bold text-white tracking-tight font-sans">
            EMORA Mark 2 Pipeline Architecture
          </h2>
        </div>
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
          Deep multimodal fusion architecture integrating DeBERTa-v3, Wav2Vec2, and Vision Transformer representations through a cross-modal Transformer encoder.
        </p>
      </div>

      {/* Architecture Pipeline Stages */}
      <div className="space-y-4 max-w-4xl mx-auto pt-2">
        {/* Stage 1: Input Modalities */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase text-[var(--text-muted)] tracking-wider">
            1. Synchronized Multimodal Inputs
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)]">
              <div className="flex items-center gap-2 text-indigo-400 mb-1">
                <FileText className="w-3.5 h-3.5" />
                <span className="text-xs font-semibold text-white font-sans">Text Utterance</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)] font-mono">Word tokens / Utterances</p>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)]">
              <div className="flex items-center gap-2 text-sky-400 mb-1">
                <Volume2 className="w-3.5 h-3.5" />
                <span className="text-xs font-semibold text-white font-sans">Acoustic Signal</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)] font-mono">16kHz audio waveform</p>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--bg-app)] border border-[var(--border-subtle)]">
              <div className="flex items-center gap-2 text-purple-400 mb-1">
                <Video className="w-3.5 h-3.5" />
                <span className="text-xs font-semibold text-white font-sans">Visual Frames</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)] font-mono">RGB video sequence</p>
            </div>
          </div>
        </div>

        <div className="flex justify-center text-[var(--text-subtle)]">
          <ArrowDown className="w-4 h-4" />
        </div>

        {/* Stage 2: Deep Encoders */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase text-[var(--text-muted)] tracking-wider">
            2. Deep Modality Encoders
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
            <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-1">
              <div className="font-semibold text-white font-sans">DeBERTa-v3-base</div>
              <div className="text-[11px] text-indigo-300">768-D contextual embeddings</div>
              <p className="text-[10px] text-[var(--text-muted)] font-sans mt-1">
                Disentangled attention capturing deep semantic & syntactic nuance.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-1">
              <div className="font-semibold text-white font-sans">Wav2Vec2-base-960h</div>
              <div className="text-[11px] text-sky-300">768-D acoustic representations</div>
              <p className="text-[10px] text-[var(--text-muted)] font-sans mt-1">
                Self-supervised acoustic speech representations encoding tone & prosody.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-1">
              <div className="font-semibold text-white font-sans">ViT-base-patch16-224</div>
              <div className="text-[11px] text-purple-300">768-D visual patch embeddings</div>
              <p className="text-[10px] text-[var(--text-muted)] font-sans mt-1">
                Vision transformer encoding facial action units and affective expressions.
              </p>
            </div>
          </div>
        </div>

        <div className="flex justify-center text-[var(--text-subtle)]">
          <ArrowDown className="w-4 h-4" />
        </div>

        {/* Stage 3: Latent Projections */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase text-[var(--text-muted)] tracking-wider">
            3. Unified Projection Layer
          </div>
          <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-center space-y-1">
            <div className="text-xs font-mono font-medium text-white">
              Linear Projections + LayerNorm → 256 Dimensions
            </div>
            <p className="text-[11px] text-[var(--text-secondary)]">
              Projects heterogeneous 768-D modality vectors into a synchronized 256-D shared latent space: <code className="text-indigo-300 font-mono">[T_proj, A_proj, V_proj]</code>
            </p>
          </div>
        </div>

        <div className="flex justify-center text-[var(--text-subtle)]">
          <ArrowDown className="w-4 h-4" />
        </div>

        {/* Stage 4: Transformer Fusion */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase text-[var(--text-muted)] tracking-wider">
            4. Cross-Modal Transformer Fusion
          </div>
          <div className="p-4 rounded-lg bg-[var(--bg-surface)] border border-indigo-500/30 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white font-sans">
                2-Layer Multimodal Transformer (8 Attention Heads, d=256, ff=1024)
              </span>
              <span className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                Self-Attention
              </span>
            </div>

            <div className="grid grid-cols-4 gap-2 text-center font-mono text-xs">
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)] text-white">
                [CLS]
              </div>
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)] text-indigo-300">
                [TEXT]
              </div>
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)] text-sky-300">
                [AUDIO]
              </div>
              <div className="p-2 rounded bg-[var(--bg-app)] border border-[var(--border-subtle)] text-purple-300">
                [VIDEO]
              </div>
            </div>

            <p className="text-[11px] text-[var(--text-muted)] text-center">
              Learns inter-modal cross-attention dependencies and alignments across spoken text, acoustic tone, and facial dynamics.
            </p>
          </div>
        </div>

        <div className="flex justify-center text-[var(--text-subtle)]">
          <ArrowDown className="w-4 h-4" />
        </div>

        {/* Stage 5: Classification Head */}
        <div className="space-y-2">
          <div className="text-[11px] font-mono uppercase text-[var(--text-muted)] tracking-wider">
            5. Multi-Label Affect Output
          </div>
          <div className="p-3.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-center space-y-1">
            <div className="text-xs font-mono font-medium text-white">
              Linear Classifier (256 → 6 Logits) + Independent Sigmoids
            </div>
            <p className="text-[11px] text-[var(--text-muted)]">
              Evaluated against per-emotion decision thresholds optimized on the validation split.
            </p>
            <div className="flex flex-wrap justify-center gap-2 pt-2 font-mono text-xs">
              {EMOTIONS.map((emotion) => {
                const meta = EMOTION_META[emotion];
                return (
                  <span
                    key={emotion}
                    className="px-2 py-0.5 rounded border text-[11px]"
                    style={{
                      borderColor: meta.borderColor,
                      color: meta.color,
                      backgroundColor: meta.lightColor,
                    }}
                  >
                    {emotion}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
