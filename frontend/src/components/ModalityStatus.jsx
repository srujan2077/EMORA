import React from 'react';
import { CheckCircle2, Clock, FileText, Volume2, Video, Activity } from 'lucide-react';

export default function ModalityStatus({ modalities, loading }) {
  const textProcessed = Boolean(modalities?.text?.processed);
  const audioProcessed = Boolean(modalities?.audio?.processed);
  const visionProcessed = Boolean(modalities?.vision?.processed);

  const items = [
    {
      name: 'TEXT',
      icon: FileText,
      processed: textProcessed,
      details: modalities?.text?.frames ? `${modalities.text.frames} tokens/frames (256-d)` : null,
      color: 'text-indigo-400',
      bgColor: 'bg-indigo-500/10',
      borderColor: 'border-indigo-500/30',
    },
    {
      name: 'AUDIO',
      icon: Volume2,
      processed: audioProcessed,
      details: modalities?.audio?.frames ? `${modalities.audio.frames} acoustic frames (256-d)` : null,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10',
      borderColor: 'border-cyan-500/30',
    },
    {
      name: 'VIDEO',
      icon: Video,
      processed: visionProcessed,
      details: modalities?.vision?.frames ? `${modalities.vision.frames} visual frames (256-d)` : null,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/30',
    },
  ];

  return (
    <div className="card-elevated p-5 sm:p-6 border border-[var(--border-subtle)] bg-[var(--bg-card)]">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-emerald-400" />
        <h3 className="text-sm font-bold uppercase tracking-wider text-white">
          Modality Processing Status
        </h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.name}
              className={`p-3.5 rounded-xl border transition-all ${
                item.processed
                  ? 'bg-emerald-500/5 border-emerald-500/30'
                  : 'bg-[var(--bg-primary)]/50 border-[var(--border-subtle)]'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className={`w-7 h-7 rounded-lg ${item.bgColor} ${item.borderColor} border flex items-center justify-center ${item.color}`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-bold text-white font-mono tracking-wider">
                    {item.name}
                  </span>
                </div>

                {item.processed ? (
                  <span className="text-[11px] font-semibold text-emerald-400 flex items-center gap-1 font-mono">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Processed
                  </span>
                ) : loading ? (
                  <span className="text-[11px] text-amber-400 animate-pulse flex items-center gap-1 font-mono">
                    <Clock className="w-3 h-3" />
                    Running...
                  </span>
                ) : (
                  <span className="text-[11px] text-[var(--text-muted)] font-mono">
                    Idle
                  </span>
                )}
              </div>

              {item.details ? (
                <div className="text-[10px] text-[var(--text-secondary)] font-mono pl-1">
                  ✓ {item.details}
                </div>
              ) : (
                <div className="text-[10px] text-[var(--text-muted)] pl-1">
                  Awaiting backend execution
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
