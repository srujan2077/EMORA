import React, { useRef, useEffect } from 'react';

/**
 * AffectiveCanvas — Interactive Research Instrument
 *
 * Renders a harmonic multimodal signal manifold that responds subtly
 * and fluidly to pointer movement, simulating the live waveform spectrum
 * of language semantics, acoustic prosody, and visual facial dynamics.
 *
 * Strict monochrome styling: pure ink lines on warm off-white.
 */
export default function AffectiveCanvas({ height = 240, active = true }) {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5, speed: 0 });
  const animFrameRef = useRef(null);
  const isVisibleRef = useRef(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = (canvas.width = canvas.offsetWidth * window.devicePixelRatio);
    let h = (canvas.height = canvas.offsetHeight * window.devicePixelRatio);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      h = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    };
    window.addEventListener('resize', handleResize);

    // Track pointer movement relative to canvas
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      mouseRef.current.targetX = Math.max(0, Math.min(1, x));
      mouseRef.current.targetY = Math.max(0, Math.min(1, y));
    };

    const handleMouseLeave = () => {
      mouseRef.current.targetX = 0.5;
      mouseRef.current.targetY = 0.5;
    };

    window.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    // IntersectionObserver to pause when offscreen
    const observer = new IntersectionObserver(([entry]) => {
      isVisibleRef.current = entry.isIntersecting;
    }, { threshold: 0.05 });
    observer.observe(canvas);

    // Reduced motion check
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let time = 0;

    const render = () => {
      if (!isVisibleRef.current || !active) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // Smooth pointer interpolation
      const m = mouseRef.current;
      m.x += (m.targetX - m.x) * 0.06;
      m.y += (m.targetY - m.y) * 0.06;

      ctx.clearRect(0, 0, width, h);

      const dpr = window.devicePixelRatio || 1;
      const centerY = h * 0.52;
      const amplitudeBase = h * 0.18;

      if (!prefersReducedMotion) {
        time += 0.016;
      }

      // Modality wave harmonics:
      // Wave 0: DeBERTa text semantics (low frequency, steady carrier)
      // Wave 1: Wav2Vec2 acoustic pitch & energy (mid frequency, dynamic vibrato)
      // Wave 2: ViT visual facial micro-movements (high frequency, subtle shimmer)
      const layers = [
        { freq: 0.0028, speed: 1.1, amp: 1.0, alpha: 0.90, lw: 1.5 * dpr, dash: [] },
        { freq: 0.0055, speed: 1.6, amp: 0.65, alpha: 0.55, lw: 1.0 * dpr, dash: [] },
        { freq: 0.011, speed: 2.2, amp: 0.38, alpha: 0.32, lw: 0.75 * dpr, dash: [4 * dpr, 4 * dpr] },
        { freq: 0.0018, speed: 0.8, amp: 1.3, alpha: 0.20, lw: 0.75 * dpr, dash: [] },
      ];

      // Draw subtle background grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1 * dpr;
      const gridStep = 40 * dpr;
      ctx.beginPath();
      for (let x = 0; x < width; x += gridStep) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
      }
      for (let y = 0; y < h; y += gridStep) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Draw harmonic waveforms
      layers.forEach((layer, layerIdx) => {
        ctx.save();
        ctx.strokeStyle = `rgba(255, 255, 255, ${layer.alpha})`;
        ctx.lineWidth = layer.lw;
        if (layer.dash.length) ctx.setLineDash(layer.dash);

        ctx.beginPath();
        const step = 4 * dpr;
        for (let x = 0; x <= width; x += step) {
          const normX = x / width;

          // Influence of mouse proximity
          const distToMouse = Math.abs(normX - m.x);
          const mouseDeflect = Math.exp(-distToMouse * 5.0) * (m.y - 0.5) * 1.8;

          // Envelope window: tapers softly at edges
          const envelope = Math.sin(normX * Math.PI);

          const w1 = Math.sin(x * layer.freq + time * layer.speed + layerIdx);
          const w2 = Math.cos(x * layer.freq * 1.7 - time * layer.speed * 0.7);
          const displacement = (w1 * 0.7 + w2 * 0.3 + mouseDeflect) * amplitudeBase * layer.amp * envelope;

          const y = centerY + displacement;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.restore();
      });

      // Draw interactive focal crosshair at mouse position
      const focalX = m.x * width;
      const focalY = centerY + Math.sin(focalX * 0.003 + time) * amplitudeBase * 0.8 * Math.sin(m.x * Math.PI);

      // Crosshair lines
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([2 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(focalX, 10 * dpr);
      ctx.lineTo(focalX, h - 10 * dpr);
      ctx.stroke();

      // Concentric target ring at focal point with subtle monochrome glow
      ctx.setLineDash([]);
      ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
      ctx.shadowBlur = 8 * dpr;
      ctx.beginPath();
      ctx.arc(focalX, focalY, 4 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.arc(focalX, focalY, 10 * dpr, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.stroke();
      ctx.restore();

      // Technical annotations along the top & bottom
      ctx.save();
      ctx.font = `${8 * dpr}px 'JetBrains Mono', monospace`;
      ctx.fillStyle = '#A0A0A0';
      ctx.fillText(`SIG: 48.0 kHz // HARMONIC PROJECTION`, 12 * dpr, 18 * dpr);
      ctx.fillText(`FOCAL: [${(m.x * 100).toFixed(1)}%, ${(m.y * 100).toFixed(1)}%]`, width - 150 * dpr, 18 * dpr);
      ctx.fillText(`CHANNELS: L-DeBERTa · A-Wav2Vec2 · V-ViT`, 12 * dpr, h - 10 * dpr);
      ctx.fillText(`TENSOR: 768-D`, width - 90 * dpr, h - 10 * dpr);
      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      observer.disconnect();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [active]);

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: `${height}px`,
        overflow: 'hidden',
        border: '1px solid var(--divider)',
        background: '#000000',
        cursor: 'crosshair',
      }}
      role="img"
      aria-label="Interactive Affective Signal Scanner — Multimodal waveform harmonics responding to cursor movement"
    >
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
      />
    </div>
  );
}
