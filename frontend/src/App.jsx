import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import SegmentSelector from './components/SegmentSelector';
import TextInput from './components/TextInput';
import MediaInput from './components/MediaInput';
import ResultsSection from './components/ResultsSection';
import ArchitectureSection from './components/ArchitectureSection';
import BenchmarksSection from './components/BenchmarksSection';
import { getHealth, getSamples, predictSegment, predictRaw } from './services/api';

/* ============================================================
   EMORA MARK 2 — Editorial Redesign
   All API logic and inference preserved from Mark 1 codebase
   ============================================================ */

export default function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [backendHealth, setBackendHealth] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSegment, setSelectedSegment] = useState('-6rXp3zJ3kc[0]');

  // Modality input states
  const [textInput, setTextInput] = useState(
    'I felt thrilled and overjoyed by the unexpected outcome, but also somewhat cautious about the future.'
  );
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);

  // Inference state
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModalityTarget, setActiveModalityTarget] = useState('all');
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('idle');

  // Pipeline stage indicators
  const [pipelineStage, setPipelineStage] = useState({
    text: false,
    audio: false,
    video: false,
    fusion: false,
  });

  // Scroll reveal
  const observerRef = useRef(null);

  useEffect(() => {
    initApp();
  }, []);

  // IntersectionObserver for section reveals
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
    );

    const revealElements = document.querySelectorAll('.reveal-on-scroll');
    revealElements.forEach((el) => observerRef.current.observe(el));

    return () => observerRef.current?.disconnect();
  });

  const initApp = async () => {
    try {
      const health = await getHealth();
      setBackendHealth(health);

      if (health.status === 'ok') {
        const samplesData = await getSamples();
        if (samplesData?.samples?.length) {
          setSamples(samplesData.samples);
          const initialSegment = samplesData.samples[0] || '-6rXp3zJ3kc[0]';
          setSelectedSegment(initialSegment);
        }
      } else {
        setError('FastAPI backend offline at http://127.0.0.1:8001');
      }
    } catch (err) {
      console.warn('Backend connection warning:', err.message);
      setError('Could not connect to FastAPI server. Please ensure the backend is running.');
    }
  };

  // Main multimodal analysis handler — PRESERVED FROM ORIGINAL
  const handleAnalyze = async (modality = 'all') => {
    setLoading(true);
    setActiveModalityTarget(modality);
    setError(null);
    setProcessingStatus('analyzing');

    setPipelineStage({ text: false, audio: false, video: false, fusion: false });
    const t1 = setTimeout(() => setPipelineStage((s) => ({ ...s, text: true })), 150);
    const t2 = setTimeout(() => setPipelineStage((s) => ({ ...s, audio: true })), 400);
    const t3 = setTimeout(() => setPipelineStage((s) => ({ ...s, video: true })), 700);
    const t4 = setTimeout(() => setPipelineStage((s) => ({ ...s, fusion: true })), 1000);

    try {
      const sendText = modality === 'audio' || modality === 'video' ? '' : textInput;
      const sendAudio = modality === 'text' || modality === 'video' ? null : audioFile;
      const sendVideo = modality === 'text' || modality === 'audio' ? null : videoFile;

      const result = await predictRaw(sendText, sendAudio, sendVideo);

      // Transform Mark 2 response structure — UNCHANGED
      const transformed = {
        model: result.model || 'Mark 2',
        dominant_emotion: result.dominant_emotion,
        segment_id: 'raw-input',
        scores: {},
        predicted_emotions: [],
        threshold: 0,
        modalities: {
          text: { processed: result.modalities?.text },
          audio: { processed: result.modalities?.audio },
          vision: { processed: result.modalities?.video },
        },
        emotions_detail: result.emotions,
      };

      let thresholdSum = 0;
      let thresholdCount = 0;

      Object.entries(result.emotions || {}).forEach(([emotion, data]) => {
        const capitalized = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        transformed.scores[capitalized] = data.probability;
        if (data.detected) {
          transformed.predicted_emotions.push(capitalized);
        }
        thresholdSum += data.threshold;
        thresholdCount++;
      });

      transformed.threshold = thresholdCount > 0 ? thresholdSum / thresholdCount : 0.5;

      setPrediction(transformed);
      setProcessingStatus('completed');
      setPipelineStage({ text: true, audio: true, video: true, fusion: true });

      // Smooth scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 200);
    } catch (err) {
      console.error('Inference error:', err);
      setError(err.message || 'Failed to execute inference on backend');
      setProcessingStatus('failed');
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      setLoading(false);
    }
  };

  // Evaluate aligned benchmark segment — PRESERVED FROM ORIGINAL
  const handleEvaluateSegment = async (segmentId) => {
    setLoading(true);
    setError(null);
    setProcessingStatus('analyzing');

    try {
      const result = await predictSegment(segmentId);

      let domScore = -1;
      let domKey = 'Happy';
      Object.entries(result.scores || {}).forEach(([k, v]) => {
        if (v > domScore) {
          domScore = v;
          domKey = k;
        }
      });

      const emotionsDetail = {};
      Object.entries(result.scores || {}).forEach(([emo, score]) => {
        emotionsDetail[emo.toLowerCase()] = {
          probability: score,
          threshold: result.threshold || 0.5,
          detected: score >= (result.threshold || 0.5),
        };
      });

      const transformed = {
        model: 'Mark 1 (CMU-MOSEI Test Benchmark)',
        dominant_emotion: domKey.toLowerCase(),
        segment_id: result.segment_id,
        scores: result.scores,
        predicted_emotions: result.predicted_emotions || [],
        threshold: result.threshold || 0.5,
        modalities: result.modalities || {},
        emotions_detail: emotionsDetail,
        ground_truth: result.ground_truth || [],
      };

      setPrediction(transformed);
      setProcessingStatus('completed');

      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 200);
    } catch (err) {
      console.error('Segment inference error:', err);
      setError(err.message || 'Failed to evaluate benchmark segment');
      setProcessingStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  const isOnline = backendHealth?.status === 'ok';

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'var(--bg-primary)',
        color: 'var(--text-primary)',
        fontFamily: 'var(--font-sans)',
      }}
    >
      {/* Navigation */}
      <Header
        backendHealth={backendHealth}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Connection Error — minimal inline notice */}
      {error && (
        <div className="editorial-container" style={{ paddingTop: '1rem' }}>
          <div
            className="animate-fade-in"
            style={{
              padding: '0.75rem 1rem',
              borderLeft: '2px solid #f87171',
              color: '#f87171',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
            }}
          >
            <span>{error}</span>
            <button
              onClick={() => {
                setError(null);
                initApp();
              }}
              className="btn-text"
              style={{ color: '#f87171', fontSize: '0.6875rem' }}
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* ═══════ ANALYSIS TAB ═══════ */}
      {activeTab === 'analysis' && (
        <>
          {/* SECTION 01 — Hero */}
          <HeroSection />

          {/* SECTION 02 — Dataset Selector */}
          <section className="editorial-section reveal-on-scroll" style={{ paddingTop: '2rem' }}>
            <div className="editorial-container">
              <SegmentSelector
                samples={samples}
                selectedSegment={selectedSegment}
                onSelectSegment={(id) => setSelectedSegment(id)}
                onEvaluateSegment={handleEvaluateSegment}
                loading={loading}
              />
            </div>
          </section>

          {/* SECTION 03 — Multimodal Input */}
          <section className="editorial-section reveal-on-scroll" style={{ paddingTop: 0 }}>
            <div className="editorial-container">
              {/* Text — Primary input, editorial emphasis */}
              <TextInput
                textValue={textInput}
                setTextValue={setTextInput}
              />

              {/* Audio + Video — Side by side, supporting */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                  gap: '3rem',
                  marginTop: '4rem',
                }}
              >
                <MediaInput
                  type="audio"
                  file={audioFile}
                  setFile={setAudioFile}
                  label="Audio"
                  modelName="Wav2Vec2-base-960h"
                  accept="audio/*"
                  sampleUrl="/samples/sample_audio.wav"
                  sampleName="sample_audio.wav"
                  sampleMime="audio/wav"
                />
                <MediaInput
                  type="video"
                  file={videoFile}
                  setFile={setVideoFile}
                  label="Video"
                  modelName="ViT-base-patch16-224-in21k"
                  accept="video/*"
                  sampleUrl="/samples/sample_video.mp4"
                  sampleName="sample_video.mp4"
                  sampleMime="video/mp4"
                />
              </div>
            </div>
          </section>

          {/* SECTION 04 — Analyze CTA */}
          <section className="editorial-section" style={{ paddingTop: '1rem', paddingBottom: '2rem' }}>
            <div className="editorial-container" style={{ textAlign: 'center' }}>
              <button
                onClick={() => handleAnalyze('all')}
                disabled={loading || !isOnline}
                className="btn-primary"
                id="analyze-button"
              >
                {loading ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="animate-spin-slow" style={{ display: 'inline-block' }}>
                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  'Analyze Emotion →'
                )}
              </button>

              {/* Pipeline progress — minimal */}
              {loading && (
                <div
                  className="animate-fade-in"
                  style={{
                    marginTop: '1.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '1.25rem',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.6875rem',
                    color: 'var(--text-muted)',
                  }}
                >
                  {['Text', 'Audio', 'Video', 'Fusion'].map((stage, i) => {
                    const key = stage.toLowerCase();
                    const done = pipelineStage[key];
                    return (
                      <React.Fragment key={stage}>
                        {i > 0 && (
                          <span style={{ color: 'var(--text-faint)' }}>—</span>
                        )}
                        <span style={{
                          color: done ? 'var(--accent-bright)' : 'var(--text-muted)',
                          transition: 'color 0.3s ease',
                        }}>
                          {stage} {done ? '✓' : '…'}
                        </span>
                      </React.Fragment>
                    );
                  })}
                </div>
              )}
            </div>
          </section>

          {/* SECTION 05 — Results */}
          {prediction && (
            <div id="results-section">
              <ResultsSection prediction={prediction} />
            </div>
          )}

          {/* SECTION 06 — Architecture (collapsed) */}
          <section className="editorial-section reveal-on-scroll">
            <ArchitectureSection />
          </section>
        </>
      )}

      {/* ═══════ ARCHITECTURE TAB ═══════ */}
      {activeTab === 'architecture' && (
        <section className="editorial-section">
          <ArchitectureSection defaultOpen={true} />
        </section>
      )}

      {/* ═══════ BENCHMARKS TAB ═══════ */}
      {activeTab === 'benchmarks' && (
        <section className="editorial-section">
          <BenchmarksSection />
        </section>
      )}

      {/* Footer — minimal */}
      <footer
        style={{
          borderTop: '1px solid var(--border-faint)',
          padding: '2rem 0',
          marginTop: '4rem',
        }}
      >
        <div
          className="editorial-container"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.625rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          <span>EMORA · Multimodal Emotion Intelligence</span>
          <span>CMU-MOSEI · mark2_best.pt</span>
        </div>
      </footer>
    </div>
  );
}
