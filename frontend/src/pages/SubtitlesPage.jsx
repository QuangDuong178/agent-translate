import { useState, useEffect, useRef } from 'react';
import {
    Film, Youtube, Upload, Globe, Play, Download, RefreshCw,
    FileText, Languages, Clock, AlertCircle, CheckCircle2,
    Loader2, ArrowRightLeft, Subtitles as SubtitlesIcon,
    Sparkles, Settings, Key, ArrowRight, Eye, EyeOff,
} from 'lucide-react';
import toast from 'react-hot-toast';
import {
    listModels, listSubtitleJobs, getSubtitleJob,
    extractSubtitles, uploadVideoForSubtitles,
    translateSubtitles, downloadSubtitleFile,
    getSettings, updateSettings,
    submitCorrection, getCorrectionStats,
} from '../api';

const WHISPER_MODELS = [
    { id: 'tiny', name: 'Tiny', desc: 'Fastest, ~1GB RAM' },
    { id: 'base', name: 'Base', desc: 'Balanced speed & accuracy' },
    { id: 'small', name: 'Small', desc: 'Better accuracy' },
    { id: 'medium', name: 'Medium', desc: 'High accuracy' },
    { id: 'large-v3', name: 'Large V3', desc: 'Best accuracy' },
];

const LANG_MAP = {
    en: { name: 'English', flag: '🇺🇸' },
    vi: { name: 'Vietnamese', flag: '🇻🇳' },
    zh: { name: 'Chinese', flag: '🇨🇳' },
    ja: { name: 'Japanese', flag: '🇯🇵' },
    ko: { name: 'Korean', flag: '🇰🇷' },
    fr: { name: 'French', flag: '🇫🇷' },
    de: { name: 'German', flag: '🇩🇪' },
    es: { name: 'Spanish', flag: '🇪🇸' },
    auto: { name: 'Auto Detect', flag: '🔍' },
};

const PIPELINE_NODES = [
    { id: 'input', label: '📥 Input', icon: Youtube },
    { id: 'download', label: '⬇️ Download', icon: Download },
    { id: 'transcribe', label: '🎤 Transcribe', icon: FileText },
    { id: 'detect', label: '🔍 Detect', icon: Globe },
    { id: 'translate', label: '🔄 Translate', icon: Languages },
    { id: 'output', label: '📄 Output', icon: CheckCircle2 },
];

const NODE_COLORS = {
    pending: { bg: '#1e293b', border: '#334155', text: '#94a3b8' },
    running: { bg: '#1e1b4b', border: '#6366f1', text: '#a5b4fc' },
    completed: { bg: '#052e16', border: '#22c55e', text: '#86efac' },
    failed: { bg: '#450a0a', border: '#ef4444', text: '#fca5a5' },
    skipped: { bg: '#1c1917', border: '#57534e', text: '#a8a29e' },
};

export default function SubtitlesPage() {
    const [tab, setTab] = useState('youtube');
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [sourceLang, setSourceLang] = useState('auto');
    const [targetLang, setTargetLang] = useState('vi');
    const [whisperModel, setWhisperModel] = useState('base');
    const [modelId, setModelId] = useState('');
    const [enableRefine, setEnableRefine] = useState(false);
    const [models, setModels] = useState([]);
    const [jobs, setJobs] = useState([]);
    const [activeJob, setActiveJob] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [srtView, setSrtView] = useState('side-by-side');
    const [showSettings, setShowSettings] = useState(false);
    const [geminiKey, setGeminiKey] = useState('');
    const [hasGeminiKey, setHasGeminiKey] = useState(false);
    const [showKey, setShowKey] = useState(false);
    const fileInputRef = useRef(null);
    const pollRef = useRef(null);
    const liveScrollRef = useRef(null);

    useEffect(() => {
        fetchModels();
        fetchJobs();
        fetchSettings();
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, []);

    // Auto-select model when target language changes
    useEffect(() => {
        if (models.length === 0 || !targetLang) return;
        const tgt = targetLang.toLowerCase();
        const scored = models
            .filter((m) => m.status === 'ready')
            .map((m) => {
                const alias = (m.alias || '').toLowerCase();
                const hf = (m.hf_name || '').toLowerCase();
                let score = 0;
                if (alias.includes(`→${tgt}`) || alias.includes(`->${tgt}`)) score += 10;
                if (hf.includes(`-${tgt}`) && !hf.includes(`-${tgt}-`)) score += 10;
                if (tgt === 'ja' && (hf.includes('en-jap') || hf.includes('en-ja'))) score += 10;
                if (m.id.startsWith('ft-') || hf.includes('fine-tuned')) score += 5;
                return { ...m, score };
            })
            .filter((m) => m.score > 0)
            .sort((a, b) => b.score - a.score);
        setModelId(scored.length > 0 ? scored[0].id : '');
    }, [targetLang, models]);

    const fetchModels = async () => {
        try {
            const { data } = await listModels();
            setModels(data.models?.filter((m) => m.status === 'ready') || []);
        } catch (err) { console.error(err); }
    };

    const fetchJobs = async () => {
        try {
            const { data } = await listSubtitleJobs();
            setJobs(data.jobs || []);
        } catch (err) { console.error(err); }
    };

    const fetchSettings = async () => {
        try {
            const { data } = await getSettings();
            setHasGeminiKey(data.gemini_api_key);
        } catch (err) { console.error(err); }
    };

    const saveGeminiKey = async () => {
        try {
            await updateSettings({ gemini_api_key: geminiKey });
            setHasGeminiKey(true);
            setGeminiKey('');
            setShowKey(false);
            toast.success('Gemini API key saved!');
        } catch (err) { toast.error('Failed to save API key'); }
    };

    const startPolling = (jobId) => {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const { data } = await getSubtitleJob(jobId);
                setActiveJob(data);
                // Smart auto-scroll: only if user is near bottom
                if (data.live_translations && liveScrollRef.current) {
                    const el = liveScrollRef.current;
                    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
                    if (isNearBottom) {
                        el.scrollTop = el.scrollHeight;
                    }
                }
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    fetchJobs();
                    if (data.status === 'completed') {
                        toast.success('Pipeline completed successfully!');
                    } else {
                        toast.error(`Failed: ${data.error}`);
                    }
                }
            } catch (err) { console.error(err); }
        }, 1000);
    };

    const handleExtract = async () => {
        if (!youtubeUrl.trim()) { toast.error('Please enter a YouTube URL'); return; }
        setIsSubmitting(true);
        try {
            const { data } = await extractSubtitles({
                youtube_url: youtubeUrl.trim(),
                source_lang: sourceLang,
                target_lang: targetLang,
                model_id: modelId || null,
                whisper_model: whisperModel,
                enable_refine: enableRefine,
            });
            toast.success('Pipeline started!');
            startPolling(data.job_id);
            const { data: jobData } = await getSubtitleJob(data.job_id);
            setActiveJob(jobData);
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to start');
        }
        setIsSubmitting(false);
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setIsSubmitting(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('source_lang', sourceLang);
            formData.append('target_lang', targetLang);
            formData.append('model_id', modelId || '');
            formData.append('whisper_model', whisperModel);
            formData.append('enable_refine', enableRefine);
            const { data } = await uploadVideoForSubtitles(formData);
            toast.success('Video uploaded, pipeline started!');
            startPolling(data.job_id);
            const { data: jobData } = await getSubtitleJob(data.job_id);
            setActiveJob(jobData);
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to upload');
        }
        setIsSubmitting(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const viewJob = async (job) => {
        try {
            const { data } = await getSubtitleJob(job.id);
            setActiveJob(data);
            if (data.status !== 'completed' && data.status !== 'failed') {
                startPolling(job.id);
            }
        } catch (err) { console.error(err); }
    };

    const formatDuration = (seconds) => {
        if (!seconds) return '—';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    // ── Pipeline Node Component ──
    const PipelineNode = ({ node, status, isLast }) => {
        const st = status || 'pending';
        const colors = NODE_COLORS[st] || NODE_COLORS.pending;
        const meta = activeJob?.pipeline?.[node.id]?.meta || {};

        return (
            <div style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{
                    background: colors.bg,
                    border: `2px solid ${colors.border}`,
                    borderRadius: 12,
                    padding: '10px 16px',
                    minWidth: 110,
                    textAlign: 'center',
                    position: 'relative',
                    transition: 'all 0.3s ease',
                    boxShadow: st === 'running' ? `0 0 15px ${colors.border}44` : 'none',
                }}>
                    <div style={{ fontSize: 16, marginBottom: 2 }}>
                        {st === 'running' ? (
                            <Loader2 size={16} className="spin" style={{ color: colors.text }} />
                        ) : st === 'completed' ? (
                            <CheckCircle2 size={16} style={{ color: colors.text }} />
                        ) : st === 'failed' ? (
                            <AlertCircle size={16} style={{ color: colors.text }} />
                        ) : (
                            <span style={{ fontSize: 14 }}>{node.label.split(' ')[0]}</span>
                        )}
                    </div>
                    <div style={{
                        fontSize: 11, fontWeight: 600, color: colors.text,
                        letterSpacing: '0.02em', whiteSpace: 'nowrap',
                    }}>
                        {node.label.split(' ').slice(1).join(' ')}
                    </div>
                    {meta.segment_count && (
                        <div style={{ fontSize: 9, color: colors.text, opacity: 0.7, marginTop: 2 }}>
                            {meta.segment_count} segments
                        </div>
                    )}
                    {meta.detected_lang && (
                        <div style={{ fontSize: 9, color: colors.text, opacity: 0.7, marginTop: 2 }}>
                            {LANG_MAP[meta.detected_lang]?.flag} {meta.detected_lang}
                        </div>
                    )}
                    {st === 'skipped' && (
                        <div style={{ fontSize: 9, color: '#78716c', marginTop: 2, fontStyle: 'italic' }}>
                            skipped
                        </div>
                    )}
                </div>
                {!isLast && (
                    <div style={{ padding: '0 4px', display: 'flex', alignItems: 'center' }}>
                        <ArrowRight size={14} style={{
                            color: st === 'completed' ? '#22c55e' : '#334155',
                            transition: 'color 0.3s',
                        }} />
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="page-container">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <div>
                    <h1 className="page-title" style={{ marginBottom: 4 }}>
                        <SubtitlesIcon style={{ display: 'inline', marginRight: 8, verticalAlign: -4 }} />
                        Video Subtitles Pipeline
                    </h1>
                    <p className="page-subtitle" style={{ margin: 0 }}>
                        Automated subtitle extraction & translation pipeline
                    </p>
                </div>
                <button
                    onClick={() => setShowSettings(!showSettings)}
                    style={{
                        display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px',
                        background: showSettings ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                        color: showSettings ? '#fff' : 'var(--text-secondary)',
                        border: '1px solid var(--border-subtle)', borderRadius: 8,
                        cursor: 'pointer', fontSize: 13, fontWeight: 500,
                    }}
                >
                    <Settings size={14} /> Settings
                </button>
            </div>

            {/* Settings Panel */}
            {showSettings && (
                <div className="card" style={{ marginBottom: 20, border: '1px solid var(--accent-primary)' }}>
                    <h3 style={{ margin: '0 0 12px', fontSize: 15, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Key size={16} /> Gemini API Key
                        {hasGeminiKey && <span style={{ fontSize: 11, color: '#22c55e', fontWeight: 400 }}>✓ Configured</span>}
                    </h3>
                    <p style={{ margin: '0 0 12px', fontSize: 13, color: 'var(--text-muted)' }}>
                        Required for the ✨ Refine step. Get a free key at{' '}
                        <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer"
                            style={{ color: 'var(--accent-primary)' }}>
                            Google AI Studio
                        </a>
                    </p>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <div style={{ flex: 1, position: 'relative' }}>
                            <input
                                type={showKey ? 'text' : 'password'}
                                value={geminiKey}
                                onChange={(e) => setGeminiKey(e.target.value)}
                                placeholder={hasGeminiKey ? '••••••••••••• (already set)' : 'Enter Gemini API key'}
                                className="input"
                                style={{ paddingRight: 36, width: '100%' }}
                            />
                            <button onClick={() => setShowKey(!showKey)}
                                style={{
                                    position: 'absolute', right: 8, top: 10, background: 'none',
                                    border: 'none', cursor: 'pointer', color: 'var(--text-muted)',
                                }}>
                                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                            </button>
                        </div>
                        <button
                            className="btn btn-primary"
                            onClick={saveGeminiKey}
                            disabled={!geminiKey.trim()}
                            style={{ padding: '0 20px', fontSize: 13, whiteSpace: 'nowrap' }}
                        >
                            Save Key
                        </button>
                    </div>
                </div>
            )}

            {/* Input Section */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', gap: 0, marginBottom: 16, borderBottom: '1px solid var(--border-subtle)' }}>
                    {[['youtube', Youtube, 'YouTube URL'], ['upload', Upload, 'Upload Video']].map(([id, Icon, label]) => (
                        <button key={id} onClick={() => setTab(id)} style={{
                            padding: '10px 20px', border: 'none', cursor: 'pointer',
                            background: tab === id ? 'var(--accent-primary)' : 'transparent',
                            color: tab === id ? '#fff' : 'var(--text-secondary)',
                            borderRadius: '8px 8px 0 0', fontWeight: 600, fontSize: 13,
                            display: 'flex', alignItems: 'center', gap: 6, transition: 'all 0.2s',
                        }}>
                            <Icon size={14} /> {label}
                        </button>
                    ))}
                </div>

                {tab === 'youtube' && (
                    <div style={{ marginBottom: 14 }}>
                        <div style={{ position: 'relative' }}>
                            <Youtube size={18} style={{ position: 'absolute', left: 14, top: 13, color: '#ef4444' }} />
                            <input type="text" value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)}
                                placeholder="https://www.youtube.com/watch?v=..."
                                className="input" style={{ paddingLeft: 42, width: '100%' }} />
                        </div>
                    </div>
                )}

                {tab === 'upload' && (
                    <div style={{
                        border: '2px dashed var(--border-subtle)', borderRadius: 12,
                        padding: 32, textAlign: 'center', cursor: 'pointer', marginBottom: 14,
                    }}
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                            e.preventDefault();
                            if (e.dataTransfer.files[0]) {
                                const dt = new DataTransfer();
                                dt.items.add(e.dataTransfer.files[0]);
                                if (fileInputRef.current) {
                                    fileInputRef.current.files = dt.files;
                                    handleFileUpload({ target: { files: dt.files } });
                                }
                            }
                        }}
                    >
                        <Upload size={32} style={{ color: 'var(--accent-primary)', marginBottom: 8 }} />
                        <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 2, fontSize: 14 }}>
                            Drop video file here or click to browse
                        </p>
                        <p style={{ color: 'var(--text-muted)', fontSize: 12, margin: 0 }}>
                            MP4, AVI, MKV, MOV, WebM, MP3, WAV
                        </p>
                        <input ref={fileInputRef} type="file" accept="video/*,audio/*" onChange={handleFileUpload} style={{ display: 'none' }} />
                    </div>
                )}

                {/* Settings Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
                    <div>
                        <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4, fontSize: 12 }}>
                            <Globe size={12} /> Source
                        </label>
                        <select className="input" value={sourceLang} onChange={(e) => setSourceLang(e.target.value)} style={{ fontSize: 13 }}>
                            <option value="auto">🔍 Auto Detect</option>
                            {Object.entries(LANG_MAP).filter(([k]) => k !== 'auto').map(([code, l]) => (
                                <option key={code} value={code}>{l.flag} {l.name}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4, fontSize: 12 }}>
                            <ArrowRightLeft size={12} /> Target
                        </label>
                        <select className="input" value={targetLang} onChange={(e) => setTargetLang(e.target.value)} style={{ fontSize: 13 }}>
                            {Object.entries(LANG_MAP).filter(([k]) => k !== 'auto' && k !== sourceLang).map(([code, l]) => (
                                <option key={code} value={code}>{l.flag} {l.name}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4, fontSize: 12 }}>
                            <FileText size={12} /> Whisper
                        </label>
                        <select className="input" value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)} style={{ fontSize: 13 }}>
                            {WHISPER_MODELS.map((m) => (
                                <option key={m.id} value={m.id}>{m.name} — {m.desc}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4, fontSize: 12 }}>
                            <Languages size={12} /> Model
                        </label>
                        <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ fontSize: 13 }}>
                            <option value="">🤖 Auto</option>
                            {models.map((m) => (
                                <option key={m.id} value={m.id}>{m.alias}</option>
                            ))}
                        </select>
                    </div>

                </div>

                {tab === 'youtube' && (
                    <button className="btn btn-primary" onClick={handleExtract}
                        disabled={isSubmitting || !youtubeUrl.trim()}
                        style={{ width: '100%', padding: '12px 0', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                        {isSubmitting ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
                        {isSubmitting ? 'Starting Pipeline...' : 'Run Pipeline'}
                    </button>
                )}
            </div>

            {/* Pipeline Visualization */}
            {activeJob && (
                <div className="card" style={{ marginBottom: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                        <h3 style={{ margin: 0, color: 'var(--text-primary)', fontSize: 15, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <Film size={16} />
                            {activeJob.video_title || 'Processing...'}
                        </h3>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                            {activeJob.duration ? formatDuration(activeJob.duration) : ''} • {activeJob.segment_count || activeJob.segments?.length || 0} segments
                        </div>
                    </div>

                    {/* Pipeline Nodes */}
                    <div style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        padding: '16px 0', overflowX: 'auto', gap: 0,
                        background: 'var(--bg-primary)', borderRadius: 12, marginBottom: 14,
                        border: '1px solid var(--border-subtle)',
                    }}>
                        {PIPELINE_NODES.map((node, i) => (
                            <PipelineNode
                                key={node.id}
                                node={node}
                                status={activeJob.pipeline?.[node.id]?.status}
                                isLast={i === PIPELINE_NODES.length - 1}
                            />
                        ))}
                    </div>

                    {/* ═══ Pipeline Step Details (Expandable) ═══ */}
                    {activeJob.pipeline && (
                        <details style={{
                            marginBottom: 14, borderRadius: 10,
                            border: '1px solid var(--border-subtle)',
                            background: 'var(--bg-primary)',
                        }}>
                            <summary style={{
                                padding: '10px 14px', cursor: 'pointer', fontSize: 12,
                                fontWeight: 600, color: 'var(--text-secondary)',
                                display: 'flex', alignItems: 'center', gap: 6,
                            }}>
                                📋 Pipeline Step Details (click to expand)
                            </summary>
                            <div style={{ padding: '0 14px 14px', fontSize: 11 }}>
                                {PIPELINE_NODES.map(node => {
                                    const step = activeJob.pipeline[node.id];
                                    if (!step || step.status === 'pending') return null;
                                    const meta = step.meta || {};
                                    return (
                                        <div key={node.id} style={{
                                            padding: '8px 10px', marginBottom: 6, borderRadius: 6,
                                            background: 'var(--bg-secondary)',
                                            border: '1px solid var(--border-subtle)',
                                        }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                                    {node.label}
                                                </span>
                                                <span style={{
                                                    fontSize: 9, padding: '1px 6px', borderRadius: 4,
                                                    background: step.status === 'completed' ? '#22c55e22' : step.status === 'failed' ? '#ef444422' : '#6366f122',
                                                    color: step.status === 'completed' ? '#22c55e' : step.status === 'failed' ? '#ef4444' : '#818cf8',
                                                }}>
                                                    {step.status}
                                                </span>
                                            </div>

                                            {/* Transcribe output */}
                                            {node.id === 'transcribe' && step.status === 'completed' && (
                                                <div style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                                    <div>🎤 Language: <strong>{LANG_MAP[meta.detected_lang]?.name || meta.detected_lang}</strong> (confidence: {Math.round((meta.probability || 0) * 100)}%)</div>
                                                    <div>📊 Segments: <strong>{meta.segment_count}</strong></div>
                                                </div>
                                            )}

                                            {/* Summarize output */}
                                            {node.id === 'summarize' && meta.output && (
                                                <div style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                                    <div>📝 Total: {meta.output.total_segments} segments, {meta.output.total_characters} chars</div>
                                                    {meta.output.key_terms?.length > 0 && (
                                                        <div style={{ marginTop: 4 }}>
                                                            🔑 Key terms: {meta.output.key_terms.slice(0, 15).map(t => (
                                                                <span key={t} style={{
                                                                    display: 'inline-block', padding: '1px 5px', margin: '1px 2px',
                                                                    borderRadius: 3, fontSize: 10,
                                                                    background: '#6366f122', color: '#a5b4fc',
                                                                }}>{t}</span>
                                                            ))}
                                                        </div>
                                                    )}
                                                    {meta.output.opening_context && (
                                                        <div style={{ marginTop: 4, fontStyle: 'italic', color: 'var(--text-muted)' }}>
                                                            💬 "{meta.output.opening_context.slice(0, 200)}..."
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {/* Detect output */}
                                            {node.id === 'detect' && meta.output && (
                                                <div style={{ color: 'var(--text-muted)' }}>
                                                    <div style={{ marginBottom: 4 }}>
                                                        🌍 Distribution: {Object.entries(meta.distribution || {}).map(([lang, count]) => (
                                                            <span key={lang} style={{
                                                                display: 'inline-block', padding: '1px 6px', margin: '1px 2px',
                                                                borderRadius: 4, fontSize: 10,
                                                                background: '#8b5cf622', color: '#c4b5fd',
                                                            }}>{LANG_MAP[lang]?.flag || ''} {lang}: {count}</span>
                                                        ))}
                                                    </div>
                                                    <div style={{ maxHeight: 120, overflowY: 'auto', fontSize: 10, lineHeight: 1.6 }}>
                                                        {meta.output.slice(0, 20).map(d => (
                                                            <div key={d.idx}>
                                                                <span style={{ color: '#818cf8', fontFamily: 'var(--font-mono)' }}>#{d.idx}</span>
                                                                {' '}<span style={{ padding: '0 3px', borderRadius: 2, background: '#6366f133', color: '#a5b4fc', fontSize: 9 }}>{LANG_MAP[d.lang]?.flag || d.lang}</span>
                                                                {' '}{d.text}
                                                            </div>
                                                        ))}
                                                        {meta.output.length > 20 && <div style={{ color: '#78716c' }}>...and {meta.output.length - 20} more</div>}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Translate output */}
                                            {node.id === 'translate' && step.status === 'completed' && (
                                                <div style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                                    <div>🔄 Segments: {meta.segments_translated || 0}</div>
                                                    {meta.methods_used?.map((m, i) => (
                                                        <div key={i}>⚙️ {m}</div>
                                                    ))}
                                                    {meta.local_count > 0 && <div>📦 Local EN→VI: {meta.local_count} segments</div>}
                                                    {meta.nllb_count > 0 && <div>🌐 NLLB/Pivot: {meta.nllb_count} segments</div>}
                                                </div>
                                            )}



                                            {/* Error */}
                                            {step.status === 'failed' && meta.error && (
                                                <div style={{ color: '#ef4444', fontSize: 10 }}>❌ {meta.error}</div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </details>
                    )}

                    {/* Progress Bar */}
                    {activeJob.status !== 'completed' && activeJob.status !== 'failed' && (
                        <div style={{ marginBottom: 14 }}>
                            <div style={{ height: 6, background: 'var(--bg-tertiary)', borderRadius: 3, overflow: 'hidden' }}>
                                <div style={{
                                    height: '100%', borderRadius: 3, transition: 'width 0.5s ease',
                                    width: `${activeJob.progress}%`,
                                    background: 'linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7)',
                                }} />
                            </div>
                            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, textAlign: 'right' }}>
                                {activeJob.progress}%
                            </p>
                        </div>
                    )}

                    {/* Language Distribution */}
                    {activeJob.lang_distribution && Object.keys(activeJob.lang_distribution).length > 0 && (
                        <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
                            {Object.entries(activeJob.lang_distribution).map(([lang, count]) => (
                                <span key={lang} style={{
                                    padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 500,
                                    background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                                    border: '1px solid var(--border-subtle)',
                                }}>
                                    {LANG_MAP[lang]?.flag || lang} {LANG_MAP[lang]?.name || lang}: {count}
                                </span>
                            ))}
                        </div>
                    )}

                    {/* ═══ Live Transcription Preview ═══ */}
                    {activeJob.live_transcriptions && activeJob.live_transcriptions.length > 0 && (
                        <div style={{ marginBottom: 14 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                <h4 style={{ margin: 0, fontSize: 13, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <FileText size={14} style={{ color: '#f59e0b' }} />
                                    Live Transcription
                                    {!activeJob.live_translations || activeJob.live_translations.length === 0 ? (
                                        <span style={{
                                            fontSize: 10, padding: '2px 8px', borderRadius: 10,
                                            background: '#f59e0b22', color: '#f59e0b', fontWeight: 600,
                                            animation: 'pulse 2s infinite',
                                        }}>🎤 LIVE</span>
                                    ) : (
                                        <span style={{
                                            fontSize: 10, padding: '2px 8px', borderRadius: 10,
                                            background: '#22c55e22', color: '#22c55e', fontWeight: 600,
                                        }}>✓ DONE</span>
                                    )}
                                </h4>
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    {activeJob.live_transcriptions.length} segments detected
                                </span>
                            </div>
                            <div style={{
                                maxHeight: 300, overflowY: 'auto', borderRadius: 8,
                                background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)',
                                scrollBehavior: 'smooth',
                            }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-subtle)', position: 'sticky', top: 0, background: 'var(--bg-secondary)', zIndex: 1 }}>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, width: 30 }}>#</th>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, width: 80 }}>Time</th>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>Transcribed Text</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {activeJob.live_transcriptions.map((seg, i) => (
                                            <tr key={i} style={{
                                                borderBottom: '1px solid var(--border-subtle)',
                                                background: i === activeJob.live_transcriptions.length - 1 ? '#f59e0b08' : 'transparent',
                                                transition: 'background 0.3s',
                                            }}>
                                                <td style={{ padding: '4px 10px', fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                    {i + 1}
                                                </td>
                                                <td style={{ padding: '4px 10px', fontSize: 10, color: '#f59e0b', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                                                    {formatDuration(seg.start)}
                                                </td>
                                                <td style={{ padding: '4px 10px', fontSize: 11, color: 'var(--text-primary)' }}>
                                                    {seg.text}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* ═══ Live Translation Preview ═══ */}
                    {activeJob.live_translations && activeJob.live_translations.length > 0 && (
                        <div style={{ marginBottom: 14 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                <h4 style={{ margin: 0, fontSize: 13, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <Languages size={14} style={{ color: 'var(--accent-primary)' }} />
                                    Live Translation
                                    {activeJob.status !== 'completed' && activeJob.status !== 'failed' ? (
                                        <span style={{
                                            fontSize: 10, padding: '2px 8px', borderRadius: 10,
                                            background: '#6366f122', color: '#818cf8', fontWeight: 600,
                                            animation: 'pulse 2s infinite',
                                        }}>LIVE</span>
                                    ) : activeJob.status === 'completed' ? (
                                        <span style={{
                                            fontSize: 10, padding: '2px 8px', borderRadius: 10,
                                            background: '#22c55e22', color: '#22c55e', fontWeight: 600,
                                        }}>✓ DONE</span>
                                    ) : (
                                        <span style={{
                                            fontSize: 10, padding: '2px 8px', borderRadius: 10,
                                            background: '#ef444422', color: '#ef4444', fontWeight: 600,
                                        }}>FAILED</span>
                                    )}
                                </h4>
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    {activeJob.live_translations.filter(s => s.status === 'done').length} / {activeJob.live_translations.length} segments
                                </span>
                            </div>
                            <div ref={liveScrollRef} style={{
                                maxHeight: 500, overflowY: 'auto', borderRadius: 8,
                                background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)',
                                scrollBehavior: 'smooth',
                            }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-subtle)', position: 'sticky', top: 0, background: 'var(--bg-secondary)', zIndex: 1 }}>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, width: 30 }}>#</th>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>Original</th>
                                            <th style={{ padding: '6px 10px', textAlign: 'left', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>
                                                {LANG_MAP[activeJob.target_lang]?.flag} Translation
                                            </th>
                                            <th style={{ padding: '6px 10px', textAlign: 'center', fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, width: 50 }}>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {activeJob.live_translations.map((seg, i) => {
                                            const statusColors = {
                                                done: { bg: '#22c55e22', color: '#22c55e', label: '✓' },
                                                pending: { bg: '#6366f122', color: '#818cf8', label: '↻' },
                                                skip: { bg: '#57534e22', color: '#a8a29e', label: '—' },
                                                error: { bg: '#ef444422', color: '#ef4444', label: '✗' },
                                                fallback: { bg: '#f59e0b22', color: '#f59e0b', label: '⚠' },
                                            };
                                            const sc = statusColors[seg.status] || statusColors.pending;
                                            return (
                                                <tr key={i} style={{
                                                    borderBottom: '1px solid var(--border-subtle)',
                                                    background: seg.status === 'done' ? '#22c55e08' : seg.status === 'pending' ? '#6366f108' : 'transparent',
                                                    transition: 'background 0.3s',
                                                }}>
                                                    <td style={{ padding: '4px 10px', fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                                        {i + 1}
                                                    </td>
                                                    <td style={{ padding: '4px 10px', fontSize: 11, color: 'var(--text-secondary)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                        <span style={{
                                                            fontSize: 8, padding: '0 3px', borderRadius: 2, marginRight: 4,
                                                            background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                                                        }}>{LANG_MAP[seg.lang]?.flag || seg.lang}</span>
                                                        {seg.original}
                                                    </td>
                                                    <td style={{
                                                        padding: '4px 10px', fontSize: 11, fontWeight: seg.status === 'done' ? 500 : 400,
                                                        color: seg.status === 'done' ? 'var(--text-primary)' : seg.status === 'error' ? '#ef4444' : 'var(--text-muted)',
                                                        maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis',
                                                        fontStyle: seg.status === 'pending' ? 'italic' : 'normal',
                                                    }}>
                                                        <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                                            {seg.status === 'pending' ? '⏳ Translating...' : seg.translated || '—'}
                                                        </div>
                                                        {seg.pivot_en && (
                                                            <div style={{
                                                                fontSize: 9, color: '#6366f1', marginTop: 1,
                                                                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                                                fontStyle: 'italic', fontWeight: 400, opacity: 0.7,
                                                            }}>
                                                                🇬🇧 {seg.pivot_en}
                                                            </div>
                                                        )}
                                                        {seg.error && (
                                                            <div style={{
                                                                fontSize: 9, color: '#f87171', marginTop: 1,
                                                                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                                                fontStyle: 'italic', fontWeight: 400,
                                                            }}>
                                                                ⚠ {seg.error}
                                                            </div>
                                                        )}
                                                    </td>
                                                    <td style={{ padding: '4px 10px', textAlign: 'center' }}>
                                                        <span title={seg.error || ''} style={{
                                                            fontSize: 9, padding: '1px 6px', borderRadius: 6,
                                                            background: sc.bg, color: sc.color, fontWeight: 600,
                                                            cursor: seg.error ? 'help' : 'default',
                                                        }}>{sc.label}</span>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                            {/* Error Summary */}
                            {(() => {
                                const errors = activeJob.live_translations.filter(s => s.status === 'error' || s.status === 'fallback');
                                if (errors.length === 0) return null;
                                const reasons = {};
                                errors.forEach(s => {
                                    const reason = s.error || 'Unknown error';
                                    reasons[reason] = (reasons[reason] || 0) + 1;
                                });
                                return (
                                    <div style={{
                                        marginTop: 8, padding: '8px 12px', borderRadius: 8,
                                        background: '#ef444412', border: '1px solid #ef444433',
                                    }}>
                                        <div style={{ fontSize: 11, fontWeight: 600, color: '#f87171', marginBottom: 4 }}>
                                            ⚠ {errors.length} segments failed:
                                        </div>
                                        {Object.entries(reasons).map(([reason, count]) => (
                                            <div key={reason} style={{ fontSize: 10, color: '#fca5a5', paddingLeft: 8 }}>
                                                • {reason} ({count} segments)
                                            </div>
                                        ))}
                                    </div>
                                );
                            })()}
                        </div>
                    )}

                    {/* SRT Preview */}
                    {activeJob.status === 'completed' && activeJob.translated_segments?.length > 0 && (
                        <>
                            <div style={{ display: 'flex', gap: 0, marginBottom: 10, borderBottom: '1px solid var(--border-subtle)' }}>
                                {['original', 'translated', 'side-by-side'].map((view) => (
                                    <button key={view} onClick={() => setSrtView(view)} style={{
                                        padding: '7px 14px', border: 'none', cursor: 'pointer',
                                        background: srtView === view ? 'var(--accent-primary)' : 'transparent',
                                        color: srtView === view ? '#fff' : 'var(--text-secondary)',
                                        borderRadius: '6px 6px 0 0', fontWeight: 500, fontSize: 12,
                                        textTransform: 'capitalize', transition: 'all 0.2s',
                                    }}>
                                        {view === 'side-by-side' ? 'Side by Side' : view}
                                    </button>
                                ))}
                            </div>

                            <div style={{
                                maxHeight: 350, overflow: 'auto', borderRadius: 8,
                                background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)',
                            }}>
                                {srtView === 'side-by-side' && activeJob.translated_segments?.length > 0 ? (
                                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                        <thead>
                                            <tr style={{ borderBottom: '1px solid var(--border-subtle)', position: 'sticky', top: 0, background: 'var(--bg-secondary)', zIndex: 1 }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, width: 60 }}>Time</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Original</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>
                                                    {LANG_MAP[activeJob.target_lang]?.flag} Translated

                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {activeJob.translated_segments.map((seg, i) => (
                                                <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.15s' }}
                                                    onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-tertiary)'}
                                                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                                                    <td style={{ padding: '6px 12px', fontSize: 10, color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', verticalAlign: 'top' }}>
                                                        {formatDuration(seg.start)}
                                                    </td>
                                                    <td style={{ padding: '6px 12px', fontSize: 12, color: 'var(--text-secondary)', verticalAlign: 'top' }}>
                                                        {seg.detected_lang && seg.detected_lang !== 'unknown' && (
                                                            <span style={{
                                                                display: 'inline-block', fontSize: 9, padding: '0px 4px',
                                                                borderRadius: 3, marginRight: 4, verticalAlign: 'middle',
                                                                background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                                                            }}>
                                                                {LANG_MAP[seg.detected_lang]?.flag || seg.detected_lang}
                                                            </span>
                                                        )}
                                                        {seg.original_text}
                                                    </td>
                                                    <td style={{ padding: '6px 12px', fontSize: 12, color: 'var(--text-primary)', fontWeight: 500, verticalAlign: 'top', position: 'relative' }}>
                                                        <div
                                                            contentEditable
                                                            suppressContentEditableWarning
                                                            onBlur={async (e) => {
                                                                const newText = e.currentTarget.textContent.trim();
                                                                if (newText && newText !== seg.text) {
                                                                    try {
                                                                        await submitCorrection({
                                                                            job_id: activeJob.id,
                                                                            segment_idx: i,
                                                                            original_text: seg.original_text,
                                                                            machine_translation: seg.text,
                                                                            corrected_text: newText,
                                                                            source_lang: seg.detected_lang || 'auto',
                                                                            target_lang: activeJob.target_lang || 'vi',
                                                                        });
                                                                        seg.text = newText;
                                                                        seg.user_corrected = true;
                                                                        toast.success(`✅ Saved! (${e.target.closest ? '' : ''}correction #${i + 1})`, { duration: 1500 });
                                                                    } catch (err) {
                                                                        toast.error('Failed to save correction');
                                                                    }
                                                                }
                                                            }}
                                                            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); } }}
                                                            style={{
                                                                outline: 'none', cursor: 'text', borderRadius: 4,
                                                                padding: '2px 4px', margin: '-2px -4px',
                                                                border: '1px solid transparent',
                                                                transition: 'border-color 0.2s, background 0.2s',
                                                            }}
                                                            onFocus={(e) => { e.currentTarget.style.borderColor = '#6366f1'; e.currentTarget.style.background = '#1e1b4b33'; }}
                                                            onBlurCapture={(e) => { e.currentTarget.style.borderColor = 'transparent'; e.currentTarget.style.background = 'transparent'; }}
                                                        >
                                                            {seg.text}
                                                        </div>
                                                        {seg.user_corrected && (
                                                            <span style={{ fontSize: 9, color: '#22c55e', marginLeft: 4 }}>✓ corrected</span>
                                                        )}
                                                        {!seg.user_corrected && (
                                                            <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 4, opacity: 0.5 }}>✏️ click to edit</span>
                                                        )}
                                                        {seg.raw_translation && seg.raw_translation !== seg.text && (
                                                            <div style={{ fontSize: 10, color: '#78716c', marginTop: 2, fontStyle: 'italic' }}>
                                                                MT: {seg.raw_translation}
                                                            </div>
                                                        )}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <pre style={{
                                        padding: 14, margin: 0, fontSize: 12, lineHeight: 1.7,
                                        color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap',
                                    }}>
                                        {srtView === 'translated' && activeJob.translated_srt
                                            ? activeJob.translated_srt
                                            : activeJob.original_srt || 'No content available'}
                                    </pre>
                                )}
                            </div>

                            {/* Download Buttons */}
                            <div style={{ display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
                                <a href={downloadSubtitleFile(activeJob.id, 'original')} className="btn"
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: 5, background: 'var(--bg-tertiary)',
                                        color: 'var(--text-primary)', padding: '8px 16px', borderRadius: 8,
                                        textDecoration: 'none', fontSize: 12, fontWeight: 500, border: '1px solid var(--border-subtle)',
                                    }}>
                                    <Download size={12} /> Original SRT
                                </a>
                                {activeJob.translated_srt && (
                                    <a href={downloadSubtitleFile(activeJob.id, 'translated')} className="btn"
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: 5,
                                            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                                            color: '#fff', padding: '8px 16px', borderRadius: 8,
                                            textDecoration: 'none', fontSize: 12, fontWeight: 600,
                                        }}>
                                        <Download size={12} /> Translated SRT
                                    </a>
                                )}
                                {activeJob.translated_srt && (
                                    <a href={downloadSubtitleFile(activeJob.id, 'bilingual')} className="btn"
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: 5, background: 'var(--bg-tertiary)',
                                            color: 'var(--text-primary)', padding: '8px 16px', borderRadius: 8,
                                            textDecoration: 'none', fontSize: 12, fontWeight: 500, border: '1px solid var(--border-subtle)',
                                        }}>
                                        <Download size={12} /> Bilingual SRT
                                    </a>
                                )}
                            </div>
                        </>
                    )}

                    {/* Error */}
                    {activeJob.status === 'failed' && activeJob.error && (
                        <div style={{
                            padding: 14, borderRadius: 8, background: '#ef444422',
                            color: '#ef4444', fontSize: 12, marginTop: 10,
                            display: 'flex', alignItems: 'center', gap: 6,
                        }}>
                            <AlertCircle size={14} /> {activeJob.error}
                        </div>
                    )}
                </div>
            )}

            {/* Recent Jobs */}
            {jobs.length > 0 && (
                <div className="card">
                    <h3 style={{ margin: '0 0 12px', color: 'var(--text-primary)', fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <Clock size={14} /> Recent Jobs
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {jobs.map((job) => (
                            <div key={job.id} onClick={() => viewJob(job)} style={{
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                padding: '10px 14px', borderRadius: 8, cursor: 'pointer',
                                background: activeJob?.id === job.id ? 'var(--bg-tertiary)' : 'var(--bg-primary)',
                                border: `1px solid ${activeJob?.id === job.id ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                transition: 'all 0.2s',
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                    <Film size={14} style={{ color: 'var(--accent-primary)' }} />
                                    <div>
                                        <div style={{ fontWeight: 500, fontSize: 13, color: 'var(--text-primary)' }}>
                                            {job.video_title || `Job ${job.id}`}
                                        </div>
                                        <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 10, marginTop: 1 }}>
                                            <span>{LANG_MAP[job.source_lang]?.flag} → {LANG_MAP[job.target_lang]?.flag}</span>
                                            {job.duration && <span>{formatDuration(job.duration)}</span>}
                                            <span>{job.segments?.length || 0} seg</span>
                                        </div>
                                    </div>
                                </div>
                                <div style={{
                                    padding: '3px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                                    background: job.status === 'completed' ? '#22c55e22' : job.status === 'failed' ? '#ef444422' : '#6366f122',
                                    color: job.status === 'completed' ? '#22c55e' : job.status === 'failed' ? '#ef4444' : '#6366f1',
                                }}>
                                    {job.status === 'completed' ? '✓ Done' : job.status === 'failed' ? '✗ Failed' : '⟳ Running'}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
