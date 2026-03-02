import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    Zap,
    Play,
    Square,
    X,
    Settings2,
    BarChart3,
    Clock,
    CheckCircle,
    AlertTriangle,
    Loader2,
    Brain,
    Database,
    TrendingDown,
} from 'lucide-react';
import {
    listModels,
    listDatasets,
    listTrainingJobs,
    startTraining,
    stopTraining,
    getLanguages,
} from '../api';

export default function TrainingPage() {
    const [models, setModels] = useState([]);
    const [datasets, setDatasets] = useState([]);
    const [jobs, setJobs] = useState([]);
    const [languages, setLanguages] = useState([]);
    const [showNew, setShowNew] = useState(false);
    const [activeTab, setActiveTab] = useState('all');

    // Training form
    const [formModelId, setFormModelId] = useState('');
    const [formDatasetId, setFormDatasetId] = useState('');
    const [formSrcLang, setFormSrcLang] = useState('en');
    const [formTgtLang, setFormTgtLang] = useState('vi');
    const [formEpochs, setFormEpochs] = useState(3);
    const [formBatchSize, setFormBatchSize] = useState(8);
    const [formLR, setFormLR] = useState('5e-5');
    const [formMaxLen, setFormMaxLen] = useState(128);
    const [formWarmup, setFormWarmup] = useState(500);
    const [formUseLora, setFormUseLora] = useState(true);
    const [formLoraR, setFormLoraR] = useState(16);
    const [formLoraAlpha, setFormLoraAlpha] = useState(32);
    const [loading, setLoading] = useState(false);

    const fetchAll = async () => {
        try {
            const [m, d, j, l] = await Promise.allSettled([
                listModels(),
                listDatasets(),
                listTrainingJobs(),
                getLanguages(),
            ]);
            if (m.status === 'fulfilled') setModels(m.value.data.models.filter((x) => x.status === 'ready'));
            if (d.status === 'fulfilled') setDatasets(d.value.data.datasets.filter((x) => x.status === 'ready'));
            if (j.status === 'fulfilled') setJobs(j.value.data.jobs);
            if (l.status === 'fulfilled') setLanguages(l.value.data.languages);
        } catch { /* */ }
    };

    useEffect(() => {
        fetchAll();
        const interval = setInterval(fetchAll, 4000);
        return () => clearInterval(interval);
    }, []);

    const handleStartTraining = async () => {
        if (!formModelId || !formDatasetId) {
            toast.error('Select a model and dataset');
            return;
        }
        try {
            setLoading(true);
            await startTraining({
                model_id: formModelId,
                dataset_id: formDatasetId,
                source_lang: formSrcLang,
                target_lang: formTgtLang,
                num_epochs: formEpochs,
                batch_size: formBatchSize,
                learning_rate: parseFloat(formLR),
                max_length: formMaxLen,
                warmup_steps: formWarmup,
                use_lora: formUseLora,
                lora_r: formLoraR,
                lora_alpha: formLoraAlpha,
            });
            toast.success('Training job started!');
            setShowNew(false);
            fetchAll();
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to start training');
        } finally {
            setLoading(false);
        }
    };

    const handleStop = async (jobId) => {
        try {
            await stopTraining(jobId);
            toast.success('Stop signal sent');
            fetchAll();
        } catch {
            toast.error('Failed to stop');
        }
    };

    const filteredJobs =
        activeTab === 'all'
            ? jobs
            : activeTab === 'active'
                ? jobs.filter((j) => ['training', 'initializing', 'loading_model', 'loading_dataset', 'tokenizing', 'saving'].includes(j.status))
                : jobs.filter((j) => j.status === 'completed');

    const LangSelect = ({ value, onChange, label }) => (
        <div className="input-group">
            <label className="input-label">{label}</label>
            <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
                {languages.length > 0
                    ? languages.map((l) => <option key={l.code} value={l.code}>{l.flag} {l.name}</option>)
                    : <>
                        <option value="en">🇺🇸 English</option>
                        <option value="vi">🇻🇳 Vietnamese</option>
                        <option value="zh">🇨🇳 Chinese</option>
                        <option value="ja">🇯🇵 Japanese</option>
                    </>
                }
            </select>
        </div>
    );

    return (
        <div>
            <motion.div className="page-header" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                <h2>Training Pipeline</h2>
                <p>Fine-tune translation models with LoRA or full training on your datasets</p>
            </motion.div>

            {/* Actions */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-lg)' }}>
                <div className="tabs" style={{ marginBottom: 0, borderBottom: 'none' }}>
                    {['all', 'active', 'completed'].map((tab) => (
                        <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                            {tab === 'active' && (
                                <span style={{ marginLeft: 6, fontSize: 11, opacity: 0.7 }}>
                                    ({jobs.filter((j) => ['training', 'initializing', 'loading_model', 'loading_dataset', 'tokenizing', 'saving'].includes(j.status)).length})
                                </span>
                            )}
                        </button>
                    ))}
                </div>
                <button
                    className="btn btn-primary"
                    onClick={() => setShowNew(true)}
                    disabled={models.length === 0 || datasets.length === 0}
                >
                    <Play size={16} /> New Training Job
                </button>
            </div>

            {/* Jobs List */}
            {filteredJobs.length === 0 ? (
                <motion.div className="empty-state" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="empty-state-icon"><Zap size={36} /></div>
                    <h3>No Training Jobs</h3>
                    <p>
                        {models.length === 0 || datasets.length === 0
                            ? 'Download a model and dataset first, then start a training job.'
                            : 'Start a new training job to fine-tune a model for translation.'}
                    </p>
                </motion.div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                    <AnimatePresence>
                        {filteredJobs.map((job, i) => (
                            <motion.div
                                key={job.id}
                                className="card"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                transition={{ delay: i * 0.05 }}
                            >
                                {/* Job Header */}
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 'var(--space-md)' }}>
                                    <div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 4 }}>
                                            <Zap size={16} style={{ color: 'var(--accent-amber)' }} />
                                            <span style={{ fontWeight: 600, fontSize: 16 }}>Job {job.id}</span>
                                            <JobStatusBadge status={job.status} />
                                        </div>
                                        <div style={{ fontSize: 13, color: 'var(--text-secondary)', display: 'flex', gap: 'var(--space-lg)', flexWrap: 'wrap' }}>
                                            <span><Brain size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />{job.model_name}</span>
                                            <span><Database size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />{job.dataset_name}</span>
                                            <span style={{ fontWeight: 500 }}>{job.source_lang} → {job.target_lang}</span>
                                        </div>
                                    </div>
                                    {['training', 'initializing', 'loading_model', 'loading_dataset', 'tokenizing'].includes(job.status) && (
                                        <button className="btn btn-danger btn-sm" onClick={() => handleStop(job.id)}>
                                            <Square size={14} /> Stop
                                        </button>
                                    )}
                                </div>

                                {/* Progress */}
                                {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped' && (
                                    <div style={{ marginBottom: 'var(--space-md)' }}>
                                        <div className="progress" style={{ height: 10 }}>
                                            <div className="progress-bar" style={{ width: `${job.progress}%` }} />
                                        </div>
                                        <div className="progress-label">
                                            <span className="progress-text">
                                                {job.status === 'training'
                                                    ? `Step ${job.current_step} / ${job.total_steps} · Epoch ${Math.floor(job.current_epoch)}`
                                                    : job.status.replace('_', ' ')}
                                            </span>
                                            <span className="progress-value">{job.progress}%</span>
                                        </div>
                                    </div>
                                )}

                                {/* Metrics */}
                                <div style={{ display: 'flex', gap: 'var(--space-xl)', flexWrap: 'wrap', marginBottom: job.error ? 'var(--space-md)' : 0 }}>
                                    <MetricItem label="Train Loss" value={job.train_loss} icon={TrendingDown} color="var(--accent-amber)" />
                                    <MetricItem label="Eval Loss" value={job.eval_loss} icon={BarChart3} color="var(--accent-cyan)" />
                                    <MetricItem label="Epoch" value={job.current_epoch ? Math.floor(job.current_epoch) : '—'} icon={Clock} color="var(--accent-indigo-light)" />
                                    <MetricItem label="Steps" value={job.current_step || '—'} icon={Zap} color="var(--accent-emerald)" />
                                </div>

                                {/* Loss Chart */}
                                {job.loss_history && job.loss_history.length > 0 && (
                                    <div style={{ marginTop: 'var(--space-md)' }}>
                                        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginBottom: 'var(--space-sm)' }}>Loss History</div>
                                        <div className="loss-chart" style={{ height: 100 }}>
                                            {job.loss_history.map((h, idx) => {
                                                const maxLoss = Math.max(...job.loss_history.map((x) => x.loss));
                                                const height = maxLoss > 0 ? (h.loss / maxLoss) * 100 : 0;
                                                return (
                                                    <div
                                                        key={idx}
                                                        className="loss-chart-bar"
                                                        style={{ height: `${height}%` }}
                                                        title={`Step ${h.step}: ${h.loss}`}
                                                    />
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {/* Error */}
                                {job.error && (
                                    <div style={{ background: 'rgba(251,113,133,0.1)', border: '1px solid rgba(251,113,133,0.2)', borderRadius: 'var(--radius-md)', padding: 'var(--space-sm) var(--space-md)', fontSize: 12, color: 'var(--accent-rose)', wordBreak: 'break-word' }}>
                                        <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                        {job.error}
                                    </div>
                                )}

                                {/* Config toggle */}
                                <details style={{ marginTop: 'var(--space-md)' }}>
                                    <summary style={{ fontSize: 12, color: 'var(--text-tertiary)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
                                        <Settings2 size={12} /> Training Configuration
                                    </summary>
                                    <div style={{ marginTop: 'var(--space-sm)', padding: 'var(--space-md)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 'var(--space-sm)' }}>
                                        {Object.entries(job.config || {}).map(([key, val]) => (
                                            <div key={key}>
                                                <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{key}</span>
                                                <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{String(val)}</div>
                                            </div>
                                        ))}
                                    </div>
                                </details>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            )}

            {/* New Training Modal */}
            <AnimatePresence>
                {showNew && (
                    <div className="modal-overlay" onClick={() => setShowNew(false)}>
                        <motion.div className="modal" style={{ maxWidth: 650 }} onClick={(e) => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
                            <div className="modal-header">
                                <span className="modal-title">
                                    <Zap size={18} style={{ marginRight: 8, verticalAlign: 'middle', color: 'var(--accent-amber)' }} />
                                    New Training Job
                                </span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowNew(false)}><X size={18} /></button>
                            </div>
                            <div className="modal-body">
                                {/* Model & Dataset */}
                                <div className="form-row">
                                    <div className="input-group">
                                        <label className="input-label"><Brain size={12} style={{ marginRight: 4 }} />Model <span className="required">*</span></label>
                                        <select className="select" value={formModelId} onChange={(e) => setFormModelId(e.target.value)}>
                                            <option value="">Select a model</option>
                                            {models.map((m) => <option key={m.id} value={m.id}>{m.alias}</option>)}
                                        </select>
                                    </div>
                                    <div className="input-group">
                                        <label className="input-label"><Database size={12} style={{ marginRight: 4 }} />Dataset <span className="required">*</span></label>
                                        <select className="select" value={formDatasetId} onChange={(e) => setFormDatasetId(e.target.value)}>
                                            <option value="">Select a dataset</option>
                                            {datasets.map((d) => <option key={d.id} value={d.id}>{d.hf_name}</option>)}
                                        </select>
                                    </div>
                                </div>

                                {/* Languages */}
                                <div className="form-row">
                                    <LangSelect label="Source Language" value={formSrcLang} onChange={setFormSrcLang} />
                                    <LangSelect label="Target Language" value={formTgtLang} onChange={setFormTgtLang} />
                                </div>

                                {/* Training params */}
                                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginTop: 'var(--space-sm)' }}>
                                    <Settings2 size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                    Training Parameters
                                </div>

                                <div className="form-row">
                                    <div className="input-group">
                                        <label className="input-label">Epochs</label>
                                        <input className="input" type="number" value={formEpochs} onChange={(e) => setFormEpochs(parseInt(e.target.value) || 1)} min={1} max={100} />
                                    </div>
                                    <div className="input-group">
                                        <label className="input-label">Batch Size</label>
                                        <input className="input" type="number" value={formBatchSize} onChange={(e) => setFormBatchSize(parseInt(e.target.value) || 1)} min={1} />
                                    </div>
                                </div>

                                <div className="form-row">
                                    <div className="input-group">
                                        <label className="input-label">Learning Rate</label>
                                        <input className="input" value={formLR} onChange={(e) => setFormLR(e.target.value)} placeholder="5e-5" />
                                    </div>
                                    <div className="input-group">
                                        <label className="input-label">Max Length</label>
                                        <input className="input" type="number" value={formMaxLen} onChange={(e) => setFormMaxLen(parseInt(e.target.value) || 64)} />
                                    </div>
                                </div>

                                <div className="form-row">
                                    <div className="input-group">
                                        <label className="input-label">Warmup Steps</label>
                                        <input className="input" type="number" value={formWarmup} onChange={(e) => setFormWarmup(parseInt(e.target.value) || 0)} />
                                    </div>
                                    <div className="input-group" />
                                </div>

                                {/* LoRA */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', marginTop: 'var(--space-sm)' }}>
                                    <div
                                        className={`toggle ${formUseLora ? 'active' : ''}`}
                                        onClick={() => setFormUseLora(!formUseLora)}
                                    />
                                    <span style={{ fontSize: 14, fontWeight: 500 }}>Use LoRA (Parameter-Efficient Fine-Tuning)</span>
                                </div>

                                {formUseLora && (
                                    <div className="form-row" style={{ marginTop: 'var(--space-sm)' }}>
                                        <div className="input-group">
                                            <label className="input-label">LoRA Rank (r)</label>
                                            <input className="input" type="number" value={formLoraR} onChange={(e) => setFormLoraR(parseInt(e.target.value) || 8)} />
                                        </div>
                                        <div className="input-group">
                                            <label className="input-label">LoRA Alpha</label>
                                            <input className="input" type="number" value={formLoraAlpha} onChange={(e) => setFormLoraAlpha(parseInt(e.target.value) || 16)} />
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div className="modal-footer">
                                <button className="btn btn-secondary" onClick={() => setShowNew(false)}>Cancel</button>
                                <button className="btn btn-success" disabled={!formModelId || !formDatasetId || loading} onClick={handleStartTraining}>
                                    {loading ? <span className="spinner" /> : <Play size={16} />} Start Training
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}

function JobStatusBadge({ status }) {
    const map = {
        initializing: { cls: 'badge-info', label: 'Initializing' },
        loading_model: { cls: 'badge-info', label: 'Loading Model' },
        loading_dataset: { cls: 'badge-info', label: 'Loading Data' },
        tokenizing: { cls: 'badge-info', label: 'Tokenizing' },
        training: { cls: 'badge-warning', label: 'Training' },
        saving: { cls: 'badge-info', label: 'Saving' },
        completed: { cls: 'badge-success', label: 'Completed' },
        failed: { cls: 'badge-danger', label: 'Failed' },
        stopped: { cls: 'badge-neutral', label: 'Stopped' },
    };
    const info = map[status] || { cls: 'badge-neutral', label: status };
    return <span className={`badge ${info.cls}`}>{info.label}</span>;
}

function MetricItem({ label, value, icon: Icon, color }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <Icon size={14} style={{ color, flexShrink: 0 }} />
            <div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{label}</div>
                <div style={{ fontSize: 15, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                    {value ?? '—'}
                </div>
            </div>
        </div>
    );
}
