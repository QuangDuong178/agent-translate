import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    Brain,
    Download,
    Trash2,
    Search,
    Plus,
    X,
    ExternalLink,
    Server,
    Globe,
    Loader2,
    CheckCircle,
    AlertTriangle,
} from 'lucide-react';
import { listModels, downloadModel, deleteModel, getCatalogModels } from '../api';

export default function ModelsPage() {
    const [models, setModels] = useState([]);
    const [catalog, setCatalog] = useState([]);
    const [showCatalog, setShowCatalog] = useState(false);
    const [showCustom, setShowCustom] = useState(false);
    const [customModelName, setCustomModelName] = useState('');
    const [customAlias, setCustomAlias] = useState('');
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(false);

    const fetchModels = async () => {
        try {
            const res = await listModels();
            setModels(res.data.models);
        } catch { /* offline */ }
    };

    const fetchCatalog = async () => {
        try {
            const res = await getCatalogModels();
            setCatalog(res.data.models);
        } catch { /* offline */ }
    };

    useEffect(() => {
        fetchModels();
        fetchCatalog();
        const interval = setInterval(fetchModels, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleDownload = async (modelName, alias) => {
        try {
            setLoading(true);
            await downloadModel({ model_name: modelName, alias: alias || undefined });
            toast.success(`Started downloading ${modelName}`);
            setShowCatalog(false);
            setShowCustom(false);
            fetchModels();
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Failed to start download');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id, name) => {
        if (!confirm(`Delete model "${name}"? This cannot be undone.`)) return;
        try {
            await deleteModel(id);
            toast.success('Model deleted');
            fetchModels();
        } catch (err) {
            toast.error('Failed to delete model');
        }
    };

    const filtered = models.filter(
        (m) =>
            m.alias.toLowerCase().includes(search.toLowerCase()) ||
            m.hf_name.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div>
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h2>Model Management</h2>
                <p>Download, manage, and organize your translation models from HuggingFace Hub</p>
            </motion.div>

            {/* Actions Bar */}
            <div
                style={{
                    display: 'flex',
                    gap: 'var(--space-md)',
                    alignItems: 'center',
                    marginBottom: 'var(--space-xl)',
                    flexWrap: 'wrap',
                }}
            >
                <div style={{ position: 'relative', flex: 1, minWidth: 240 }}>
                    <Search
                        size={16}
                        style={{
                            position: 'absolute',
                            left: 12,
                            top: '50%',
                            transform: 'translateY(-50%)',
                            color: 'var(--text-tertiary)',
                        }}
                    />
                    <input
                        className="input"
                        placeholder="Search models..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        style={{ paddingLeft: 36 }}
                    />
                </div>
                <button className="btn btn-primary" onClick={() => setShowCatalog(true)}>
                    <Download size={16} /> Browse Catalog
                </button>
                <button className="btn btn-secondary" onClick={() => setShowCustom(true)}>
                    <Plus size={16} /> Custom Model
                </button>
            </div>

            {/* Models Grid */}
            {filtered.length === 0 ? (
                <motion.div
                    className="empty-state"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <div className="empty-state-icon">
                        <Brain size={36} />
                    </div>
                    <h3>No Models Yet</h3>
                    <p>
                        Download your first model from the catalog or enter a custom HuggingFace model ID to get started.
                    </p>
                    <button
                        className="btn btn-primary"
                        style={{ marginTop: 'var(--space-lg)' }}
                        onClick={() => setShowCatalog(true)}
                    >
                        <Download size={16} /> Browse Model Catalog
                    </button>
                </motion.div>
            ) : (
                <div className="grid-auto">
                    <AnimatePresence>
                        {filtered.map((model, i) => (
                            <motion.div
                                key={model.id}
                                className="card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ delay: i * 0.05 }}
                            >
                                <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between', marginBottom: 'var(--space-md)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                                        <div className="card-title-icon indigo">
                                            <Brain size={18} />
                                        </div>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: 15 }}>{model.alias}</div>
                                            <div style={{ fontSize: 12, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>{model.hf_name}</div>
                                        </div>
                                    </div>
                                    <ModelStatusBadge status={model.status} />
                                </div>

                                {model.status === 'downloading' && (
                                    <div style={{ marginBottom: 'var(--space-md)' }}>
                                        <div className="progress">
                                            <div className="progress-bar" style={{ width: `${model.progress}%` }} />
                                        </div>
                                        <div className="progress-label">
                                            <span className="progress-text">Downloading...</span>
                                            <span className="progress-value">{model.progress}%</span>
                                        </div>
                                    </div>
                                )}

                                {model.error && (
                                    <div
                                        style={{
                                            background: 'rgba(251, 113, 133, 0.1)',
                                            border: '1px solid rgba(251, 113, 133, 0.2)',
                                            borderRadius: 'var(--radius-md)',
                                            padding: 'var(--space-sm) var(--space-md)',
                                            fontSize: 12,
                                            color: 'var(--accent-rose)',
                                            marginBottom: 'var(--space-md)',
                                            wordBreak: 'break-word',
                                        }}
                                    >
                                        <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                        {model.error}
                                    </div>
                                )}

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                                        {model.size_gb ? `${model.size_gb} GB` : '—'}
                                    </div>
                                    <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                                        <button
                                            className="btn btn-ghost btn-icon"
                                            title="View on HuggingFace"
                                            onClick={() => window.open(`https://huggingface.co/${model.hf_name}`, '_blank')}
                                        >
                                            <ExternalLink size={14} />
                                        </button>
                                        <button
                                            className="btn btn-ghost btn-icon"
                                            title="Delete model"
                                            onClick={() => handleDelete(model.id, model.alias)}
                                            style={{ color: 'var(--accent-rose)' }}
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            )}

            {/* Catalog Modal */}
            <AnimatePresence>
                {showCatalog && (
                    <div className="modal-overlay" onClick={() => setShowCatalog(false)}>
                        <motion.div
                            className="modal"
                            style={{ maxWidth: 700 }}
                            onClick={(e) => e.stopPropagation()}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                        >
                            <div className="modal-header">
                                <span className="modal-title">
                                    <Server size={18} style={{ marginRight: 8, verticalAlign: 'middle', color: 'var(--accent-indigo-light)' }} />
                                    Model Catalog
                                </span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowCatalog(false)}>
                                    <X size={18} />
                                </button>
                            </div>
                            <div className="modal-body">
                                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 'var(--space-md)' }}>
                                    Select a pre-configured model to download from HuggingFace Hub.
                                </p>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                                    {catalog.map((cat) => (
                                        <div
                                            key={cat.id}
                                            className="catalog-card"
                                            onClick={() => handleDownload(cat.id, cat.name)}
                                        >
                                            <div className="catalog-card-name">{cat.name}</div>
                                            <div className="catalog-card-id">{cat.id}</div>
                                            <div className="catalog-card-desc">{cat.description}</div>
                                            <div className="catalog-card-meta">
                                                <span className="badge badge-primary">{cat.size}</span>
                                                <span className="badge badge-info">
                                                    <Globe size={10} /> {cat.languages} lang{cat.languages > 1 ? 's' : ''}
                                                </span>
                                                <span className="badge badge-neutral">{cat.type}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Custom Model Modal */}
            <AnimatePresence>
                {showCustom && (
                    <div className="modal-overlay" onClick={() => setShowCustom(false)}>
                        <motion.div
                            className="modal"
                            onClick={(e) => e.stopPropagation()}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                        >
                            <div className="modal-header">
                                <span className="modal-title">Download Custom Model</span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowCustom(false)}>
                                    <X size={18} />
                                </button>
                            </div>
                            <div className="modal-body">
                                <div className="input-group">
                                    <label className="input-label">
                                        HuggingFace Model ID <span className="required">*</span>
                                    </label>
                                    <input
                                        className="input"
                                        placeholder="e.g. facebook/mbart-large-50-many-to-many-mmt"
                                        value={customModelName}
                                        onChange={(e) => setCustomModelName(e.target.value)}
                                    />
                                </div>
                                <div className="input-group">
                                    <label className="input-label">Alias (optional)</label>
                                    <input
                                        className="input"
                                        placeholder="e.g. mBART-50"
                                        value={customAlias}
                                        onChange={(e) => setCustomAlias(e.target.value)}
                                    />
                                </div>
                            </div>
                            <div className="modal-footer">
                                <button className="btn btn-secondary" onClick={() => setShowCustom(false)}>
                                    Cancel
                                </button>
                                <button
                                    className="btn btn-primary"
                                    disabled={!customModelName || loading}
                                    onClick={() => handleDownload(customModelName, customAlias)}
                                >
                                    {loading ? <Loader2 size={16} className="spinner" /> : <Download size={16} />}
                                    Download
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}

function ModelStatusBadge({ status }) {
    const map = {
        ready: { cls: 'badge-success', icon: CheckCircle, label: 'Ready' },
        downloading: { cls: 'badge-info', icon: Loader2, label: 'Downloading' },
        error: { cls: 'badge-danger', icon: AlertTriangle, label: 'Error' },
    };
    const info = map[status] || { cls: 'badge-neutral', icon: null, label: status };
    const Icon = info.icon;
    return (
        <span className={`badge ${info.cls}`}>
            {Icon && <Icon size={12} />}
            {info.label}
        </span>
    );
}
