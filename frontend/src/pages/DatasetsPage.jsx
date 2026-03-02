import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    Database,
    Download,
    Upload,
    Trash2,
    Search,
    X,
    Globe,
    FileText,
    Loader2,
    CheckCircle,
    AlertTriangle,
    Layers,
} from 'lucide-react';
import {
    listDatasets,
    downloadDataset,
    uploadDataset,
    deleteDataset,
    getCatalogDatasets,
    getLanguages,
} from '../api';

export default function DatasetsPage() {
    const [datasets, setDatasets] = useState([]);
    const [catalog, setCatalog] = useState([]);
    const [languages, setLanguages] = useState([]);
    const [showCatalog, setShowCatalog] = useState(false);
    const [showCustom, setShowCustom] = useState(false);
    const [showUpload, setShowUpload] = useState(false);
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(false);

    // Custom download form
    const [customName, setCustomName] = useState('');
    const [customSrcLang, setCustomSrcLang] = useState('en');
    const [customTgtLang, setCustomTgtLang] = useState('vi');
    const [customSplit, setCustomSplit] = useState('train');
    const [customMaxSamples, setCustomMaxSamples] = useState('');

    // Upload form
    const [uploadName, setUploadName] = useState('');
    const [uploadFile, setUploadFile] = useState(null);
    const [uploadSrcLang, setUploadSrcLang] = useState('en');
    const [uploadTgtLang, setUploadTgtLang] = useState('vi');

    const fetchDatasets = async () => {
        try {
            const res = await listDatasets();
            setDatasets(res.data.datasets);
        } catch { /* */ }
    };

    useEffect(() => {
        fetchDatasets();
        getCatalogDatasets()
            .then((r) => setCatalog(r.data.datasets))
            .catch(() => { });
        getLanguages()
            .then((r) => setLanguages(r.data.languages))
            .catch(() => { });
        const interval = setInterval(fetchDatasets, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleCatalogDownload = async (cat) => {
        try {
            setLoading(true);
            await downloadDataset({
                dataset_name: cat.id,
                source_lang: 'en',
                target_lang: 'vi',
                split: 'train',
                max_samples: 10000,
            });
            toast.success(`Started downloading ${cat.name}`);
            setShowCatalog(false);
            fetchDatasets();
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Download failed');
        } finally {
            setLoading(false);
        }
    };

    const handleCustomDownload = async () => {
        if (!customName) return;
        try {
            setLoading(true);
            await downloadDataset({
                dataset_name: customName,
                source_lang: customSrcLang,
                target_lang: customTgtLang,
                split: customSplit,
                max_samples: customMaxSamples ? parseInt(customMaxSamples) : null,
            });
            toast.success('Download started');
            setShowCustom(false);
            fetchDatasets();
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Download failed');
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async () => {
        if (!uploadFile || !uploadName) return;
        try {
            setLoading(true);
            const formData = new FormData();
            formData.append('name', uploadName);
            formData.append('source_lang', uploadSrcLang);
            formData.append('target_lang', uploadTgtLang);
            formData.append('file', uploadFile);
            await uploadDataset(formData);
            toast.success('Dataset uploaded');
            setShowUpload(false);
            setUploadFile(null);
            setUploadName('');
            fetchDatasets();
        } catch (err) {
            toast.error('Upload failed');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id, name) => {
        if (!confirm(`Delete dataset "${name}"?`)) return;
        try {
            await deleteDataset(id);
            toast.success('Dataset deleted');
            fetchDatasets();
        } catch {
            toast.error('Failed to delete');
        }
    };

    const filtered = datasets.filter(
        (d) =>
            d.hf_name.toLowerCase().includes(search.toLowerCase()) ||
            d.source_lang.includes(search.toLowerCase()) ||
            d.target_lang.includes(search.toLowerCase())
    );

    const LangSelect = ({ value, onChange, label }) => (
        <div className="input-group">
            <label className="input-label">{label}</label>
            <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
                {languages.length > 0
                    ? languages.map((l) => (
                        <option key={l.code} value={l.code}>
                            {l.flag} {l.name} ({l.code})
                        </option>
                    ))
                    : <>
                        <option value="en">🇺🇸 English</option>
                        <option value="vi">🇻🇳 Vietnamese</option>
                        <option value="zh">🇨🇳 Chinese</option>
                        <option value="ja">🇯🇵 Japanese</option>
                        <option value="ko">🇰🇷 Korean</option>
                        <option value="fr">🇫🇷 French</option>
                        <option value="de">🇩🇪 German</option>
                        <option value="es">🇪🇸 Spanish</option>
                    </>
                }
            </select>
        </div>
    );

    return (
        <div>
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h2>Dataset Management</h2>
                <p>Download, upload, and manage translation datasets for training</p>
            </motion.div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: 'var(--space-md)', alignItems: 'center', marginBottom: 'var(--space-xl)', flexWrap: 'wrap' }}>
                <div style={{ position: 'relative', flex: 1, minWidth: 240 }}>
                    <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-tertiary)' }} />
                    <input className="input" placeholder="Search datasets..." value={search} onChange={(e) => setSearch(e.target.value)} style={{ paddingLeft: 36 }} />
                </div>
                <button className="btn btn-primary" onClick={() => setShowCatalog(true)}>
                    <Download size={16} /> Browse Catalog
                </button>
                <button className="btn btn-secondary" onClick={() => setShowCustom(true)}>
                    <Globe size={16} /> Custom Download
                </button>
                <button className="btn btn-secondary" onClick={() => setShowUpload(true)}>
                    <Upload size={16} /> Upload File
                </button>
            </div>

            {/* Datasets Grid */}
            {filtered.length === 0 ? (
                <motion.div className="empty-state" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="empty-state-icon"><Database size={36} /></div>
                    <h3>No Datasets Yet</h3>
                    <p>Download translation datasets from the catalog, enter a custom HuggingFace ID, or upload your own data.</p>
                    <button className="btn btn-primary" style={{ marginTop: 'var(--space-lg)' }} onClick={() => setShowCatalog(true)}>
                        <Download size={16} /> Browse Dataset Catalog
                    </button>
                </motion.div>
            ) : (
                <div className="grid-auto">
                    <AnimatePresence>
                        {filtered.map((ds, i) => (
                            <motion.div
                                key={ds.id}
                                className="card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ delay: i * 0.05 }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 'var(--space-md)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                                        <div className="card-title-icon cyan"><Database size={18} /></div>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: 15 }}>{ds.hf_name}</div>
                                            <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                                                {ds.source_lang} → {ds.target_lang} · {ds.split}
                                            </div>
                                        </div>
                                    </div>
                                    <DatasetStatusBadge status={ds.status} />
                                </div>

                                {ds.status === 'downloading' && (
                                    <div style={{ marginBottom: 'var(--space-md)' }}>
                                        <div className="progress">
                                            <div className="progress-bar" style={{ width: `${ds.progress}%` }} />
                                        </div>
                                        <div className="progress-label">
                                            <span className="progress-text">Downloading...</span>
                                            <span className="progress-value">{ds.progress}%</span>
                                        </div>
                                    </div>
                                )}

                                {ds.error && (
                                    <div style={{ background: 'rgba(251,113,133,0.1)', border: '1px solid rgba(251,113,133,0.2)', borderRadius: 'var(--radius-md)', padding: 'var(--space-sm) var(--space-md)', fontSize: 12, color: 'var(--accent-rose)', marginBottom: 'var(--space-md)', wordBreak: 'break-word' }}>
                                        <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                        {ds.error}
                                    </div>
                                )}

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                                        {ds.num_samples && (
                                            <span className="badge badge-primary">
                                                <Layers size={10} /> {ds.num_samples.toLocaleString()} samples
                                            </span>
                                        )}
                                    </div>
                                    <button className="btn btn-ghost btn-icon" onClick={() => handleDelete(ds.id, ds.hf_name)} style={{ color: 'var(--accent-rose)' }}>
                                        <Trash2 size={14} />
                                    </button>
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
                        <motion.div className="modal" style={{ maxWidth: 700 }} onClick={(e) => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
                            <div className="modal-header">
                                <span className="modal-title"><FileText size={18} style={{ marginRight: 8, verticalAlign: 'middle', color: 'var(--accent-cyan)' }} />Dataset Catalog</span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowCatalog(false)}><X size={18} /></button>
                            </div>
                            <div className="modal-body">
                                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 'var(--space-md)' }}>
                                    Popular translation datasets. Click to download (first 10,000 samples by default).
                                </p>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                                    {catalog.map((cat) => (
                                        <div key={cat.id} className="catalog-card" onClick={() => handleCatalogDownload(cat)}>
                                            <div className="catalog-card-name">{cat.name}</div>
                                            <div className="catalog-card-id">{cat.id}</div>
                                            <div className="catalog-card-desc">{cat.description}</div>
                                            <div className="catalog-card-meta">
                                                <span className="badge badge-primary">{cat.size}</span>
                                                <span className="badge badge-info">{cat.pairs}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Custom Download Modal */}
            <AnimatePresence>
                {showCustom && (
                    <div className="modal-overlay" onClick={() => setShowCustom(false)}>
                        <motion.div className="modal" onClick={(e) => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
                            <div className="modal-header">
                                <span className="modal-title">Download Custom Dataset</span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowCustom(false)}><X size={18} /></button>
                            </div>
                            <div className="modal-body">
                                <div className="input-group">
                                    <label className="input-label">HuggingFace Dataset ID <span className="required">*</span></label>
                                    <input className="input" placeholder="e.g. Helsinki-NLP/opus-100" value={customName} onChange={(e) => setCustomName(e.target.value)} />
                                </div>
                                <div className="form-row">
                                    <LangSelect label="Source Language" value={customSrcLang} onChange={setCustomSrcLang} />
                                    <LangSelect label="Target Language" value={customTgtLang} onChange={setCustomTgtLang} />
                                </div>
                                <div className="form-row">
                                    <div className="input-group">
                                        <label className="input-label">Split</label>
                                        <select className="select" value={customSplit} onChange={(e) => setCustomSplit(e.target.value)}>
                                            <option value="train">Train</option>
                                            <option value="test">Test</option>
                                            <option value="validation">Validation</option>
                                        </select>
                                    </div>
                                    <div className="input-group">
                                        <label className="input-label">Max Samples</label>
                                        <input className="input" type="number" placeholder="Leave empty for all" value={customMaxSamples} onChange={(e) => setCustomMaxSamples(e.target.value)} />
                                    </div>
                                </div>
                            </div>
                            <div className="modal-footer">
                                <button className="btn btn-secondary" onClick={() => setShowCustom(false)}>Cancel</button>
                                <button className="btn btn-primary" disabled={!customName || loading} onClick={handleCustomDownload}>
                                    {loading ? <span className="spinner" /> : <Download size={16} />} Download
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Upload Modal */}
            <AnimatePresence>
                {showUpload && (
                    <div className="modal-overlay" onClick={() => setShowUpload(false)}>
                        <motion.div className="modal" onClick={(e) => e.stopPropagation()} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
                            <div className="modal-header">
                                <span className="modal-title">Upload Dataset File</span>
                                <button className="btn btn-ghost btn-icon" onClick={() => setShowUpload(false)}><X size={18} /></button>
                            </div>
                            <div className="modal-body">
                                <div className="input-group">
                                    <label className="input-label">Dataset Name <span className="required">*</span></label>
                                    <input className="input" placeholder="e.g. my-custom-dataset" value={uploadName} onChange={(e) => setUploadName(e.target.value)} />
                                </div>
                                <div className="form-row">
                                    <LangSelect label="Source Language" value={uploadSrcLang} onChange={setUploadSrcLang} />
                                    <LangSelect label="Target Language" value={uploadTgtLang} onChange={setUploadTgtLang} />
                                </div>
                                <div className="input-group">
                                    <label className="input-label">File <span className="required">*</span></label>
                                    <div
                                        style={{
                                            border: '2px dashed var(--border-primary)',
                                            borderRadius: 'var(--radius-md)',
                                            padding: 'var(--space-xl)',
                                            textAlign: 'center',
                                            cursor: 'pointer',
                                            transition: 'border-color var(--transition-fast)',
                                        }}
                                        onClick={() => document.getElementById('file-upload').click()}
                                    >
                                        <Upload size={24} style={{ color: 'var(--text-tertiary)', marginBottom: 8 }} />
                                        <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                                            {uploadFile ? uploadFile.name : 'Click to select a file (.json, .jsonl, .csv, .tsv)'}
                                        </p>
                                    </div>
                                    <input
                                        id="file-upload"
                                        type="file"
                                        accept=".json,.jsonl,.csv,.tsv"
                                        style={{ display: 'none' }}
                                        onChange={(e) => setUploadFile(e.target.files[0])}
                                    />
                                </div>
                            </div>
                            <div className="modal-footer">
                                <button className="btn btn-secondary" onClick={() => setShowUpload(false)}>Cancel</button>
                                <button className="btn btn-primary" disabled={!uploadFile || !uploadName || loading} onClick={handleUpload}>
                                    {loading ? <span className="spinner" /> : <Upload size={16} />} Upload
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}

function DatasetStatusBadge({ status }) {
    const map = {
        ready: { cls: 'badge-success', label: 'Ready' },
        downloading: { cls: 'badge-info', label: 'Downloading' },
        error: { cls: 'badge-danger', label: 'Error' },
    };
    const info = map[status] || { cls: 'badge-neutral', label: status };
    return <span className={`badge ${info.cls}`}>{info.label}</span>;
}
