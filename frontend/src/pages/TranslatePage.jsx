import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    Languages,
    ArrowRightLeft,
    Send,
    Copy,
    Volume2,
    Loader2,
    Brain,
    Sparkles,
    History,
    Trash2,
} from 'lucide-react';
import { listModels, translate, getLanguages } from '../api';

export default function TranslatePage() {
    const [models, setModels] = useState([]);
    const [languages, setLanguages] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [sourceLang, setSourceLang] = useState('en');
    const [targetLang, setTargetLang] = useState('vi');
    const [inputText, setInputText] = useState('');
    const [outputText, setOutputText] = useState('');
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);

    useEffect(() => {
        listModels()
            .then((r) => {
                const ready = r.data.models.filter((m) => m.status === 'ready');
                setModels(ready);
                if (ready.length > 0) setSelectedModel(ready[0].id);
            })
            .catch(() => { });
        getLanguages()
            .then((r) => setLanguages(r.data.languages))
            .catch(() => { });
    }, []);

    const handleTranslate = async () => {
        if (!inputText.trim() || !selectedModel) {
            toast.error('Enter text and select a model');
            return;
        }
        try {
            setLoading(true);
            setOutputText('');
            const res = await translate({
                model_id: selectedModel,
                text: inputText,
                source_lang: sourceLang,
                target_lang: targetLang,
            });
            setOutputText(res.data.translated);
            setHistory((prev) => [
                {
                    id: Date.now(),
                    input: inputText,
                    output: res.data.translated,
                    from: sourceLang,
                    to: targetLang,
                    model: res.data.model_used,
                    time: new Date().toLocaleTimeString(),
                },
                ...prev.slice(0, 19),
            ]);
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Translation failed');
        } finally {
            setLoading(false);
        }
    };

    const handleSwap = () => {
        setSourceLang(targetLang);
        setTargetLang(sourceLang);
        setInputText(outputText);
        setOutputText(inputText);
    };

    const handleCopy = (text) => {
        navigator.clipboard.writeText(text);
        toast.success('Copied to clipboard');
    };

    const getLangInfo = (code) => languages.find((l) => l.code === code) || { name: code, flag: '🌐' };

    const LangSelector = ({ value, onChange }) => (
        <select
            className="select"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-primary)',
                fontWeight: 600,
                fontSize: 14,
                padding: '4px 28px 4px 4px',
                cursor: 'pointer',
            }}
        >
            {languages.length > 0
                ? languages.map((l) => (
                    <option key={l.code} value={l.code} style={{ background: 'var(--bg-secondary)' }}>
                        {l.flag} {l.name}
                    </option>
                ))
                : <>
                    <option value="en" style={{ background: 'var(--bg-secondary)' }}>🇺🇸 English</option>
                    <option value="vi" style={{ background: 'var(--bg-secondary)' }}>🇻🇳 Vietnamese</option>
                    <option value="zh" style={{ background: 'var(--bg-secondary)' }}>🇨🇳 Chinese</option>
                    <option value="ja" style={{ background: 'var(--bg-secondary)' }}>🇯🇵 Japanese</option>
                    <option value="ko" style={{ background: 'var(--bg-secondary)' }}>🇰🇷 Korean</option>
                    <option value="fr" style={{ background: 'var(--bg-secondary)' }}>🇫🇷 French</option>
                    <option value="de" style={{ background: 'var(--bg-secondary)' }}>🇩🇪 German</option>
                    <option value="es" style={{ background: 'var(--bg-secondary)' }}>🇪🇸 Spanish</option>
                </>
            }
        </select>
    );

    return (
        <div>
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h2>Translation</h2>
                <p>Translate text using your trained or downloaded models</p>
            </motion.div>

            {/* Model Selection */}
            <motion.div
                className="card"
                style={{ marginBottom: 'var(--space-xl)', padding: 'var(--space-md) var(--space-lg)' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                    <Brain size={18} style={{ color: 'var(--accent-indigo-light)' }} />
                    <span style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-secondary)' }}>Model:</span>
                    <select
                        className="select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        style={{ minWidth: 280 }}
                    >
                        {models.length === 0 ? (
                            <option value="">No models available — download one first</option>
                        ) : (
                            models.map((m) => (
                                <option key={m.id} value={m.id}>
                                    {m.alias} ({m.hf_name})
                                </option>
                            ))
                        )}
                    </select>
                    {models.length > 0 && (
                        <span className="badge badge-success">
                            <Sparkles size={10} /> Ready
                        </span>
                    )}
                </div>
            </motion.div>

            {/* Translation Area */}
            <motion.div
                className="translate-area"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
            >
                {/* Source */}
                <div className="translate-box">
                    <div className="translate-box-header">
                        <div className="translate-box-lang">
                            <span style={{ fontSize: 18 }}>{getLangInfo(sourceLang).flag}</span>
                            <LangSelector value={sourceLang} onChange={setSourceLang} />
                        </div>
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={() => setInputText('')}
                            style={{ opacity: inputText ? 1 : 0.3 }}
                        >
                            <Trash2 size={14} />
                        </button>
                    </div>
                    <textarea
                        className="translate-textarea"
                        placeholder="Enter text to translate..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                                handleTranslate();
                            }
                        }}
                    />
                    <div className="translate-footer">
                        <span className="char-count">{inputText.length} characters</span>
                        <button
                            className="btn btn-primary"
                            disabled={!inputText.trim() || !selectedModel || loading}
                            onClick={handleTranslate}
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={16} style={{ animation: 'spin 0.7s linear infinite' }} />
                                    Translating...
                                </>
                            ) : (
                                <>
                                    <Send size={16} /> Translate
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Swap Button */}
                <div className="translate-swap">
                    <button
                        className="btn btn-secondary btn-icon"
                        onClick={handleSwap}
                        style={{
                            width: 44,
                            height: 44,
                            borderRadius: 'var(--radius-full)',
                            transition: 'all var(--transition-base)',
                        }}
                        title="Swap languages"
                    >
                        <ArrowRightLeft size={18} />
                    </button>
                </div>

                {/* Target */}
                <div className="translate-box">
                    <div className="translate-box-header">
                        <div className="translate-box-lang">
                            <span style={{ fontSize: 18 }}>{getLangInfo(targetLang).flag}</span>
                            <LangSelector value={targetLang} onChange={setTargetLang} />
                        </div>
                        <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                            <button
                                className="btn btn-ghost btn-sm"
                                onClick={() => handleCopy(outputText)}
                                disabled={!outputText}
                            >
                                <Copy size={14} />
                            </button>
                        </div>
                    </div>
                    <div
                        className="translate-textarea"
                        style={{
                            display: 'flex',
                            alignItems: outputText ? 'flex-start' : 'center',
                            justifyContent: outputText ? 'flex-start' : 'center',
                            color: outputText ? 'var(--text-primary)' : 'var(--text-tertiary)',
                            whiteSpace: 'pre-wrap',
                            overflow: 'auto',
                        }}
                    >
                        {loading ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', margin: 'auto' }}>
                                <Loader2 size={20} style={{ animation: 'spin 0.7s linear infinite', color: 'var(--accent-indigo-light)' }} />
                                <span>Translating...</span>
                            </div>
                        ) : outputText ? (
                            outputText
                        ) : (
                            'Translation will appear here...'
                        )}
                    </div>
                    <div className="translate-footer">
                        <span className="char-count">{outputText.length} characters</span>
                        <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                            ⌘+Enter to translate
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* History */}
            {history.length > 0 && (
                <motion.div
                    className="card"
                    style={{ marginTop: 'var(--space-xl)' }}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                >
                    <div className="card-header">
                        <div className="card-title">
                            <div className="card-title-icon indigo">
                                <History size={18} />
                            </div>
                            Translation History
                        </div>
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={() => setHistory([])}
                        >
                            Clear
                        </button>
                    </div>
                    <div className="table-container">
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Direction</th>
                                    <th>Input</th>
                                    <th>Output</th>
                                    <th>Model</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((h) => (
                                    <tr key={h.id}>
                                        <td style={{ fontSize: 12, color: 'var(--text-tertiary)', whiteSpace: 'nowrap' }}>{h.time}</td>
                                        <td>
                                            <span className="badge badge-primary">
                                                {getLangInfo(h.from).flag} → {getLangInfo(h.to).flag}
                                            </span>
                                        </td>
                                        <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            {h.input}
                                        </td>
                                        <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            {h.output}
                                        </td>
                                        <td style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>{h.model}</td>
                                        <td>
                                            <button className="btn btn-ghost btn-icon btn-sm" onClick={() => handleCopy(h.output)}>
                                                <Copy size={12} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
