import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Brain,
    Database,
    Zap,
    Languages,
    ArrowRight,
    TrendingUp,
    Activity,
    CheckCircle2,
    AlertCircle,
    Clock,
    Download,
    Sparkles,
} from 'lucide-react';
import { listModels, listDatasets, listTrainingJobs, getSystemInfo } from '../api';

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.08, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
    }),
};

export default function DashboardPage() {
    const navigate = useNavigate();
    const [models, setModels] = useState([]);
    const [datasets, setDatasets] = useState([]);
    const [jobs, setJobs] = useState([]);
    const [system, setSystem] = useState(null);

    useEffect(() => {
        const fetchAll = async () => {
            try {
                const [m, d, j, s] = await Promise.allSettled([
                    listModels(),
                    listDatasets(),
                    listTrainingJobs(),
                    getSystemInfo(),
                ]);
                if (m.status === 'fulfilled') setModels(m.value.data.models);
                if (d.status === 'fulfilled') setDatasets(d.value.data.datasets);
                if (j.status === 'fulfilled') setJobs(j.value.data.jobs);
                if (s.status === 'fulfilled') setSystem(s.value.data);
            } catch { /* offline */ }
        };
        fetchAll();
        const interval = setInterval(fetchAll, 8000);
        return () => clearInterval(interval);
    }, []);

    const readyModels = models.filter((m) => m.status === 'ready').length;
    const readyDatasets = datasets.filter((d) => d.status === 'ready').length;
    const activeJobs = jobs.filter((j) => ['training', 'initializing', 'loading_model', 'loading_dataset', 'tokenizing'].includes(j.status)).length;
    const completedJobs = jobs.filter((j) => j.status === 'completed').length;

    const stats = [
        {
            label: 'Models Ready',
            value: readyModels,
            total: models.length,
            icon: Brain,
            color: 'indigo',
            bg: 'rgba(99, 102, 241, 0.15)',
            path: '/models',
        },
        {
            label: 'Datasets Loaded',
            value: readyDatasets,
            total: datasets.length,
            icon: Database,
            color: 'cyan',
            bg: 'rgba(34, 211, 238, 0.15)',
            path: '/datasets',
        },
        {
            label: 'Active Training',
            value: activeJobs,
            total: jobs.length,
            icon: Zap,
            color: 'amber',
            bg: 'rgba(251, 191, 36, 0.15)',
            path: '/training',
        },
        {
            label: 'Completed',
            value: completedJobs,
            total: jobs.length,
            icon: CheckCircle2,
            color: 'emerald',
            bg: 'rgba(52, 211, 153, 0.15)',
            path: '/training',
        },
    ];

    const colorMap = {
        indigo: 'var(--accent-indigo-light)',
        cyan: 'var(--accent-cyan)',
        amber: 'var(--accent-amber)',
        emerald: 'var(--accent-emerald)',
    };

    const quickActions = [
        {
            label: 'Translate Text',
            desc: 'Use trained models for instant translation',
            icon: Languages,
            color: 'var(--gradient-primary)',
            path: '/translate',
        },
        {
            label: 'Download Model',
            desc: 'Clone models from HuggingFace Hub',
            icon: Download,
            color: 'var(--gradient-secondary)',
            path: '/models',
        },
        {
            label: 'Start Training',
            desc: 'Fine-tune models on your datasets',
            icon: Sparkles,
            color: 'var(--gradient-warm)',
            path: '/training',
        },
    ];

    return (
        <div>
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <h2>Dashboard</h2>
                <p>Overview of your translation training system</p>
            </motion.div>

            {/* Stats */}
            <div className="grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
                {stats.map((stat, i) => {
                    const Icon = stat.icon;
                    return (
                        <motion.div
                            key={stat.label}
                            className="stat-card"
                            custom={i}
                            initial="hidden"
                            animate="visible"
                            variants={fadeUp}
                            onClick={() => navigate(stat.path)}
                            style={{ cursor: 'pointer' }}
                        >
                            <div
                                className="stat-card-icon"
                                style={{ background: stat.bg, color: colorMap[stat.color] }}
                            >
                                <Icon size={22} />
                            </div>
                            <div className="stat-card-value">{stat.value}</div>
                            <div className="stat-card-label">
                                {stat.label}
                                {stat.total > 0 && (
                                    <span style={{ color: 'var(--text-tertiary)', marginLeft: 4 }}>
                                        / {stat.total} total
                                    </span>
                                )}
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Quick Actions */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
            >
                <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 'var(--space-md)' }}>
                    Quick Actions
                </h3>
                <div className="grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
                    {quickActions.map((action, i) => {
                        const Icon = action.icon;
                        return (
                            <motion.div
                                key={action.label}
                                className="card"
                                custom={i + 4}
                                initial="hidden"
                                animate="visible"
                                variants={fadeUp}
                                onClick={() => navigate(action.path)}
                                style={{ cursor: 'pointer' }}
                            >
                                <div
                                    style={{
                                        width: 44,
                                        height: 44,
                                        borderRadius: 'var(--radius-md)',
                                        background: action.color,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        marginBottom: 'var(--space-md)',
                                    }}
                                >
                                    <Icon size={22} color="white" />
                                </div>
                                <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>
                                    {action.label}
                                </div>
                                <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                                    {action.desc}
                                </div>
                                <div
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 4,
                                        marginTop: 'var(--space-md)',
                                        fontSize: 13,
                                        color: 'var(--accent-indigo-light)',
                                        fontWeight: 500,
                                    }}
                                >
                                    Get started <ArrowRight size={14} />
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </motion.div>

            {/* Recent Activity */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                <div className="grid-2">
                    {/* Recent Jobs */}
                    <div className="card">
                        <div className="card-header">
                            <div className="card-title">
                                <div className="card-title-icon amber">
                                    <Activity size={18} />
                                </div>
                                Recent Training Jobs
                            </div>
                        </div>
                        {jobs.length === 0 ? (
                            <div style={{ textAlign: 'center', padding: 'var(--space-xl)', color: 'var(--text-tertiary)' }}>
                                <Clock size={32} style={{ marginBottom: 8, opacity: 0.5 }} />
                                <p style={{ fontSize: 13 }}>No training jobs yet</p>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                                {jobs.slice(-5).reverse().map((job) => (
                                    <div
                                        key={job.id}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            padding: 'var(--space-sm) var(--space-md)',
                                            background: 'var(--bg-tertiary)',
                                            borderRadius: 'var(--radius-md)',
                                        }}
                                    >
                                        <div>
                                            <div style={{ fontSize: 13, fontWeight: 500 }}>
                                                {job.model_name}
                                            </div>
                                            <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                                                {job.source_lang} → {job.target_lang}
                                            </div>
                                        </div>
                                        <StatusBadge status={job.status} />
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* System Info */}
                    <div className="card">
                        <div className="card-header">
                            <div className="card-title">
                                <div className="card-title-icon cyan">
                                    <TrendingUp size={18} />
                                </div>
                                System Resources
                            </div>
                        </div>
                        {system ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
                                <ResourceBar label="CPU" value={system.cpu_percent} />
                                <ResourceBar label="Memory" value={system.memory_percent} detail={`${system.memory_used_gb} / ${system.memory_total_gb} GB`} />
                                <ResourceBar label="Disk" value={system.disk_percent} detail={`${system.disk_used_gb} / ${system.disk_total_gb} GB`} />
                                {system.gpu?.length > 0 && (
                                    <div>
                                        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
                                            GPU: {system.gpu[0].name}
                                        </div>
                                        <span className="badge badge-success">Available</span>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ textAlign: 'center', padding: 'var(--space-xl)', color: 'var(--text-tertiary)' }}>
                                <AlertCircle size={32} style={{ marginBottom: 8, opacity: 0.5 }} />
                                <p style={{ fontSize: 13 }}>Backend not connected</p>
                                <p style={{ fontSize: 12, marginTop: 4 }}>Start the server: <code style={{ fontFamily: 'var(--font-mono)' }}>python -m backend.main</code></p>
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>
        </div>
    );
}

function StatusBadge({ status }) {
    const map = {
        ready: { cls: 'badge-success', label: 'Ready' },
        completed: { cls: 'badge-success', label: 'Completed' },
        training: { cls: 'badge-warning', label: 'Training' },
        downloading: { cls: 'badge-info', label: 'Downloading' },
        initializing: { cls: 'badge-info', label: 'Initializing' },
        loading_model: { cls: 'badge-info', label: 'Loading model' },
        loading_dataset: { cls: 'badge-info', label: 'Loading data' },
        tokenizing: { cls: 'badge-info', label: 'Tokenizing' },
        saving: { cls: 'badge-info', label: 'Saving' },
        failed: { cls: 'badge-danger', label: 'Failed' },
        error: { cls: 'badge-danger', label: 'Error' },
        stopped: { cls: 'badge-neutral', label: 'Stopped' },
    };
    const info = map[status] || { cls: 'badge-neutral', label: status };
    return <span className={`badge ${info.cls}`}>{info.label}</span>;
}

function ResourceBar({ label, value, detail }) {
    const barCls = value > 85 ? 'danger' : value > 65 ? 'warning' : '';
    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)' }}>
                    {label}
                </span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                    {detail || `${value}%`}
                </span>
            </div>
            <div className="progress">
                <div className={`progress-bar ${barCls}`} style={{ width: `${value}%` }} />
            </div>
        </div>
    );
}
