import { NavLink, useLocation } from 'react-router-dom';
import {
    Languages,
    Brain,
    Database,
    Zap,
    LayoutDashboard,
    Cpu,
    HardDrive,
    MemoryStick,
    Film,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { getSystemInfo } from '../api';

const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/translate', label: 'Translation', icon: Languages },
    { path: '/models', label: 'Models', icon: Brain },
    { path: '/datasets', label: 'Datasets', icon: Database },
    { path: '/training', label: 'Training', icon: Zap },
    { path: '/subtitles', label: 'Subtitles', icon: Film },
];

export default function Sidebar() {
    const location = useLocation();
    const [system, setSystem] = useState(null);

    useEffect(() => {
        const fetchSystem = async () => {
            try {
                const res = await getSystemInfo();
                setSystem(res.data);
            } catch {
                /* server may not be running */
            }
        };
        fetchSystem();
        const interval = setInterval(fetchSystem, 10000);
        return () => clearInterval(interval);
    }, []);

    const getBarClass = (pct) => {
        if (pct > 85) return 'status-bar-fill danger';
        if (pct > 65) return 'status-bar-fill warning';
        return 'status-bar-fill';
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">AT</div>
                <div className="sidebar-brand">
                    <h1>Agent Translate</h1>
                    <span>AI Training System</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                <div className="nav-section-label">Main</div>
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive =
                        item.path === '/'
                            ? location.pathname === '/'
                            : location.pathname.startsWith(item.path);
                    return (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={`nav-item ${isActive ? 'active' : ''}`}
                        >
                            <Icon className="nav-item-icon" size={20} />
                            <span>{item.label}</span>
                        </NavLink>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                <div className="system-status">
                    <div className="status-item">
                        <span className="status-label">
                            <Cpu size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                            CPU
                        </span>
                        <span className="status-value">
                            {system ? `${system.cpu_percent}%` : '—'}
                        </span>
                    </div>
                    {system && (
                        <div className="status-bar">
                            <div
                                className={getBarClass(system.cpu_percent)}
                                style={{ width: `${system.cpu_percent}%` }}
                            />
                        </div>
                    )}

                    <div className="status-item" style={{ marginTop: 8 }}>
                        <span className="status-label">
                            <MemoryStick size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                            RAM
                        </span>
                        <span className="status-value">
                            {system
                                ? `${system.memory_used_gb}/${system.memory_total_gb} GB`
                                : '—'}
                        </span>
                    </div>
                    {system && (
                        <div className="status-bar">
                            <div
                                className={getBarClass(system.memory_percent)}
                                style={{ width: `${system.memory_percent}%` }}
                            />
                        </div>
                    )}

                    <div className="status-item" style={{ marginTop: 8 }}>
                        <span className="status-label">
                            <HardDrive size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                            Disk
                        </span>
                        <span className="status-value">
                            {system
                                ? `${system.disk_used_gb}/${system.disk_total_gb} GB`
                                : '—'}
                        </span>
                    </div>
                    {system && (
                        <div className="status-bar">
                            <div
                                className={getBarClass(system.disk_percent)}
                                style={{ width: `${system.disk_percent}%` }}
                            />
                        </div>
                    )}

                    {system?.gpu?.length > 0 && (
                        <>
                            <div className="status-item" style={{ marginTop: 8 }}>
                                <span className="status-label">GPU</span>
                                <span className="status-value">{system.gpu[0].name}</span>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </aside>
    );
}
