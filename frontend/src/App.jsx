import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import TranslatePage from './pages/TranslatePage';
import ModelsPage from './pages/ModelsPage';
import DatasetsPage from './pages/DatasetsPage';
import TrainingPage from './pages/TrainingPage';
import SubtitlesPage from './pages/SubtitlesPage';

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/translate" element={<TranslatePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/subtitles" element={<SubtitlesPage />} />
        </Routes>
      </main>
    </div>
  );
}
