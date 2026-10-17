import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { RecordingRenderer } from './RecordingRenderer';
import './styles.css';

createRoot(document.getElementById('root')!).render(window.location.pathname === '/recording-render' ? <RecordingRenderer /> : <StrictMode><App /></StrictMode>);
