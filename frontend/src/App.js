import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Layout from './components/Layout';
import AnalysePage    from './pages/AnalysePage';
import { LoginPage, RegisterPage } from './pages/AuthPage';
import HistoryPage    from './pages/HistoryPage';
import FeedbackPage   from './pages/FeedbackPage';
import AdminPage      from './pages/AdminPage';

// Redirect to login if not authenticated
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>Loading…</div>;
  if (!user)   return <Navigate to="/login" replace />;
  return children;
}

// Redirect to home if already logged in
function GuestRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (user)    return <Navigate to="/" replace />;
  return children;
}

function AppRoutes() {
  return (
    <Layout>
      <Routes>
        <Route path="/"         element={<AnalysePage />} />
        <Route path="/history"  element={<ProtectedRoute><HistoryPage /></ProtectedRoute>} />
        <Route path="/feedback" element={<FeedbackPage />} />
        <Route path="/admin"    element={<ProtectedRoute><AdminPage /></ProtectedRoute>} />
        <Route path="/login"    element={<GuestRoute><LoginPage /></GuestRoute>} />
        <Route path="/register" element={<GuestRoute><RegisterPage /></GuestRoute>} />
        <Route path="*"         element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
