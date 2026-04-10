import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import styles from './HistoryPage.module.css';

export default function HistoryPage() {
  const { isAdmin } = useAuth();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');

  useEffect(() => {
    if (!isAdmin) { setLoading(false); return; }
    api.history(100)
      .then(d => setArticles(d.articles || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [isAdmin]);

  if (!isAdmin) {
    return (
      <div className={styles.page}>
        <div className={styles.gated}>
          <div className={styles.gatedIcon}>⊘</div>
          <div className={styles.gatedTitle}>Admin access required</div>
          <div className={styles.gatedSub}>This page is only accessible to admin accounts.</div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.eyebrow}>Admin</div>
        <h1 className={styles.title}>Analysis history</h1>
        <p className={styles.sub}>{articles.length} recent analyses</p>
      </div>

      {loading && <div className={styles.loading}><span className={styles.spinner} /> Loading…</div>}
      {error   && <div className={styles.error}>{error}</div>}

      {!loading && !error && articles.length === 0 && (
        <div className={styles.empty}>No analyses recorded yet.</div>
      )}

      {articles.length > 0 && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Hash</th>
                <th>Language</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Chain</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {articles.map(a => (
                <tr key={a.id}>
                  <td className={styles.mono}>{a.content_hash?.slice(0, 16)}…</td>
                  <td><span className={styles.langPill}>{a.language}</span></td>
                  <td>
                    <span className={a.prediction === 'fake' ? styles.fake : styles.real}>
                      {a.prediction?.toUpperCase()}
                    </span>
                  </td>
                  <td className={styles.mono}>{Math.round(a.confidence * 100)}%</td>
                  <td>{a.blockchain_verified
                    ? <span className={styles.verified}>✓</span>
                    : <span className={styles.unverified}>–</span>}
                  </td>
                  <td className={styles.time}>{new Date(a.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}