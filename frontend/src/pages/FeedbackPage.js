import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import styles from './FeedbackPage.module.css';

export default function FeedbackPage() {
  const { user, isAdmin, isFactChecker } = useAuth();
  const [stats, setStats]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');

  useEffect(() => {
    if (!isFactChecker && !isAdmin) return;
    setLoading(true);
    api.feedbackStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [isAdmin, isFactChecker]);

  const accuracy = stats ? Math.round(stats.accuracy * 100) : 0;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.eyebrow}>Community</div>
        <h1 className={styles.title}>Feedback centre</h1>
        <p className={styles.sub}>
          Help improve the model by flagging incorrect predictions after analysis.
        </p>
      </div>

      {/* Stats (admin/fact_checker only) */}
      {(isAdmin || isFactChecker) && (
        <div className={styles.statsSection}>
          <div className={styles.statsTitle}>Model correction stats</div>
          {loading && <div className={styles.loading}><span className={styles.spinner} /> Loading…</div>}
          {error   && <div className={styles.error}>{error}</div>}
          {stats && !loading && (
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <div className={styles.statNum}>{stats.total}</div>
                <div className={styles.statLabel}>Total feedback</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statNum}>{stats.correct}</div>
                <div className={styles.statLabel}>Correct predictions</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statNum}>{stats.wrong}</div>
                <div className={styles.statLabel}>Wrong predictions</div>
              </div>
              <div className={`${styles.statCard} ${styles.statAccuracy}`}>
                <div className={`${styles.statNum} ${accuracy >= 90 ? styles.good : styles.warn}`}>
                  {accuracy}%
                </div>
                <div className={styles.statLabel}>Model accuracy (feedback)</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* How to provide feedback */}
      <div className={styles.infoSection}>
        <div className={styles.infoTitle}>How to submit feedback</div>
        <ol className={styles.steps}>
          <li>Go to <strong>Analyse</strong> and run a detection on any article.</li>
          <li>Scroll to the bottom of the result card.</li>
          <li>Click <strong>"Flag incorrect prediction"</strong>.</li>
          <li>Select the correct label and optionally add notes.</li>
          <li>Submit — your correction is saved and reviewed by fact-checkers.</li>
        </ol>
      </div>

      {/* Role info */}
      <div className={styles.rolesSection}>
        <div className={styles.rolesTitle}>User roles</div>
        <div className={styles.roleGrid}>
          {[
            { role: 'user',         desc: 'Submit feedback after analysis. No login required for analysis.' },
            { role: 'fact_checker', desc: 'Review feedback, add on-chain annotations, access feedback stats.' },
            { role: 'publisher',    desc: 'Register articles on the Ethereum blockchain for provenance tracking.' },
            { role: 'admin',        desc: 'Full access: history, feedback stats, all role permissions.' },
          ].map(r => (
            <div key={r.role} className={styles.roleCard}>
              <div className={`${styles.roleBadge} ${styles['role_' + r.role]}`}>{r.role}</div>
              <div className={styles.roleDesc}>{r.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}