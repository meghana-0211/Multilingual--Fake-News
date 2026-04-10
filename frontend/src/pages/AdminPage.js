import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import styles from './AdminPage.module.css';

const FLAG_OPTIONS = [
  { value: 0, label: 'Misleading' },
  { value: 1, label: 'False' },
  { value: 2, label: 'Satire' },
  { value: 3, label: 'Unverified' },
  { value: 4, label: 'Correct' },
];

const LANGUAGES = ['hindi', 'gujarati', 'marathi', 'telugu'];

function Section({ title, eyebrow, children }) {
  return (
    <div className={styles.section}>
      {eyebrow && <div className={styles.eyebrow}>{eyebrow}</div>}
      <h2 className={styles.sectionTitle}>{title}</h2>
      {children}
    </div>
  );
}

function StatusDot({ ok }) {
  return <span className={ok ? styles.dotGreen : styles.dotRed} />;
}

export default function AdminPage() {
  const { isAdmin, isPublisher, isFactChecker } = useAuth();

  // System health
  const [health, setHealth]   = useState(null);

  // Register article
  const [regText, setRegText]     = useState('');
  const [regLang, setRegLang]     = useState('hindi');
  const [regResult, setRegResult] = useState(null);
  const [regLoading, setRegLoading] = useState(false);
  const [regError, setRegError]   = useState('');

  // Add annotation
  const [annText, setAnnText]       = useState('');
  const [annFlag, setAnnFlag]       = useState(1);
  const [annConf, setAnnConf]       = useState(80);
  const [annIpfs, setAnnIpfs]       = useState('');
  const [annResult, setAnnResult]   = useState(null);
  const [annLoading, setAnnLoading] = useState(false);
  const [annError, setAnnError]     = useState('');

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'error' }));
  }, []);

  const handleRegister = async () => {
    if (!regText.trim()) return;
    setRegLoading(true); setRegError(''); setRegResult(null);
    try {
      const r = await api.registerArticle(regText.trim(), regLang);
      setRegResult(r);
    } catch (e) {
      setRegError(e.message);
    } finally {
      setRegLoading(false);
    }
  };

  const handleAnnotate = async () => {
    if (!annText.trim()) return;
    setAnnLoading(true); setAnnError(''); setAnnResult(null);
    try {
      const r = await api.addAnnotation(annText.trim(), annFlag, annIpfs, annConf);
      setAnnResult(r);
    } catch (e) {
      setAnnError(e.message);
    } finally {
      setAnnLoading(false);
    }
  };

  if (!isAdmin && !isPublisher && !isFactChecker) {
    return (
      <div className={styles.page}>
        <div className={styles.gated}>
          <div className={styles.gatedIcon}>⊘</div>
          <div className={styles.gatedTitle}>Restricted area</div>
          <div className={styles.gatedSub}>Publisher, fact-checker, or admin role required.</div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <div className={styles.headerEyebrow}>Admin panel</div>
        <h1 className={styles.headerTitle}>System control</h1>
      </div>

      {/* ── System status ── */}
      <Section title="System status" eyebrow="Infrastructure">
        <div className={styles.statusGrid}>
          <div className={styles.statusCard}>
            <StatusDot ok={health?.status === 'ok'} />
            <div className={styles.statusLabel}>Backend API</div>
            <div className={styles.statusValue}>{health ? health.status : '…'}</div>
          </div>
          <div className={styles.statusCard}>
            <StatusDot ok={health?.model_loaded} />
            <div className={styles.statusLabel}>ML Model</div>
            <div className={styles.statusValue}>{health?.model_loaded ? 'Loaded' : 'Not loaded'}</div>
          </div>
          <div className={styles.statusCard}>
            <StatusDot ok={health?.blockchain} />
            <div className={styles.statusLabel}>Blockchain</div>
            <div className={styles.statusValue}>{health?.blockchain ? 'Connected' : 'Offline'}</div>
          </div>
          <div className={styles.statusCard}>
            <StatusDot ok={health?.explainer} />
            <div className={styles.statusLabel}>Explainer</div>
            <div className={styles.statusValue}>{health?.explainer ? 'Ready' : 'Not ready'}</div>
          </div>
        </div>
      </Section>

      {/* ── Register article on blockchain ── */}
      {(isPublisher || isAdmin) && (
        <Section title="Register article on blockchain" eyebrow="Publisher tools">
          <p className={styles.sectionDesc}>
            Hash and register an article on Ethereum. Creates a tamper-proof timestamp.
            Requires Ganache to be running with a verified publisher account.
          </p>
          <div className={styles.formGroup}>
            <div className={styles.formRow}>
              <label>Language</label>
              <select
                value={regLang}
                onChange={e => setRegLang(e.target.value)}
                className={styles.select}
              >
                {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            </div>
            <textarea
              className={styles.textarea}
              value={regText}
              onChange={e => setRegText(e.target.value)}
              placeholder="Paste article text to register on-chain…"
              rows={5}
            />
            {regError  && <div className={styles.error}>{regError}</div>}
            {regResult && (
              <div className={styles.success}>
                <div className={styles.successLine}>
                  <span className={styles.successKey}>Transaction</span>
                  <span className={styles.mono}>{regResult.transactionHash?.slice(0, 32)}…</span>
                </div>
                <div className={styles.successLine}>
                  <span className={styles.successKey}>Content hash</span>
                  <span className={styles.mono}>{regResult.contentHash?.slice(0, 32)}…</span>
                </div>
              </div>
            )}
            <button
              className={styles.actionBtn}
              onClick={handleRegister}
              disabled={regLoading || !regText.trim()}
            >
              {regLoading ? <><span className={styles.spinner} /> Registering…</> : 'Register on blockchain'}
            </button>
          </div>
        </Section>
      )}

      {/* ── Add annotation ── */}
      {(isFactChecker || isAdmin) && (
        <Section title="Add fact-check annotation" eyebrow="Fact-checker tools">
          <p className={styles.sectionDesc}>
            Flag an article with a fact-check annotation. Stored permanently on-chain and
            visible to all users in the blockchain provenance section.
          </p>
          <div className={styles.formGroup}>
            <div className={styles.formRowGroup}>
              <div className={styles.formRow}>
                <label>Flag type</label>
                <select
                  value={annFlag}
                  onChange={e => setAnnFlag(Number(e.target.value))}
                  className={styles.select}
                >
                  {FLAG_OPTIONS.map(f => (
                    <option key={f.value} value={f.value}>{f.label}</option>
                  ))}
                </select>
              </div>
              <div className={styles.formRow}>
                <label>Confidence: <strong>{annConf}%</strong></label>
                <input
                  type="range"
                  min={0} max={100}
                  value={annConf}
                  onChange={e => setAnnConf(Number(e.target.value))}
                  className={styles.range}
                />
              </div>
              <div className={styles.formRow}>
                <label>IPFS hash <span className={styles.optional}>(optional)</span></label>
                <input
                  type="text"
                  value={annIpfs}
                  onChange={e => setAnnIpfs(e.target.value)}
                  placeholder="Qm…"
                  className={styles.input}
                />
              </div>
            </div>
            <textarea
              className={styles.textarea}
              value={annText}
              onChange={e => setAnnText(e.target.value)}
              placeholder="Paste the article text to annotate…"
              rows={5}
            />
            {annError  && <div className={styles.error}>{annError}</div>}
            {annResult && (
              <div className={styles.success}>
                <div className={styles.successLine}>
                  <span className={styles.successKey}>Transaction</span>
                  <span className={styles.mono}>{annResult.transactionHash?.slice(0, 32)}…</span>
                </div>
              </div>
            )}
            <button
              className={styles.actionBtn}
              onClick={handleAnnotate}
              disabled={annLoading || !annText.trim()}
            >
              {annLoading ? <><span className={styles.spinner} /> Submitting…</> : 'Submit annotation'}
            </button>
          </div>
        </Section>
      )}

      {/* ── Seed reminder ── */}
      {isAdmin && (
        <Section title="Useful commands" eyebrow="Dev reference">
          <div className={styles.codeBlock}>
            <div className={styles.codeLine}>
              <span className={styles.codePrompt}>$</span>
              <span className={styles.codeText}>cd backend && python scripts/seed_admin.py</span>
            </div>
            <div className={styles.codeLine}>
              <span className={styles.codePrompt}>$</span>
              <span className={styles.codeText}>cd backend && pytest tests/ -v</span>
            </div>
            <div className={styles.codeLine}>
              <span className={styles.codePrompt}>$</span>
              <span className={styles.codeText}>ganache-cli --deterministic</span>
            </div>
            <div className={styles.codeLine}>
              <span className={styles.codePrompt}>$</span>
              <span className={styles.codeText}>cd blockchain && truffle migrate --network development</span>
            </div>
          </div>
        </Section>
      )}
    </div>
  );
}
