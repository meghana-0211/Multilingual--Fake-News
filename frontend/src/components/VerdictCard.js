import React, { useState } from 'react';
import { api } from '../services/api';
import styles from './VerdictCard.module.css';

const FLAG_LABELS = ['Misleading', 'False', 'Satire', 'Unverified', 'Correct'];

export default function VerdictCard({ result, text, language }) {
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [correctLabel, setCorrectLabel] = useState('');
  const [notes, setNotes] = useState('');

  const isFake = result.prediction === 'fake';
  const pct = Math.round(result.confidence * 100);
  const exp = result.explanation || {};
  const bc  = result.blockchain  || {};

  const topWords = exp.combined_top_words  || [];
  const limeFeats = exp.lime_features      || [];

  const handleFeedback = async () => {
    if (!correctLabel) return;
    setFeedbackLoading(true);
    try {
      await api.submitFeedback({
        text,
        language,
        predictedLabel: result.prediction,
        correctLabel,
        confidence: result.confidence,
        notes,
      });
      setFeedbackSent(true);
      setShowFeedback(false);
    } catch (e) {
      alert('Feedback submission failed: ' + e.message);
    } finally {
      setFeedbackLoading(false);
    }
  };

  return (
    <div className={`${styles.card} ${isFake ? styles.fake : styles.real}`}>

      {/* ── Verdict banner ── */}
      <div className={styles.banner}>
        <div className={styles.verdictLeft}>
          <div className={styles.indicator} />
          <div>
            <div className={styles.verdictLabel}>{isFake ? 'LIKELY FAKE' : 'LIKELY REAL'}</div>
            <div className={styles.verdictSub}>
              {isFake ? 'This article shows signs of misinformation' : 'No strong misinformation signals detected'}
            </div>
          </div>
        </div>
        <div className={styles.confidenceBlock}>
          <div className={styles.confidenceNum}>{pct}<span>%</span></div>
          <div className={styles.confidenceLabel}>confidence</div>
        </div>
      </div>

      {/* ── Score bar ── */}
      <div className={styles.scoreRow}>
        <div className={styles.scoreBarWrap}>
          <div
            className={`${styles.scoreBar} ${isFake ? styles.fakeBar : styles.realBar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className={styles.scoreNums}>
          <span className={styles.scoreReal}>Real {Math.round(result.scores?.real * 100)}%</span>
          <span className={styles.scoreFake}>Fake {Math.round(result.scores?.fake * 100)}%</span>
        </div>
      </div>

      {/* ── Content hash ── */}
      <div className={styles.hashRow}>
        <span className={styles.hashLabel}>SHA-256</span>
        <span className={styles.hashValue}>{result.contentHash?.slice(0, 32)}…</span>
        {bc.verified
          ? <span className={styles.bcVerified}>✓ On-chain</span>
          : <span className={styles.bcPending}>⊘ Not registered</span>}
      </div>

      {/* ── Grid: explanation + blockchain ── */}
      <div className={styles.grid}>

        {/* Influential words */}
        {topWords.length > 0 && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Key signals</h3>
            <div className={styles.wordList}>
              {topWords.slice(0, 10).map(({ word, score }, i) => (
                <div key={i} className={styles.wordRow}>
                  <span className={styles.wordText}>{word}</span>
                  <div className={styles.wordBarWrap}>
                    <div
                      className={`${styles.wordBar} ${score > 0 ? styles.wordBarPos : styles.wordBarNeg}`}
                      style={{ width: `${Math.min(Math.abs(score) * 120, 100)}%` }}
                    />
                  </div>
                  <span className={`${styles.wordScore} ${score > 0 ? styles.pos : styles.neg}`}>
                    {score > 0 ? '+' : ''}{score.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* LIME features */}
        {limeFeats.length > 0 && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>LIME analysis</h3>
            <div className={styles.wordList}>
              {limeFeats.slice(0, 8).map(({ word, score }, i) => (
                <div key={i} className={styles.wordRow}>
                  <span className={styles.wordText}>{word}</span>
                  <span className={`${styles.wordScore} ${score > 0 ? styles.pos : styles.neg}`}>
                    {score > 0 ? '▲' : '▼'} {Math.abs(score).toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Blockchain status */}
        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>Blockchain provenance</h3>
          <div className={styles.bcBlock}>
            <div className={styles.bcRow}>
              <span className={styles.bcKey}>Status</span>
              <span className={bc.verified ? styles.bcVerified : styles.bcPending}>
                {bc.verified ? 'Verified on-chain' : 'Not registered'}
              </span>
            </div>
            {bc.publisher && (
              <div className={styles.bcRow}>
                <span className={styles.bcKey}>Publisher</span>
                <span className={styles.bcVal}>{bc.publisher.slice(0, 18)}…</span>
              </div>
            )}
            {bc.timestamp && (
              <div className={styles.bcRow}>
                <span className={styles.bcKey}>Registered</span>
                <span className={styles.bcVal}>{new Date(bc.timestamp * 1000).toLocaleString()}</span>
              </div>
            )}
            {bc.annotations?.length > 0 && (
              <div className={styles.bcRow}>
                <span className={styles.bcKey}>Annotations</span>
                <span className={styles.bcVal}>{bc.annotations.length} fact-check flag(s)</span>
              </div>
            )}
            {bc.annotations?.map((ann, i) => (
              <div key={i} className={styles.annotation}>
                <span className={styles.flagType}>{FLAG_LABELS[ann.flag_type] ?? ann.flag_type}</span>
                <span className={styles.flagConf}>{ann.confidence}% confidence</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* ── Feedback ── */}
      <div className={styles.feedbackSection}>
        {feedbackSent ? (
          <div className={styles.feedbackThanks}>✓ Feedback recorded — thank you.</div>
        ) : showFeedback ? (
          <div className={styles.feedbackForm}>
            <div className={styles.feedbackRow}>
              <label>Correct label:</label>
              <select value={correctLabel} onChange={e => setCorrectLabel(e.target.value)} className={styles.select}>
                <option value="">— select —</option>
                <option value="real">Real</option>
                <option value="fake">Fake</option>
              </select>
            </div>
            <div className={styles.feedbackRow}>
              <label>Notes (optional):</label>
              <input
                type="text"
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Why is the prediction wrong?"
                className={styles.input}
              />
            </div>
            <div className={styles.feedbackActions}>
              <button className={styles.submitBtn} onClick={handleFeedback} disabled={feedbackLoading || !correctLabel}>
                {feedbackLoading ? 'Submitting…' : 'Submit'}
              </button>
              <button className={styles.cancelBtn} onClick={() => setShowFeedback(false)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button className={styles.feedbackBtn} onClick={() => setShowFeedback(true)}>
            Flag incorrect prediction
          </button>
        )}
      </div>
    </div>
  );
}
