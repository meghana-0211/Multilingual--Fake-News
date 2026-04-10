import React, { useState } from 'react';
import { api } from '../services/api';
import VerdictCard from '../components/VerdictCard';
import styles from './AnalysePage.module.css';

const LANGUAGES = [
  { value: 'hindi',    label: 'हिंदी',      eng: 'Hindi'    },
  { value: 'gujarati', label: 'ગુજરાતી',    eng: 'Gujarati' },
  { value: 'marathi',  label: 'मराठी',      eng: 'Marathi'  },
  { value: 'telugu',   label: 'తెలుగు',     eng: 'Telugu'   },
];

const EXAMPLES = {
  hindi: 'कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है। विशेषज्ञों का दावा है कि यह वायरस गर्म पानी से मर जाता है।',
  gujarati: 'ભારત સરકારે આ વર્ષે નવી શૈક્ષણિક નીતિ 2024 જાહેર કરી છે.',
  marathi: 'भारत सरकारने आज नवीन शिक्षण धोरण 2024 जाहीर केले.',
  telugu: 'భారత ప్రభుత్వం ఈ సంవత్సరం కొత్త విద్యా విధానాన్ని ప్రకటించింది.',
};

export default function AnalysePage() {
  const [text, setText]         = useState('');
  const [language, setLanguage] = useState('hindi');
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  const handleAnalyse = async () => {
    if (!text.trim()) { setError('Please enter article text.'); return; }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await api.analyze(text.trim(), language);
      setResult(data);
    } catch (e) {
      setError(e.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const handleExample = () => {
    setText(EXAMPLES[language] || EXAMPLES.hindi);
    setResult(null);
  };

  return (
    <div className={styles.page}>
      {/* ── Hero ── */}
      <div className={styles.hero}>
        <div className={styles.heroEyebrow}>Multilingual fake news detection</div>
        <h1 className={styles.heroTitle}>
          Verify the<br />
          <span className={styles.heroAccent}>truth</span>
        </h1>
        <p className={styles.heroSub}>
          Powered by IndicBERT + BiLSTM ensemble. Results verified on Ethereum.
        </p>
      </div>

      {/* ── Input panel ── */}
      <div className={styles.inputPanel}>
        {/* Language selector */}
        <div className={styles.langBar}>
          {LANGUAGES.map(l => (
            <button
              key={l.value}
              className={`${styles.langBtn} ${language === l.value ? styles.langActive : ''}`}
              onClick={() => { setLanguage(l.value); setResult(null); setText(''); }}
            >
              <span className={styles.langNative}>{l.label}</span>
              <span className={styles.langEng}>{l.eng}</span>
            </button>
          ))}
        </div>

        {/* Textarea */}
        <div className={styles.textareaWrap}>
          <textarea
            className={styles.textarea}
            value={text}
            onChange={e => { setText(e.target.value); setResult(null); }}
            placeholder={`Paste ${LANGUAGES.find(l => l.value === language)?.eng || ''} article text here…`}
            rows={7}
          />
          <div className={styles.textareaFooter}>
            <span className={styles.charCount}>{text.length} chars</span>
            <button className={styles.exampleBtn} onClick={handleExample}>Load example</button>
          </div>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        {/* Actions */}
        <div className={styles.actions}>
          <button
            className={styles.analyseBtn}
            onClick={handleAnalyse}
            disabled={loading || !text.trim()}
          >
            {loading ? (
              <><span className={styles.spinner} /> Analysing…</>
            ) : (
              'Analyse article'
            )}
          </button>
          {text && (
            <button className={styles.clearBtn} onClick={() => { setText(''); setResult(null); }}>
              Clear
            </button>
          )}
        </div>
      </div>

      {/* ── Result ── */}
      {result && (
        <div className={styles.resultWrap}>
          <VerdictCard result={result} text={text} language={language} />
        </div>
      )}
    </div>
  );
}
