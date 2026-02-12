import React, { useState } from 'react';
import { useBrain19 } from './hooks/useBrain19';
import Brain19Visualizer from './Brain19Visualizer';

const App = () => {
  const { snapshot, connected, loading, lastAnswer, ask, ingest, refreshSnapshot, fetchSnapshot } = useBrain19();
  const [question, setQuestion] = useState('');
  const [ingestText, setIngestText] = useState('');
  const [asking, setAsking] = useState(false);

  const handleAsk = (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    ask(question);
    setQuestion('');
  };

  const handleIngest = (e) => {
    e.preventDefault();
    if (!ingestText.trim()) return;
    ingest(ingestText);
    setIngestText('');
  };

  // Reset asking when answer arrives
  React.useEffect(() => {
    if (lastAnswer) setAsking(false);
  }, [lastAnswer]);

  if (loading && !snapshot) {
    return (
      <div style={styles.loading}>
        <div style={styles.spinner} />
        <p>Connecting to Brain19...</p>
        <button onClick={fetchSnapshot} style={styles.retryBtn}>Retry via REST</button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>🧠 Brain19</h1>
        <div style={styles.headerRight}>
          <span style={{
            ...styles.statusDot,
            backgroundColor: connected ? '#4ade80' : '#f87171',
          }} />
          <span style={styles.statusText}>
            {connected ? 'LIVE' : 'Disconnected'}
          </span>
          {snapshot?.status?.concepts && (
            <span style={styles.statBadge}>{snapshot.status.concepts} concepts</span>
          )}
          <button onClick={refreshSnapshot} style={styles.refreshBtn}>↻</button>
        </div>
      </div>

      {/* Main content */}
      <div style={styles.main}>
        {/* Left: Visualization */}
        <div style={styles.vizArea}>
          {snapshot ? (
            <Brain19Visualizer snapshot={snapshot} />
          ) : (
            <div style={styles.noData}>No snapshot data</div>
          )}
        </div>

        {/* Right: Interaction panel */}
        <div style={styles.panel}>
          {/* Ask */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>💬 Ask Brain19</h3>
            <form onSubmit={handleAsk} style={styles.form}>
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question..."
                style={styles.input}
                disabled={!connected}
              />
              <button type="submit" style={styles.submitBtn} disabled={!connected || asking}>
                {asking ? '...' : 'Ask'}
              </button>
            </form>
            {lastAnswer && (
              <div style={styles.answerBox}>
                <pre style={styles.answerText}>{lastAnswer}</pre>
              </div>
            )}
          </div>

          {/* Ingest */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>📥 Ingest Knowledge</h3>
            <form onSubmit={handleIngest} style={styles.form}>
              <textarea
                value={ingestText}
                onChange={(e) => setIngestText(e.target.value)}
                placeholder="Paste knowledge to ingest..."
                style={styles.textarea}
                disabled={!connected}
              />
              <button type="submit" style={styles.submitBtn} disabled={!connected}>
                Ingest
              </button>
            </form>
          </div>

          {/* Status */}
          {snapshot?.status && (
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>📊 Status</h3>
              <div style={styles.statusGrid}>
                {Object.entries(snapshot.status).map(([k, v]) => (
                  <div key={k} style={styles.statusItem}>
                    <span style={styles.statusKey}>{k}</span>
                    <span style={styles.statusVal}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: { minHeight: '100vh', background: '#0a0a0f', color: '#e0e0e0', fontFamily: "'Inter', sans-serif" },
  loading: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: 16 },
  spinner: { width: 40, height: 40, border: '3px solid #333', borderTopColor: '#6366f1', borderRadius: '50%', animation: 'spin 1s linear infinite' },
  retryBtn: { padding: '8px 16px', background: '#333', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 24px', borderBottom: '1px solid #1e1e2e', background: '#0f0f1a' },
  title: { margin: 0, fontSize: 22, fontWeight: 700, background: 'linear-gradient(135deg, #6366f1, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 12 },
  statusDot: { width: 10, height: 10, borderRadius: '50%' },
  statusText: { fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 },
  statBadge: { fontSize: 12, padding: '4px 8px', background: '#1e1e2e', borderRadius: 4 },
  refreshBtn: { background: 'none', border: '1px solid #333', color: '#aaa', padding: '4px 10px', borderRadius: 4, cursor: 'pointer', fontSize: 16 },
  main: { display: 'flex', height: 'calc(100vh - 57px)' },
  vizArea: { flex: 1, overflow: 'hidden' },
  noData: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#555' },
  panel: { width: 380, borderLeft: '1px solid #1e1e2e', overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 16 },
  section: { background: '#111122', borderRadius: 8, padding: 16 },
  sectionTitle: { margin: '0 0 12px', fontSize: 14, fontWeight: 600, color: '#a78bfa' },
  form: { display: 'flex', flexDirection: 'column', gap: 8 },
  input: { padding: '10px 12px', background: '#0a0a0f', border: '1px solid #2a2a3e', borderRadius: 6, color: '#e0e0e0', fontSize: 14, outline: 'none' },
  textarea: { padding: '10px 12px', background: '#0a0a0f', border: '1px solid #2a2a3e', borderRadius: 6, color: '#e0e0e0', fontSize: 14, outline: 'none', minHeight: 80, resize: 'vertical', fontFamily: 'inherit' },
  submitBtn: { padding: '8px 16px', background: '#6366f1', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 14 },
  answerBox: { marginTop: 12, padding: 12, background: '#0a0a0f', borderRadius: 6, border: '1px solid #2a2a3e', maxHeight: 200, overflowY: 'auto' },
  answerText: { margin: 0, fontSize: 13, whiteSpace: 'pre-wrap', lineHeight: 1.5, color: '#c0c0d0' },
  statusGrid: { display: 'flex', flexDirection: 'column', gap: 6 },
  statusItem: { display: 'flex', justifyContent: 'space-between', fontSize: 13 },
  statusKey: { color: '#888' },
  statusVal: { color: '#e0e0e0', fontWeight: 500 },
};

export default App;
