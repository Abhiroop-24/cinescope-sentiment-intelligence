import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const analyze = async (event) => {
    event.preventDefault();
    const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    setResult(await response.json());
  };
  return <main><header><b>CINESCOPE / 20</b><span>PES · UE24CS352A</span></header>
    <h1>What did<br/>you think?</h1><p>An explainable IMDb sentiment model.</p>
    <form onSubmit={analyze}><textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste a movie review here…" required/><button>ANALYZE REVIEW</button></form>
    {result && <section><small>MODEL VERDICT</small><h2>{result.label}</h2><p>Positive probability: {(result.probability * 100).toFixed(1)}%</p><div className="terms">{(result.explanation || []).map(item => <span key={item.term}>{item.term} · {item.direction}</span>)}</div></section>}
  </main>;
}
createRoot(document.getElementById('root')).render(<App />);
