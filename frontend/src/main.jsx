import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const examples = {
  positive: 'A beautiful, thoughtful film with brilliant acting and a genuinely moving story.',
  negative: 'A painfully boring movie with weak writing, terrible acting, and no emotional payoff.',
  uncertain: 'The movie had some good moments, but the uneven story and slow pacing made it difficult to enjoy.'
};

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [meta, setMeta] = useState(null);
  const [dark, setDark] = useState(false);
  const [history, setHistory] = useState([]);
  useEffect(() => { fetch('/api/metadata').then(r => r.json()).then(setMeta).catch(() => {}); }, []);
  const analyze = async event => {
    event?.preventDefault();
    if (!text.trim()) return;
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const data = await response.json(); setResult(data);
    setHistory(items => [{label:data.label, probability:data.probability}, ...items].slice(0, 5));
  };
  const ranking = useMemo(() => (meta?.comparison || []).slice().sort((a,b) => b.accuracy-a.accuracy).slice(0, 8), [meta]);
  const model = meta?.model || {};
  return <div className={dark ? 'app dark' : 'app'}>
    <header><b>CINESCOPE / 20</b><span>PES · UE24CS352A</span><button className="theme" onClick={() => setDark(!dark)}>{dark ? '☼' : '☾'} mode</button></header>
    <section className="hero"><div><small>EXPLAINABLE MOVIE REVIEW INTELLIGENCE</small><h1>What did<br/><em>you think?</em></h1><p>Read the feeling behind the words. A CUDA-trained sentiment model with transparent, human-readable explanations.</p></div><div className="status"><i/> GPU ACCELERATION ACTIVE<br/><strong>{model.gpu || 'RTX 2050'}</strong></div></section>
    <section className="lab"><div className="section-title"><small>01 / LIVE ANALYSIS</small><h2>Make a call.</h2></div>
      <div className="examples">{Object.keys(examples).map(key => <button key={key} onClick={() => setText(examples[key])}>Try {key}</button>)}</div>
      <form onSubmit={analyze}><textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste a movie review here…"/><button className="primary">ANALYZE REVIEW <span>↗</span></button></form>
      {result && <div className="verdict"><div><small>MODEL VERDICT</small><h3>{result.label}</h3><p>{(result.probability*100).toFixed(1)}% positive probability</p><div className="gauge"><span style={{width:`${result.probability*100}%`}}/></div></div><div><small>INFLUENTIAL LANGUAGE</small><div className="terms">{result.explanation?.map(item => <span className={item.direction} key={item.term}>{item.term}<b>{item.direction === 'positive' ? '+' : '−'}</b></span>)}</div></div></div>}
    </section>
    <section className="stats"><div className="section-title"><small>02 / THE EXPERIMENT</small><h2>By the numbers.</h2></div><div className="stat-grid"><Stat value={model.train_samples?.toLocaleString() || '25k'} label="training reviews"/><Stat value={model.test_samples?.toLocaleString() || '25k'} label="test reviews"/><Stat value={`${((model.metrics?.accuracy || 0)*100).toFixed(1)}%`} label="test accuracy"/><Stat value={model.max_features?.toLocaleString() || '12k'} label="TF-IDF features"/></div></section>
    <section className="comparison"><div className="section-title"><small>03 / MODEL BENCHMARK</small><h2>How it compares.</h2></div><div className="bars">{ranking.map((item, i) => <div className="bar-row" key={`${item.model}-${item.representation}`}><label>{item.model.replaceAll('_',' ')} <small>{item.representation}</small></label><div><span style={{width:`${item.accuracy*100}%`}}/><b>{(item.accuracy*100).toFixed(1)}%</b></div></div>)}</div></section>
    <section className="lower"><div><small>04 / RECENT ANALYSES</small><h2>History.</h2>{history.length ? history.map((item, i) => <p className="history" key={i}><b className={item.label}>{item.label}</b><span>{(item.probability*100).toFixed(1)}% positive probability</span></p>) : <p className="muted">Your last five predictions will appear here.</p>}</div><div><small>05 / METHOD</small><h2>How it works.</h2><p className="muted">Reviews are normalized, converted into TF-IDF features, and classified by logistic regression. Each explanation term is a learned feature contribution, not a causal claim.</p></div></section>
    <footer>CINESCOPE · REPRODUCTION ATTEMPT OF WU & SHIN, 2020 · <a href="#top">BACK TO TOP ↑</a></footer>
  </div>;
}
function Stat({value,label}) { return <div className="stat"><strong>{value}</strong><span>{label}</span></div>; }
createRoot(document.getElementById('root')).render(<App/>);
