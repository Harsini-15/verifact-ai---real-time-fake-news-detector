import React, { useState } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Search, 
  Loader2, 
  ExternalLink, 
  Info, 
  CheckCircle2, 
  XCircle,
  BarChart3,
  BrainCircuit,
  Globe
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { preprocess } from './utils/nlp';

interface Article {
  title: string;
  source: string;
  url: string;
}

interface PredictionResult {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  keywords: string[];
  explanation: string;
  importantWords: string[];
  articles: Article[];
}

export default function App() {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCheckNews = async () => {
    if (!inputText || inputText.length < 10) {
      setError('Please enter at least 10 characters of news text.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to analyze text');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-white font-sans selection:bg-emerald-500/30">
      {/* Background Glow */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="flex items-center justify-center mb-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <ShieldCheck className="text-black w-6 h-6" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight">VeriFact <span className="text-emerald-500">AI</span></h1>
          </div>
        </header>

        {/* Hero Section */}
        <div className="text-center mb-12">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-6xl font-bold mb-6 tracking-tight"
          >
            Detect Truth in <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-500">Real-Time</span>
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-zinc-400 text-lg max-w-2xl mx-auto"
          >
            Advanced NLP and Machine Learning to classify news authenticity. 
            Backed by real-time search grounding for maximum accuracy.
          </motion.p>
        </div>

        {/* Main Input Area */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-2xl backdrop-blur-xl shadow-2xl mb-12"
        >
          <div className="relative mb-4">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste news article text or headline here..."
              className="w-full h-48 bg-zinc-950/50 border border-zinc-800 rounded-xl p-4 text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all resize-none"
            />
          </div>
          
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4 text-xs text-zinc-500">
              <div className="flex items-center gap-1.5">
                <BrainCircuit className="w-4 h-4" />
                <span>NLP Preprocessing</span>
              </div>
              <div className="flex items-center gap-1.5">
                <BarChart3 className="w-4 h-4" />
                <span>TF-IDF Vectorization</span>
              </div>
            </div>
            
            <button
              onClick={handleCheckNews}
              disabled={loading}
              className="w-full md:w-auto px-8 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:bg-zinc-700 text-black font-bold rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  Check News
                </>
              )}
            </button>
          </div>
        </motion.div>

        {/* Error Message */}
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-3"
          >
            <XCircle className="w-5 h-5" />
            {error}
          </motion.div>
        )}

        {/* Results Section */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="max-w-3xl mx-auto space-y-8"
            >
              {/* Prediction Card */}
              <div className={`p-8 rounded-3xl border ${result.prediction === 'REAL' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'}`}>
                <div className="flex flex-col md:flex-row items-start justify-between gap-6 mb-8">
                  <div>
                    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-3 ${result.prediction === 'REAL' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                      {result.prediction === 'REAL' ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                      {result.prediction} News Detected
                    </div>
                    <h3 className="text-3xl font-bold mb-2">
                      {result.prediction === 'REAL' ? 'This information appears authentic.' : 'This information may be misleading.'}
                    </h3>
                  </div>
                  <div className="text-left md:text-right">
                    <div className="text-sm text-zinc-500 mb-1 font-medium uppercase tracking-widest">Confidence</div>
                    <div className={`text-4xl font-black ${result.prediction === 'REAL' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {(result.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <p className="text-zinc-300 leading-relaxed mb-8 text-lg">
                  {result.explanation}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  <div className="p-4 bg-zinc-950/50 rounded-2xl border border-zinc-800">
                    <div className="flex items-center gap-2 text-zinc-400 text-sm mb-3">
                      <Info className="w-4 h-4" />
                      <span>Key Indicators</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {result.importantWords.map((word, i) => (
                        <span key={i} className="px-2 py-1 bg-zinc-800 rounded-md text-xs text-zinc-300 border border-zinc-700">
                          {word}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="p-4 bg-zinc-950/50 rounded-2xl border border-zinc-800">
                    <div className="flex items-center gap-2 text-zinc-400 text-sm mb-3">
                      <Globe className="w-4 h-4" />
                      <span>Extracted Keywords</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {result.keywords.map((word, i) => (
                        <span key={i} className="px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded-md text-xs border border-emerald-500/20">
                          {word}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Trusted Sources Integrated */}
                <div className="pt-8 border-t border-zinc-800/50">
                  <h4 className="text-lg font-bold mb-6 flex items-center gap-2">
                    <Globe className="w-5 h-5 text-emerald-400" />
                    {result.prediction === 'REAL' ? 'Trusted Sources' : 'Contextual Evidence'}
                  </h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {result.articles.length > 0 ? (
                      result.articles.map((article, i) => (
                        <motion.div 
                          key={i}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1 * i }}
                          className="p-4 bg-zinc-950/50 border border-zinc-800 rounded-2xl hover:border-emerald-500/30 transition-all group flex flex-col justify-between"
                        >
                          <div>
                            <div className="text-[10px] text-emerald-500 font-bold mb-2 uppercase tracking-wider">{article.source}</div>
                            <h5 className="text-xs font-medium text-zinc-200 mb-4 line-clamp-2 group-hover:text-white transition-colors">
                              {article.title}
                            </h5>
                          </div>
                          <a 
                            href={article.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-[10px] font-bold text-white bg-zinc-800 hover:bg-zinc-700 px-3 py-2 rounded-lg transition-all w-full justify-center"
                          >
                            Read Full Article
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </motion.div>
                      ))
                    ) : (
                      <div className="col-span-full text-center py-4 text-zinc-500 text-sm">
                        No direct matches found in real-time search.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="mt-24 pt-12 border-t border-zinc-900 text-center">
          <p className="text-zinc-600 text-sm">
            &copy; 2026 VeriFact AI. Powered by Custom ML Models (LR, NB, LSTM) & Fast API.
          </p>
        </footer>
      </div>
    </div>
  );
}
