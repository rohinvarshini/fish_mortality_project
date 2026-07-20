import React, { useState } from 'react';
import axios from 'axios';
import { Card, CardHeader } from '../components/Card';
import { Play, ShieldAlert, Activity, CheckCircle, AlertTriangle } from 'lucide-react';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

export function ModelDashboard() {
  const [formData, setFormData] = useState({
    DO: '',
    pH: '',
    temperature: '',
    turbidity: '',
    ammonia: '',
    fish_weight: ''
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to connect to the model API. Ensure the Python server is running on port 8000.");
    } finally {
      setIsLoading(false);
    }
  };

  const riskColors = {
    Low: 'bg-emerald-50 text-emerald-600 border-emerald-200',
    Moderate: 'bg-yellow-50 text-yellow-600 border-yellow-200',
    High: 'bg-red-50 text-red-600 border-red-200',
  };

  const InputField = ({ id, label, placeholder, unit }) => (
    <div className="space-y-1.5">
      <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <input
          type="number"
          step="0.01"
          name={id}
          value={formData[id]}
          onChange={handleChange}
          placeholder={placeholder}
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-mono"
          required
        />
        {unit && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-semibold text-slate-400 bg-white px-1">
            {unit}
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* LEFT COLUMN: Input Form */}
      <div className="lg:col-span-4">
        <Card className="h-full border-t-4 border-t-blue-600 rounded-t-none">
          <CardHeader title="Model Input Console" subtitle="Enter real-time environmental metrics" />
          
          <form onSubmit={handleSubmit} className="space-y-4 mt-2">
            <InputField id="DO" label="Dissolved Oxygen (DO)" placeholder="e.g., 5.5" unit="mg/L" />
            <InputField id="pH" label="pH Level" placeholder="e.g., 7.2" unit="" />
            <InputField id="temperature" label="Temperature" placeholder="e.g., 26.5" unit="°C" />
            <InputField id="turbidity" label="Turbidity" placeholder="e.g., 15.0" unit="NTU" />
            <InputField id="ammonia" label="Ammonia (NH3)" placeholder="e.g., 0.5" unit="mg/L" />
            <InputField id="fish_weight" label="Avg Fish Weight" placeholder="e.g., 551.0" unit="g" />

            <button 
              type="submit"
              disabled={isLoading}
              className="w-full mt-6 py-3 px-4 bg-black hover:bg-slate-800 text-white rounded-lg font-medium transition-all shadow-md disabled:opacity-50 flex justify-center items-center group"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  <Play size={18} className="mr-2 group-hover:text-blue-400 transition-colors" />
                  Run Inference
                </>
              )}
            </button>
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 text-red-600 text-sm rounded-lg mt-3">
                {error}
              </div>
            )}
          </form>
        </Card>
      </div>

      {/* RIGHT COLUMN: Output Dashboard */}
      <div className="lg:col-span-8 space-y-6">
        <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Real-Time Inference Results</h2>
        
        {!result && !isLoading && (
          <div className="h-64 border-2 border-dashed border-slate-200 rounded-xl flex flex-col items-center justify-center text-slate-400 bg-slate-50/50">
            <Activity size={48} className="mb-4 opacity-50" />
            <p>Enter data and click "Run Inference" to evaluate the models.</p>
          </div>
        )}

        {result && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* CLASSIFIER RESULT */}
              <Card className="col-span-1 md:col-span-2 max-w-xl mx-auto w-full">
                <CardHeader title="Risk Classification" subtitle="fish_mortality_lstm_model.pth output" />
                <div className="flex flex-col items-center justify-center py-8">
                  <div className={`px-12 py-4 rounded-full border-2 font-black text-3xl tracking-wider ${riskColors[result.risk_label]}`}>
                    {result.risk_label.toUpperCase()} RISK
                  </div>
                  <div className="mt-8 w-full max-w-sm">
                    <div className="flex justify-between text-sm font-semibold mb-1 text-slate-600">
                      <span>Model Confidence</span>
                      <span>{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className={cn("h-full rounded-full transition-all duration-1000", result.risk_label === 'High' ? 'bg-red-500' : 'bg-blue-500')}
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </Card>

            </div>

            {/* PIPELINE LOGS */}
            <Card className="bg-slate-900 border-none">
              <div className="flex items-center justify-between mb-3 text-slate-400 text-xs font-mono">
                <span>SYSTEM CONSOLE LOGS</span>
                <span>STATUS: SUCCESS</span>
              </div>
              <div className="font-mono text-sm text-emerald-400 bg-black/50 p-4 rounded-lg overflow-x-auto space-y-1">
                <p>&gt; Received payload [DO: {formData.DO}, pH: {formData.pH}, Weight: {formData.fish_weight}...]</p>
                <p>&gt; Applying scaler.pkl transform (6 features)...</p>
                <p>&gt; Tensor reshaped to (1, 24, 6) assuming steady-state history...</p>
                <p>&gt; Executing forward pass on Unified LSTM model...</p>
                <p>&gt; Logits returned. Applying structural Softmax...</p>
                <p className="text-blue-400 pt-2 font-bold">&gt; INFERENCE COMPLETE. LATENCY: {result.latency_ms}ms.</p>
              </div>
            </Card>

          </motion.div>
        )}
      </div>
    </div>
  );
}
