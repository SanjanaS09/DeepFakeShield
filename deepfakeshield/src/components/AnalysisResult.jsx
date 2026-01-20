import React, { useState } from 'react';
import './AnalysisResult.css';

export default function AnalysisResult({ result, loading }) {
  const [expandedSection, setExpandedSection] = useState(null);

  if (loading) {
    return (
      <div className="analysis-container loading">
        <div className="spinner"></div>
        <p>Analyzing media...</p>
      </div>
    );
  }

  if (!result) return null;

  // ✅ FIX: Properly extract data
  const prediction = result?.prediction || result?.label || 'UNKNOWN';
  const confidence = Number(result?.confidence) || 0;
  const probabilities = result?.probabilities || { REAL: 0, FAKE: 0 };
  const xai = result?.xai || {};

  // ✅ FIX: Determine if fake
  const isFake = prediction === 'FAKE' || prediction === 1;
  const isReal = prediction === 'REAL' || prediction === 0;

  // ✅ FIX: Calculate probabilities safely
  const realProb = Number(probabilities.REAL) || (isReal ? confidence : 1 - confidence);
  const fakeProb = Number(probabilities.FAKE) || (isFake ? confidence : 1 - confidence);

  // ✅ FIX: Format percentages
  const realPercent = (realProb * 100).toFixed(2);
  const fakePercent = (fakeProb * 100).toFixed(2);
  const confidencePercent = (confidence * 100).toFixed(2);

  // ✅ Determine styling
  const resultClass = isFake ? 'result-fake' : 'result-real';
  const statusIcon = isFake ? '🚨' : '✅';
  const statusText = isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC CONTENT';

  return (
    <div className={`analysis-result ${resultClass}`}>
      {/* ✅ MAIN RESULT SECTION */}
      <div className="result-header">
        <div className="result-status">
          <span className="status-icon">{statusIcon}</span>
          <div className="status-info">
            <h2>{statusText}</h2>
            <p className="confidence-badge">
              Confidence: <strong>{confidencePercent}%</strong>
            </p>
          </div>
        </div>

        {/* ✅ CONFIDENCE CIRCLE */}
        <div className="confidence-circle">
          <svg viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" className="bg-circle" />
            <circle
              cx="50"
              cy="50"
              r="45"
              className="progress-circle"
              style={{
                strokeDasharray: `${confidence * 282.7} 282.7`
              }}
            />
          </svg>
          <div className="circle-text">
            <span className="percentage">{confidencePercent}%</span>
            <span className="label">{isFake ? 'Fake' : 'Real'}</span>
          </div>
        </div>
      </div>

      {/* ✅ PROBABILITY BREAKDOWN */}
      <div className="probability-section">
        <h3>📊 Probability Analysis</h3>
        <div className="probability-bars">
          {/* Real Probability */}
          <div className="prob-item">
            <div className="prob-label">
              <span>🟢 Real Probability</span>
              <span className="prob-value">{realPercent}%</span>
            </div>
            <div className="prob-bar">
              <div 
                className="prob-fill real" 
                style={{ width: `${realProb * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Fake Probability */}
          <div className="prob-item">
            <div className="prob-label">
              <span>🔴 Fake Probability</span>
              <span className="prob-value">{fakePercent}%</span>
            </div>
            <div className="prob-bar">
              <div 
                className="prob-fill fake" 
                style={{ width: `${fakeProb * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* ✅ XAI EXPLANATIONS */}
      <div className="xai-section">
        <h3>🔍 Analysis & Reasoning</h3>

        {/* Explanation */}
        {xai?.explanation && (
          <div className="xai-item">
            <h4>📌 Conclusion</h4>
            <p className="explanation">{xai.explanation}</p>
          </div>
        )}

        {/* Reasoning */}
        {xai?.reasoning && Array.isArray(xai.reasoning) && (
          <div className="xai-item">
            <h4>💭 Why We Believe This</h4>
            <ul className="reasoning-list">
              {xai.reasoning.map((reason, idx) => (
                <li key={idx}>
                  <span className="bullet">•</span>
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Key Indicators */}
        {xai?.key_indicators && typeof xai.key_indicators === 'object' && (
          <div className="xai-item">
            <h4>⚙️ Technical Indicators</h4>
            <div className="indicators-grid">
              {Object.entries(xai.key_indicators).map(([key, value]) => (
                <div key={key} className="indicator-card">
                  <span className="indicator-name">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <span className={`indicator-value ${
                    ['High', 'Natural', 'Minimal', 'Normal', 'Sharp', 'Consistent'].includes(String(value))
                      ? 'good'
                      : 'bad'
                  }`}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {xai?.recommendations && Array.isArray(xai.recommendations) && (
          <div className="xai-item">
            <h4>✨ Recommendations</h4>
            <ul className="recommendations-list">
              {xai.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Confidence Level */}
        {xai?.confidence_level && (
          <div className="xai-item">
            <h4>📈 Confidence Level</h4>
            <p className="confidence-level">{xai.confidence_level}</p>
          </div>
        )}
      </div>

      {/* ✅ DEBUG INFO */}
      <details className="debug-info">
        <summary>📋 Full Response Data</summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  );
}
