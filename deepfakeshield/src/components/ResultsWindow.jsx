import React, { useEffect, useState } from 'react';
import './ResultsWindow.css';

const ResultsWindow = ({ result, isLoading }) => {
  const [displayResult, setDisplayResult] = useState(null);

  useEffect(() => {
    if (result) {
      console.log('📊 Result received:', result);
      setDisplayResult(result);
    }
  }, [result]);

  if (isLoading) {
    return (
      <div className="results-window loading">
        <div className="spinner"></div>
        <p>Analyzing media...</p>
      </div>
    );
  }

  if (!displayResult) {
    return (
      <div className="results-window empty">
        <p>Upload media to see detection results</p>
      </div>
    );
  }

  const prediction = displayResult.prediction || displayResult.label || 'UNKNOWN';
  const confidence = (displayResult.confidence || 0) * 100;
  const isDeepfake = prediction === 'FAKE';
  const xai = displayResult.xai || {};

  return (
    <div className={`results-window ${isDeepfake ? 'fake' : 'real'}`}>
      {/* Header */}
      <div className="result-header">
        <div className={`result-badge ${isDeepfake ? 'fake' : 'real'}`}>
          {isDeepfake ? '🚨 DEEPFAKE' : '✅ AUTHENTIC'}
        </div>
        <h2>{xai.explanation || (isDeepfake ? 'Deepfake Detected' : 'Authentic Content')}</h2>
      </div>

      {/* Confidence Bar */}
      <div className="confidence-section">
        <div className="confidence-label">
          <span>Detection Confidence</span>
          <span className="confidence-value">{confidence.toFixed(1)}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className={`confidence-fill ${isDeepfake ? 'fake' : 'real'}`}
            style={{ width: `${confidence}%` }}
          ></div>
        </div>
        <div className="confidence-level">{xai.confidence_level || 'Unknown'}</div>
      </div>

      {/* Probabilities */}
      <div className="probabilities">
        <div className="prob-item real">
          <span>REAL</span>
          <span className="prob-value">{((displayResult.probabilities?.REAL || 0) * 100).toFixed(1)}%</span>
        </div>
        <div className="prob-item fake">
          <span>FAKE</span>
          <span className="prob-value">{((displayResult.probabilities?.FAKE || 0) * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Reasoning */}
      {xai.reasoning && xai.reasoning.length > 0 && (
        <div className="reasoning-section">
          <h3>📋 Analysis Reasoning</h3>
          <ul className="reasoning-list">
            {xai.reasoning.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Key Indicators */}
      {xai.key_indicators && Object.keys(xai.key_indicators).length > 0 && (
        <div className="indicators-section">
          <h3>🔍 Key Indicators</h3>
          <div className="indicators-grid">
            {Object.entries(xai.key_indicators).map(([key, value]) => (
              <div key={key} className="indicator-item">
                <span className="indicator-label">{key.replace(/_/g, ' ')}</span>
                <span className={`indicator-value ${typeof value === 'string' ? value.toLowerCase() : ''}`}>
                  {value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {xai.recommendations && xai.recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>💡 Recommendations</h3>
          <ul className="recommendations-list">
            {xai.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      {/* File Info */}
      <div className="file-info">
        <p><strong>File:</strong> {displayResult.file_name}</p>
        {displayResult.frames_analyzed && (
          <p><strong>Frames Analyzed:</strong> {displayResult.frames_analyzed}</p>
        )}
      </div>

      <style>{`
        .results-window {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border-radius: 12px;
          padding: 24px;
          color: #fff;
          border: 2px solid rgba(255,255,255,0.1);
          max-height: 600px;
          overflow-y: auto;
        }

        .results-window.loading,
        .results-window.empty {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 300px;
          text-align: center;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(255,255,255,0.2);
          border-top-color: #2196F3;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .result-header {
          margin-bottom: 20px;
          padding-bottom: 16px;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .result-badge {
          display: inline-block;
          padding: 8px 16px;
          border-radius: 20px;
          font-weight: bold;
          font-size: 14px;
          margin-bottom: 12px;
        }

        .result-badge.fake {
          background: rgba(244, 67, 54, 0.2);
          color: #FF6B6B;
          border: 1px solid #FF6B6B;
        }

        .result-badge.real {
          background: rgba(76, 175, 80, 0.2);
          color: #4CAF50;
          border: 1px solid #4CAF50;
        }

        .result-header h2 {
          margin: 0;
          font-size: 24px;
        }

        .confidence-section {
          background: rgba(255,255,255,0.05);
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .confidence-label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 14px;
        }

        .confidence-value {
          font-weight: bold;
          font-size: 18px;
        }

        .confidence-bar {
          width: 100%;
          height: 24px;
          background: rgba(255,255,255,0.1);
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        .confidence-fill {
          height: 100%;
          transition: width 0.5s ease;
          background: linear-gradient(90deg, #4CAF50, #8BC34A);
        }

        .confidence-fill.fake {
          background: linear-gradient(90deg, #FF6B6B, #FF9800);
        }

        .confidence-level {
          font-size: 12px;
          color: #aaa;
          text-align: right;
        }

        .probabilities {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 20px;
        }

        .prob-item {
          background: rgba(255,255,255,0.05);
          padding: 12px;
          border-radius: 6px;
          text-align: center;
          border-left: 4px solid;
        }

        .prob-item.real {
          border-left-color: #4CAF50;
        }

        .prob-item.fake {
          border-left-color: #FF6B6B;
        }

        .prob-value {
          display: block;
          font-size: 20px;
          font-weight: bold;
          margin-top: 4px;
        }

        .reasoning-section,
        .indicators-section,
        .recommendations-section {
          margin-bottom: 20px;
        }

        .reasoning-section h3,
        .indicators-section h3,
        .recommendations-section h3 {
          margin: 0 0 12px 0;
          font-size: 16px;
        }

        .reasoning-list,
        .recommendations-list {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .reasoning-list li,
        .recommendations-list li {
          padding: 8px 12px;
          margin-bottom: 8px;
          background: rgba(255,255,255,0.05);
          border-radius: 6px;
          border-left: 3px solid #2196F3;
          font-size: 13px;
        }

        .indicators-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }

        .indicator-item {
          display: flex;
          flex-direction: column;
          background: rgba(255,255,255,0.05);
          padding: 12px;
          border-radius: 6px;
          font-size: 12px;
        }

        .indicator-label {
          color: #aaa;
          text-transform: capitalize;
        }

        .indicator-value {
          font-weight: bold;
          margin-top: 4px;
        }

        .indicator-value.high,
        .indicator-value.natural,
        .indicator-value.consistent,
        .indicator-value.stable {
          color: #4CAF50;
        }

        .indicator-value.low,
        .indicator-value.unnatural,
        .indicator-value.variable,
        .indicator-value.jumpy {
          color: #FF6B6B;
        }

        .file-info {
          background: rgba(255,255,255,0.05);
          padding: 12px;
          border-radius: 6px;
          font-size: 12px;
          color: #aaa;
        }

        .file-info p {
          margin: 4px 0;
        }

        ::-webkit-scrollbar {
          width: 6px;
        }

        ::-webkit-scrollbar-track {
          background: rgba(255,255,255,0.05);
        }

        ::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.2);
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
};

export default ResultsWindow;