import React from 'react';
import './ResultsWindow.css';

const ResultsWindow = ({ result, mediaType, file }) => {
  if (!result) return null;

  const confidencePct = (result.confidence * 100).toFixed(2);
  const fakePct = (result.fakeProbability * 100).toFixed(2);
  const realPct = (result.realProbability * 100).toFixed(2);

  const isFake = result.isFake;
  const arcRadius = 50;
  const circumference = 2 * Math.PI * arcRadius;
  const arcLength = (confidencePct / 100) * circumference;

  return (
    <div className={`results-window ${isFake ? 'fake' : 'real'}`}>
      <div className="results-header">
        <h3>{isFake ? '⚠️ ALERT' : '✅ AUTHENTIC'}</h3>
        <p>{isFake ? 'Deepfake Detected' : 'Content Verified as Authentic'}</p>
      </div>

      <div className="results-main">
        <div className="confidence-display">
          <div className="confidence-meter">
            <svg className="confidence-gauge" viewBox="0 0 140 140">
              <defs>
                <linearGradient id="arc-gradient-real" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#28919fff" />
                  <stop offset="100%" stopColor="#4caf50" />
                </linearGradient>
                <linearGradient id="arc-gradient-fake" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="rgb(253, 30, 41)" />
                  <stop offset="100%" stopColor="#960e1aff" />
                </linearGradient>
              </defs>

              {/* Background Circle */}
              <circle
                cx="70"
                cy="70"
                r={arcRadius}
                fill="none"
                stroke="rgba(255, 255, 255, 0.1)"
                strokeWidth="8"
                transform="rotate(-90 70 70)"
              />

              {/* Dynamic Arc - Changes based on isFake */}
              <circle
                cx="70"
                cy="70"
                r={arcRadius}
                fill="none"
                stroke={isFake ? 'url(#arc-gradient-fake)' : 'url(#arc-gradient-real)'}
                strokeWidth="8"
                strokeDasharray={`${arcLength} ${circumference}`}
                strokeLinecap="round"
                transform="rotate(-90 70 70)"
                style={{ transition: 'stroke-dasharray 1s ease-out' }}
              />

              {/* Center Text */}
              <text
                x="70"
                y="70"
                textAnchor="middle"
                dy="8"
                className="confidence-text"
                fontSize="32"
                fontWeight="bold"
              >
                {confidencePct}%
              </text>
            </svg>
          </div>
          <div className="confidence-label">
            <p>{isFake ? 'Fake Detected' : 'Authentic'}</p>
          </div>
        </div>

        <div className="results-details">
          <div className="result-item">
            <span className="result-label">Label:</span>
            <span className="result-value">{result.label}</span>
          </div>

          <div className="result-item">
            <span className="result-label">File:</span>
            <span className="result-value">{file?.name}</span>
          </div>

          <div className="result-item">
            <span className="result-label">Media Type:</span>
            <span className="result-value">{mediaType.toUpperCase()}</span>
          </div>

          {result.processingTime && (
            <div className="result-item">
              <span className="result-label">Processing Time:</span>
              <span className="result-value">{result.processingTime.toFixed(2)}s</span>
            </div>
          )}

          {result.framesAnalyzed && (
            <div className="result-item">
              <span className="result-label">Frames Analyzed:</span>
              <span className="result-value">{result.framesAnalyzed}</span>
            </div>
          )}

          {result.featuresExtracted && (
            <div className="result-item features">
              <span className="result-label">Features Extracted:</span>
              <div className="features-list">
                {typeof result.featuresExtracted === 'object' ? (
                  Object.entries(result.featuresExtracted).map(([key, value]) => (
                    <span key={key} className="feature-tag">
                      {key}: {typeof value === 'boolean' ? (value ? '✓' : '✗') : value}
                    </span>
                  ))
                ) : (
                  <span className="feature-tag">{result.featuresExtracted}</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* PROBABILITY BREAKDOWN */}
        <div className="probability-section">
          <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#fff' }}>
            Probability Analysis
          </h3>

          <div className="probability-row">
            <div className="probability-item">
              <div className="prob-label">✅ Real Probability</div>
              <div className="prob-value">{realPct}%</div>
              <div className="prob-bar">
                <div
                  className="prob-fill real-fill"
                  style={{ width: `${realPct}%` }}
                />
              </div>
            </div>

            <div className="probability-item">
              <div className="prob-label">❌ Fake Probability</div>
              <div className="prob-value">{fakePct}%</div>
              <div className="prob-bar">
                <div
                  className="prob-fill fake-fill"
                  style={{ width: `${fakePct}%` }}
                />
              </div>
            </div>
          </div>
          </div>
        </div>


        <div className="results-recommendation">
          {isFake ? (
            <div className="recommendation-warning">
              <h4>⚠️ Warning</h4>
              <p>This content has been detected as a potential deepfake. Please verify the source before sharing.</p>
            </div>
          ) : (
            <div className="recommendation-success">
              <h4>✓ Status</h4>
              <p>This content appears to be authentic based on our analysis.</p>
            </div>
          )}
        </div>
      </div>
      );
};

      export default ResultsWindow;