import React from 'react';
import './ResultsWindow.css';

const ResultsWindow = ({ result, mediaType, file }) => {
  const confidence = (result.confidence * 100).toFixed(2);
  const isFake = result.is_fake;

  return (
    <div className={`results-window ${isFake ? 'fake' : 'real'}`}>
      <div className="results-header">
        <h3>{isFake ? '⚠️ ALERT' : '✅ AUTHENTIC'}</h3>
        <p>{isFake ? 'Deepfake Detected' : 'Content Verified as Authentic'}</p>
      </div>

      <div className="results-main">
        <div className="confidence-display">
          <div className={`confidence-meter ${isFake ? 'fake' : 'real'}`}>
            <svg viewBox="0 0 200 200" className="confidence-gauge">
              <circle cx="100" cy="100" r="90" fill="none" stroke="#ddd" strokeWidth="20" />
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke={isFake ? '#f44336' : '#4caf50'}
                strokeWidth="20"
                strokeDasharray={`${(confidence / 100) * 565} 565`}
                strokeDashoffset="-141"
                className="confidence-progress"
              />
              <text x="100" y="115" textAnchor="middle" className="confidence-text">
                {confidence}%
              </text>
            </svg>
          </div>
          <div className="confidence-label">
            <p>Confidence Score</p>
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

          {result.processing_time && (
            <div className="result-item">
              <span className="result-label">Processing Time:</span>
              <span className="result-value">{result.processing_time.toFixed(2)}s</span>
            </div>
          )}

          {result.frames_analyzed && (
            <div className="result-item">
              <span className="result-label">Frames Analyzed:</span>
              <span className="result-value">{result.frames_analyzed}</span>
            </div>
          )}

          {result.features_extracted && (
            <div className="result-item features">
              <span className="result-label">Features Extracted:</span>
              <div className="features-list">
                {typeof result.features_extracted === 'object' ? (
                  Object.entries(result.features_extracted).map(([key, value]) => (
                    <span key={key} className="feature-tag">
                      {key}: {typeof value === 'boolean' ? (value ? '✓' : '✗') : value}
                    </span>
                  ))
                ) : (
                  <span className="feature-tag">{result.features_extracted}</span>
                )}
              </div>
            </div>
          )}
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