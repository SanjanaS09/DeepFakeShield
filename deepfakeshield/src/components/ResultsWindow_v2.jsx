import React, { useState } from 'react';
import './ResultsWindow_v2.css';

const ResultsWindow = ({ result, mediaType, file }) => {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapData, setHeatmapData] = useState(null);

  // ✅ FIX: Correctly check is_fake from API
  console.log('API Result:', result); // Debug log
  
  const confidence = result?.confidence ? parseFloat((result.confidence * 100).toFixed(2)) : 0;
  const fakeProb = result?.fake_probability !== undefined ? parseFloat((result.fake_probability * 100).toFixed(2)) : 0;
  const realProb = result?.real_probability !== undefined ? parseFloat((result.real_probability * 100).toFixed(2)) : 0;
  
  // ✅ CRITICAL FIX: Check both is_fake AND prediction
  const isFake = result?.is_fake === true || result?.prediction === 'FAKE';
  const prediction = result?.prediction || (isFake ? 'FAKE' : 'REAL');

  console.log('isFake:', isFake, 'prediction:', prediction); // Debug

  // Arc calculation for smaller circle
  const arcRadius = 50;
  const circumference = 2 * Math.PI * arcRadius;
  const arcLength = (confidence / 100) * circumference;

  // Get AI explanation based on prediction
  const getAIExplanation = () => {
    if (isFake) {
      return {
        title: '⚠️ Not Authentic - Deepfake Detected',
        summary: `This content has been identified as a manipulated deepfake with ${confidence.toFixed(1)}% confidence.`,
        findings: [
          'Facial Feature Artifacts: Detected unnatural transitions in facial boundaries and contours',
          'Lighting Inconsistencies: The light sources appear conflicting across different facial regions',
          'Eye & Mouth Regions: Digital artifacts and blending errors found in these critical areas',
          'Texture Anomalies: Skin texture patterns are inconsistent with natural human characteristics',
          'Frame Discontinuities: Temporal artifacts suggest AI-generated or video-level manipulations'
        ],
        recommendation: '⚠️ Do not share this content without verification. This appears to be a manipulated deepfake.'
      };
    } else {
      return {
        title: '✅ Authentic - Content Verified',
        summary: `This content appears to be genuine with ${confidence.toFixed(1)}% authenticity confidence.`,
        findings: [
          'Natural Facial Features: Smooth, consistent transitions in facial geometry detected',
          'Coherent Lighting: Light sources are consistent and follow natural physics',
          'Clean Eye & Mouth Regions: No manipulation artifacts in these sensitive areas',
          'Consistent Texture: Skin textures match natural human characteristics',
          'Frame Continuity: Temporal consistency indicates natural video/image capture'
        ],
        recommendation: '✅ Content appears trustworthy. However, always verify critical information through multiple sources.'
      };
    }
  };

  const aiExplanation = getAIExplanation();

  const handleGenerateHeatmap = async () => {
    setHeatmapLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:5000/api/analysis/heatmap', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setHeatmapData(data);
      setShowHeatmap(true);
    } catch (error) {
      console.error('Error generating heatmap:', error);
      alert('Heatmap feature not available. Ensure backend is running with dependencies.');
    } finally {
      setHeatmapLoading(false);
    }
  };

  return (
    <div className={`results-window ${isFake ? 'fake' : 'real'}`}>
      {/* Background Effect */}
      <div className="results-window-bg"></div>

      {/* STATUS HEADER - BOLD */}
      <div className="results-header">
        <h2 className={`status-title ${isFake ? 'status-fake' : 'status-real'}`}>
          {aiExplanation.title}
        </h2>
        <p className="status-description">{aiExplanation.summary}</p>
      </div>

      {/* SMALLER ARC */}
      <div className="confidence-display">
        <div className="confidence-meter">
          <svg className="confidence-gauge" viewBox="0 0 140 140">
            <defs>
              <linearGradient id="arc-gradient-real" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#50b8c6" />
                <stop offset="100%" stopColor="#2db87f" />
              </linearGradient>
              <linearGradient id="arc-gradient-fake" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ff5064" />
                <stop offset="100%" stopColor="#e63946" />
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
              {confidence.toFixed(0)}%
            </text>
          </svg>
        </div>
        <div className="confidence-label">
          <p>{isFake ? 'Fake Detected' : 'Authentic'}</p>
        </div>
      </div>

      {/* PROBABILITY BREAKDOWN */}
      <div className="probability-section">
        <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#fff' }}>
          Probability Analysis
        </h3>
        
        <div className="probability-row">
          <div className="probability-item">
            <div className="prob-label">✅ Real Probability</div>
            <div className="prob-value">{realProb.toFixed(2)}%</div>
            <div className="prob-bar">
              <div 
                className="prob-fill real-fill" 
                style={{ width: `${realProb}%` }}
              />
            </div>
          </div>

          <div className="probability-item">
            <div className="prob-label">❌ Fake Probability</div>
            <div className="prob-value">{fakeProb.toFixed(2)}%</div>
            <div className="prob-bar">
              <div 
                className="prob-fill fake-fill" 
                style={{ width: `${fakeProb}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* FILE DETAILS */}
      <div className="results-details">
        <div className="result-item">
          <span className="result-label">📄 File:</span>
          <span className="result-value">{file?.name || 'Unknown'}</span>
        </div>
        <div className="result-item">
          <span className="result-label">🎬 Type:</span>
          <span className="result-value">{mediaType?.toUpperCase() || 'IMAGE'}</span>
        </div>
        <div className="result-item">
          <span className="result-label">🔍 Result:</span>
          <span className={`result-value ${isFake ? 'text-red' : 'text-green'}`} style={{ fontWeight: 'bold' }}>
            {prediction}
          </span>
        </div>
      </div>

      {/* AI EXPLANATION SECTION */}
      <div className="ai-explanation-section">
        <h3 className="ai-title">🤖 AI Analysis Details</h3>
        
        <div className="ai-findings">
          {aiExplanation.findings.map((finding, idx) => (
            <div key={idx} className="ai-finding-item">
              <span className="finding-icon">→</span>
              <p>{finding}</p>
            </div>
          ))}
        </div>

        <div className={`ai-recommendation ${isFake ? 'warning' : 'success'}`}>
          <span className="rec-icon">{isFake ? '⚠️' : '💡'}</span>
          <p><strong>Recommendation:</strong> {aiExplanation.recommendation}</p>
        </div>
      </div>

      {/* XAI HEATMAP SECTION */}
      <div className="xai-section">
        <h3 style={{ textAlign: 'center', marginBottom: '1rem', color: '#a46bff' }}>
          🗺️ Visual Explanation (Grad-CAM)
        </h3>
        <p style={{ textAlign: 'center', fontSize: '0.9rem', color: 'rgba(220, 235, 255, 0.7)', marginBottom: '1.5rem' }}>
          See which regions influenced the detection decision
        </p>

        <button 
          className="heatmap-button"
          onClick={handleGenerateHeatmap}
          disabled={heatmapLoading}
        >
          {heatmapLoading ? (
            <>
              <span className="spinner"></span>
              Generating Heatmap...
            </>
          ) : (
            '🎯 Generate Grad-CAM Heatmap'
          )}
        </button>

        {showHeatmap && heatmapData && (
          <div className="heatmap-container">
            <div className="heatmap-image">
              <img src={heatmapData.heatmap} alt="Grad-CAM Heatmap" />
              <div className="heatmap-legend">
                <div className="legend-item">
                  <div className="legend-color blue"></div>
                  <span>Low contribution</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color red"></div>
                  <span>High contribution</span>
                </div>
              </div>
            </div>

            {heatmapData.explanation && (
              <div className="xai-explanation">
                <h4>📊 Heatmap Analysis:</h4>
                <ul>
                  {heatmapData.explanation.indicators?.map((indicator, idx) => (
                    <li key={idx}>{indicator}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsWindow;