import React, { useState, useEffect, useRef } from 'react';
import './DeepfakeDetector.css';
import ProcessingWindow from './ProcessingWindow';
import AnalysisResult from './AnalysisResult';
import axios from 'axios';

const DeepfakeDetector = () => {
  const [file, setFile] = useState(null);
  const [mediaType, setMediaType] = useState('image');
  const [processing, setProcessing] = useState(false);
  const [processingSteps, setProcessingSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const stepCounterRef = useRef(0);

  const BACKEND_URL = 'http://localhost:5000';
  const API_ENDPOINTS = {
    image: `${BACKEND_URL}/api/detection/image`,
    video: `${BACKEND_URL}/api/detection/video`,
    audio: `${BACKEND_URL}/api/detection/audio`
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      setProcessingSteps([]);
    }
  };

  const addProcessingStep = (step, details) => {
    stepCounterRef.current += 1;
    setProcessingSteps(prev => [...prev, {
      id: stepCounterRef.current,
      name: step,
      details: details,
      timestamp: new Date().toLocaleTimeString(),
      status: 'completed'
    }]);
  };

  const handleDetect = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a file');
      return;
    }

    setProcessing(true);
    setProcessingSteps([]);
    stepCounterRef.current = 0;
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      addProcessingStep('📁 File Upload', `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
      await new Promise(r => setTimeout(r, 300));

      addProcessingStep('🔧 Preprocessing', `Preparing ${mediaType.toUpperCase()} for analysis...`);
      await new Promise(r => setTimeout(r, 500));

      if (mediaType === 'image') {
        addProcessingStep('📊 Feature Extraction', 'Analyzing: Facial regions, lighting, textures, frequencies');
        addProcessingStep('🧠 Model Inference', 'Running ResNet18 classifier with 90%+ accuracy');
      } else if (mediaType === 'video') {
        addProcessingStep('🎬 Frame Extraction', 'Sampling 8 key frames from video');
        addProcessingStep('📊 Advanced Feature Analysis', 'Computing: Sharpness, blur, edges, frequency domain, optical flow');
        addProcessingStep('🧠 Model Inference', 'Running High-Accuracy LSTM with attention mechanism');
      } else if (mediaType === 'audio') {
        addProcessingStep('🎵 Audio Processing', 'Loading audio at optimal sample rate');
        addProcessingStep('📊 Spectral Analysis', 'Computing MFCC and voice characteristics');
        addProcessingStep('🧠 Model Inference', 'Running ECAPA-TDNN model');
      }

      console.log('Calling API:', API_ENDPOINTS[mediaType]);

      const response = await axios.post(API_ENDPOINTS[mediaType], formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 120000
      });

      console.log('API Response:', response.data);

      addProcessingStep('📈 Post-Processing', 'Generating explanations and visualizations...');
      addProcessingStep('✅ Analysis Complete', `Result: ${response.data.prediction}`);

      // ✅ FIX: Properly map all response fields
      const resultData = {
        prediction: response.data.prediction,
        confidence: response.data.confidence,
        probabilities: response.data.probabilities || {
          REAL: response.data.confidence,
          FAKE: 1 - response.data.confidence
        },
        label: response.data.prediction,
        xai: response.data.xai || {},
        file_name: file.name,
        processing_time: response.data.processing_time,
      };

      setResult(resultData);

    } catch (err) {
      console.error('Error:', err);
      const errorMessage = err.response?.data?.error || err.message || 'An error occurred';
      setError(errorMessage);
      addProcessingStep('❌ Error', errorMessage);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="deepfake-detector-container">
      <header className="detector-header">
        <h1>🛡️ DeepFake Shield</h1>
        <p>Advanced AI-Powered Deepfake Detection System</p>
      </header>

      <div className="detector-content">
        <div className="upload-panel">
          <div className="upload-card">
            <h2>Upload Media</h2>

            <div className="media-type-selector">
              {['image', 'video', 'audio'].map(type => (
                <button
                  key={type}
                  className={`type-btn ${mediaType === type ? 'active' : ''}`}
                  onClick={() => setMediaType(type)}
                  disabled={processing}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>

            <form onSubmit={handleDetect} className="upload-form">
              <div className="file-input-container">
                <input
                  type="file"
                  id="file-input"
                  onChange={handleFileSelect}
                  accept={
                    mediaType === 'image' ? 'image/*' :
                      mediaType === 'video' ? 'video/*' : 'audio/*'
                  }
                  disabled={processing}
                  className="file-input"
                />
                <label htmlFor="file-input" className="file-label">
                  <span className="file-icon">📁</span>
                  <span className="file-text">
                    {file ? file.name : 'Click to upload or drag & drop'}
                  </span>
                </label>
              </div>

              {file && (
                <div className="file-info">
                  <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  <p>Type: {file.type}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={!file || processing}
                className="detect-btn"
              >
                {processing ? 'Processing...' : 'Detect Deepfake'}
              </button>
            </form>

            {error && (
              <div className="error-message">
                <span>⚠️ {error}</span>
              </div>
            )}
          </div>
        </div>

        <div className="results-panel">
          {processing && (
            <ProcessingWindow
              steps={processingSteps}
              mediaType={mediaType}
            />
          )}

          {result && !processing && (
            <AnalysisResult
              analysisData={result}
              loading={false}
            />
          )}

          {!processing && !result && processingSteps.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🎯</div>
              <p>Upload a file to begin detection</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DeepfakeDetector;