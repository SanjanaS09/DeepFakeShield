// import React, { useState, useRef } from 'react';
// import './DeepfakeDetector.css';
// import ProcessingWindow from './ProcessingWindow';
// import ResultsWindow from './ResultsWindow';
// import axios from 'axios';

// const DeepfakeDetector = () => {
//   const [file, setFile] = useState(null);
//   const [mediaType, setMediaType] = useState('image');
//   const [processing, setProcessing] = useState(false);
//   const [processingSteps, setProcessingSteps] = useState([]);
//   const [result, setResult] = useState(null);
//   const [error, setError] = useState(null);
//   const stepCounterRef = useRef(0); // ✅ FIX: Use counter for unique keys

//   // ✅ FIX: CORRECT API ENDPOINTS
//   const BACKEND_URL = 'http://localhost:5000';

//   const API_ENDPOINTS = {
//     image: `${BACKEND_URL}/api/detection/image`,     // ✅ CHANGED: /api/detection/image
//     video: `${BACKEND_URL}/api/detection/video`,     // ✅ CHANGED: /api/detection/video
//     audio: `${BACKEND_URL}/api/detection/audio`,     // ✅ CHANGED: /api/detection/audio
//   };

//   const handleFileSelect = (e) => {
//     const selectedFile = e.target.files[0];
//     if (selectedFile) {
//       setFile(selectedFile);
//       setError(null);
//       setResult(null);
//       setProcessingSteps([]);  // ✅ Clear previous steps
//     }
//   };

//   const addProcessingStep = (step, details = '') => {
//     // ✅ FIX: Use counter instead of Date.now() to ensure unique keys
//     stepCounterRef.current += 1;

//     setProcessingSteps(prev => [
//       ...prev,
//       {
//         id: stepCounterRef.current,  // ✅ Unique incrementing ID
//         name: step,
//         details: details,
//         timestamp: new Date().toLocaleTimeString(),
//         status: 'completed'
//       }
//     ]);
//   };

//   const handleDetect = async (e) => {
//     e.preventDefault();
//     if (!file) {
//       setError('Please select a file');
//       return;
//     }

//     setProcessing(true);
//     setProcessingSteps([]);
//     stepCounterRef.current = 0;  // ✅ Reset counter
//     setResult(null);
//     setError(null);

//     try {
//       const formData = new FormData();
//       formData.append('file', file);

//       addProcessingStep('📁 File Upload', `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
//       await new Promise(r => setTimeout(r, 500));

//       addProcessingStep('🔧 Preprocessing', `Preparing ${mediaType.toUpperCase()} data...`);
//       await new Promise(r => setTimeout(r, 800));

//       if (mediaType === 'image') {
//         addProcessingStep('📊 Image Analysis', 'Resizing to 224x224, normalizing pixel values');
//         addProcessingStep('✨ Visual Feature Extraction', 'Extracting: Color distribution, texture patterns, face regions');
//       } else if (mediaType === 'video') {
//         addProcessingStep('🎬 Frame Extraction', 'Extracting 8 key frames from video');
//         addProcessingStep('📊 Temporal Analysis', 'Analyzing motion and temporal consistency');
//         addProcessingStep('✨ Temporal Feature Extraction', 'Extracting: Optical flow, frame differences, motion vectors');
//       } else if (mediaType === 'audio') {
//         addProcessingStep('🎵 Audio Loading', 'Loading audio at 22050 Hz sample rate');
//         addProcessingStep('📊 Spectral Analysis', 'Computing MFCC coefficients');
//         addProcessingStep('✨ Audio Feature Extraction', 'Extracting: MFCC, Spectral centroid, Zero-crossing rate');
//       }

//       console.log('🚀 Calling API:', API_ENDPOINTS[mediaType]);

//       // ✅ Make API call with CORRECT endpoint
//       const response = await axios.post(API_ENDPOINTS[mediaType], formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });

//       console.log('✅ API Response:', response.data);

//       addProcessingStep('🧠 Model Inference', `Running ${mediaType} detection model...`);
//       await new Promise(r => setTimeout(r, 600));

//       addProcessingStep('📈 Post-Processing', 'Computing confidence scores and generating report');

//       // ✅ Process response with correct field names from your backend
//       setResult({
//         is_fake: response.data.prediction === 'FAKE',  // ✅ Changed from response.data.label
//         confidence: response.data.confidence,
//         label: response.data.prediction,  // ✅ Changed from response.data.label
//         processing_time: response.data.processing_time,
//         frames_analyzed: response.data.file_info?.frames_analyzed,
//         features_extracted: response.data.feature_breakdown || response.data.temporal_analysis,
//         file_name: file.name,
//       });

//       addProcessingStep('✅ Detection Complete', `Result: ${response.data.prediction}`);

//     } catch (err) {
//       console.error('❌ Error:', err);
//       const errorMessage = err.response?.data?.error || err.message || 'An error occurred';
//       setError(errorMessage);
//       addProcessingStep('❌ Error', errorMessage);
//     } finally {
//       setProcessing(false);
//     }
//   };

//   return (
//     <div className="deepfake-detector-container">
//       <header className="detector-header">
//         <h1>DeepFake Shield</h1>
//         <p>Advanced AI-Powered Deepfake Detection System</p>
//       </header>

//       <div className="detector-content">
//         <div className="upload-panel">
//           <div className="upload-card">
//             <h2>Upload Media</h2>

//             <div className="media-type-selector">
//               {['image', 'video', 'audio'].map(type => (
//                 <button
//                   key={type}  // ✅ Use static string key for buttons
//                   className={`type-btn ${mediaType === type ? 'active' : ''}`}
//                   onClick={() => setMediaType(type)}
//                   disabled={processing}
//                 >
//                   {type.toUpperCase()}
//                 </button>
//               ))}
//             </div>

//             <form onSubmit={handleDetect} className="upload-form">
//               <div className="file-input-container">
//                 <input
//                   type="file"
//                   id="file-input"
//                   onChange={handleFileSelect}
//                   accept={
//                     mediaType === 'image' ? 'image/*' :
//                     mediaType === 'video' ? 'video/*' : 'audio/*'
//                   }
//                   disabled={processing}
//                   className="file-input"
//                 />
//                 <label htmlFor="file-input" className="file-label">
//                   <span className="file-icon">📎</span>
//                   <span className="file-text">
//                     {file ? file.name : 'Click to upload or drag & drop'}
//                   </span>
//                 </label>
//               </div>

//               {file && (
//                 <div className="file-info">
//                   <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
//                   <p>Type: {file.type}</p>
//                 </div>
//               )}

//               <button
//                 type="submit"
//                 disabled={!file || processing}
//                 className="detect-btn"
//               >
//                 {processing ? 'Processing...' : 'Detect Deepfake'}
//               </button>
//             </form>

//             {error && (
//               <div className="error-message">
//                 <span>{error}</span>
//               </div>
//             )}
//           </div>
//         </div>

//         <div className="results-panel">
//           {processing && (
//             <ProcessingWindow 
//               steps={processingSteps}
//               mediaType={mediaType}
//             />
//           )}

//           {result && !processing && (
//             <ResultsWindow 
//               result={result}
//               mediaType={mediaType}
//               file={file}
//             />
//           )}

//           {!processing && !result && !processingSteps.length && (
//             <div className="empty-state">
//               <div className="empty-icon">🎯</div>
//               <p>Upload a file to begin detection</p>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default DeepfakeDetector;

import React, { useState, useEffect, useRef } from 'react';
import './DeepfakeDetector.css';
import ProcessingWindow from './ProcessingWindow';
import ResultsWindow from './ResultsWindow';
import axios from 'axios';
import io from 'socket.io-client';

const DeepfakeDetector = () => {
  const [file, setFile] = useState(null);
  const [mediaType, setMediaType] = useState('image');
  const [processing, setProcessing] = useState(false);
  const [processingSteps, setProcessingSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  const socketRef = useRef(null);
  const stepCounterRef = useRef(0);

  const BACKEND_URL = 'http://localhost:5000';
  const API_ENDPOINTS = {
    image: `${BACKEND_URL}/api/detection/image`,
    video: `${BACKEND_URL}/api/detection/video`,
    audio: `${BACKEND_URL}/api/detection/audio`
  };

  // Initialize WebSocket connection
  useEffect(() => {
    socketRef.current = io(BACKEND_URL, {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      console.log('WebSocket connected:', socketRef.current.id);
      setSessionId(socketRef.current.id);
    });

    socketRef.current.on('connection_response', (data) => {
      console.log('Connection response:', data);
    });

    socketRef.current.on('processing_step', (step) => {
      console.log('Processing step received:', step);
      addProcessingStep(step.name, step.details);
    });

    socketRef.current.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

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
      // Emit start processing event
      if (socketRef.current) {
        socketRef.current.emit('start_processing', { filename: file.name });
      }

      // Prepare form data
      const formData = new FormData();
      formData.append('file', file);
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      // Add initial step
      addProcessingStep('File Upload', `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

      // Simulate some initial processing steps
      await new Promise(r => setTimeout(r, 300));
      addProcessingStep('Preprocessing', `Preparing ${mediaType.toUpperCase()} data...`);

      // Make API call
      console.log('Calling API:', API_ENDPOINTS[mediaType]);
      const response = await axios.post(API_ENDPOINTS[mediaType], formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 120000  // 2 minutes timeout
      });

      console.log('API Response:', response.data);

      // Process response
      setResult({
        isFake: response.data.is_fake,                     // or prediction === "FAKE"
        confidence: response.data.confidence,  
        label: response.data.prediction,            
        fakeProbability: response.data.fake_probability,
        realProbability: response.data.real_probability,
        processingTime: response.data.processing_time,     // match backend key
        framesAnalyzed: response.data.file_info?.frames_analyzed,
        featuresExtracted: response.data.feature_breakdown || response.data.temporal_analysis,
        filename: file.name,
      });

      addProcessingStep('Detection Complete', `Result: ${response.data.prediction}`);
      console.log('Detection complete. Result set.', 'Result:', result);

    } catch (err) {
      console.error('Error:', err);
      const errorMessage = err.response?.data?.error || err.message || 'An error occurred';
      setError(errorMessage);
      addProcessingStep('Error', errorMessage);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="deepfake-detector-container">
      <header className="detector-header">
        <h1>🛡️ DeepFake Shield</h1>
        <p>Advanced AI-Powered Deepfake Detection System</p>
        <div className="connection-status">
          {sessionId ? (
            <span className="status-connected">🟢 Connected</span>
          ) : (
            <span className="status-disconnected">🔴 Disconnected</span>
          )}
        </div>
      </header>

      <div className="detector-content">
        {/* Upload Panel */}
        <div className="upload-panel">
          <div className="upload-card">
            <h2>Upload Media</h2>

            {/* Media Type Selector */}
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

            {/* Upload Form */}
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

        {/* Results Panel */}
        <div className="results-panel">
          {processing && (
            <ProcessingWindow
              steps={processingSteps}
              mediaType={mediaType}
            />
          )}

          {result && !processing && (
            <ResultsWindow
              result={result}
              mediaType={mediaType}
              file={file}
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