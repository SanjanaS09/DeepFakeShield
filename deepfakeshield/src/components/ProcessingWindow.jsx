// import React, { useEffect, useState } from 'react';
// import './ProcessingWindow.css';

// const ProcessingWindow = ({ steps, mediaType }) => {
//   const [scrolled, setScrolled] = useState(false);

//   useEffect(() => {
//     const container = document.querySelector('.processing-steps-list');
//     if (container) {
//       container.scrollTop = container.scrollHeight;
//     }
//   }, [steps]);

//   const getStageIcon = (step) => {
//     if (step.includes('Upload')) return '📤';
//     if (step.includes('Preprocessing')) return '🔧';
//     if (step.includes('Analysis')) return '📊';
//     if (step.includes('Extraction')) return '✨';
//     if (step.includes('Inference')) return '🧠';
//     if (step.includes('Post')) return '📈';
//     if (step.includes('Complete')) return '✅';
//     if (step.includes('Error')) return '❌';
//     return '🔄';
//   };

//   return (
//     <div className="processing-window">
//       <div className="processing-header">
//         <h3>⚙️ Processing Pipeline</h3>
//         <span className="processing-badge">Live</span>
//       </div>

//       <div className="processing-visualization">
//         <div className="pipeline-stages">
//           {['Upload', 'Preprocessing', 'Features', 'Inference', 'Complete'].map((stage, idx) => (
//             <div
//               key={idx}
//               className={`pipeline-stage ${steps.some(s => s.name.includes(stage)) ? 'completed' : ''
//                 }`}
//             >
//               <div className="stage-number">{idx + 1}</div>
//               <div className="stage-label">{stage}</div>
//             </div>
//           ))}
//         </div>
//       </div>

//       <div className="processing-details">
//         <h4>📋 Processing Steps:</h4>
//         <div className="processing-steps-list">
//           {steps.map((step, idx) => (
//             <div key={`step-${idx}-${step.name}`}  className="processing-step">
//                 <div className="step-header">
//                   <span className="step-icon">{getStageIcon(step.name)}</span>
//                   <span className="step-name">{step.name}</span>
//                   <span className="step-time">{step.timestamp}</span>
//                 </div>
//                 {step.details && (
//                   <div className="step-details">
//                     📝 {step.details}
//                   </div>
//                 )}
//                 <div className="step-status">
//                   ✓ {step.status}
//                 </div>
//               </div>
//           ))}
//             </div>
//       </div>

//         <div className="processing-progress">
//           <div className="progress-bar">
//             <div
//               className="progress-fill"
//               style={{
//                 width: `${Math.min((steps.length / 7) * 100, 100)}%`
//               }}
//             ></div>
//           </div>
//           <p className="progress-text">{Math.min(steps.length, 7)} / 7 steps completed</p>
//         </div>
//       </div>
//       );
// };

//       export default ProcessingWindow;

import React, { useEffect, useState } from 'react';
import './ProcessingWindow.css';

const ProcessingWindow = ({ steps, mediaType }) => {
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const container = document.querySelector('.processing-steps-list');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [steps]);
  
  const getStageIcon = (step) => {
    if (step.includes('Upload')) return '📤';
    if (step.includes('Preprocessing')) return '⚙️';
    if (step.includes('Analysis') || step.includes('Detection')) return '🔍';
    if (step.includes('Extraction')) return '🧬';
    if (step.includes('Inference') || step.includes('Model')) return '🤖';
    if (step.includes('Post')) return '📊';
    if (step.includes('Complete')) return '✅';
    if (step.includes('Error')) return '❌';
    return '🔄';
  };
  
  const pipelineStages = [
    'Upload',
    'Preprocessing',
    'Features',
    'Inference',
    'Complete'
  ];
  
  const getCompletedStages = () => {
    return pipelineStages.filter(stage =>
      steps.some(s => s.name.includes(stage))
    ).length;
  };
  
  return (
    <div className="processing-window">
      {/* Header */}
      <div className="processing-header">
        <h3>🔄 Processing Pipeline</h3>
        <span className="processing-badge">Live</span>
      </div>
      
      {/* Pipeline Visualization */}
      <div className="processing-visualization">
        <div className="pipeline-stages">
          {pipelineStages.map((stage, idx) => {
            const isCompleted = steps.some(s => s.name.includes(stage));
            return (
              <div
                key={idx}
                className={`pipeline-stage ${isCompleted ? 'completed' : ''}`}
              >
                <div className="stage-number">{idx + 1}</div>
                <div className="stage-label">{stage}</div>
              </div>
            );
          })}
        </div>
      </div>
      
      {/* Processing Steps */}
      <div className="processing-details">
        <h4>Processing Steps</h4>
        <div className="processing-steps-list">
          {steps.map((step) => (
            <div key={`step-${step.id}-${step.name}`} className="processing-step">
              <div className="step-header">
                <span className="step-icon">{getStageIcon(step.name)}</span>
                <span className="step-name">{step.name}</span>
                <span className="step-time">{step.timestamp}</span>
              </div>
              {step.details && (
                <div className="step-details">
                  {step.details}
                </div>
              )}
              <div className="step-status">{step.status}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="processing-progress">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${Math.min((getCompletedStages() / 5) * 100, 100)}%` }}
          />
        </div>
        <p className="progress-text">
          {getCompletedStages()}/{pipelineStages.length} stages completed
        </p>
      </div>
    </div>
  );
};

export default ProcessingWindow;