import React from "react";
import HeatmapOverlay from "./HeatmapOverlay";

// dummy example, expects data from backend
const AnalysisResult = ({ data }) => (
  <div className="glass-card analysis-result">
    <h3>Results</h3>
    <p><strong>Status:</strong> {data.decision ? data.decision : "Unknown"}</p>
    <p><strong>Confidence:</strong> {data.confidence_score || "N/A"}</p>
    <div className="result-section">
      <HeatmapOverlay heatmapData={data.heatmap} />
      <div className="xai-block">
        <h4>XAI Explanation</h4>
        {/* Render explanation, reasons for deepfake decision, etc. */}
        <p>{data.details || "Details provided by backend"}</p>
      </div>
    </div>
  </div>
);

export default AnalysisResult;
