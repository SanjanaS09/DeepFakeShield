import React from "react";
import { HeatMapGrid } from "react-heatmap-grid";

const HeatmapOverlay = ({ heatmapData }) => {
  if (!heatmapData) return null;
  
  const cellStyle = {
    background: 'linear-gradient(135deg, rgba(84, 172, 191, 0.3), rgba(167, 235, 242, 0.5))',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '4px'
  };
  
  return (
    <div className="heatmap-overlay">
      <h4>Detection Heatmap</h4>
      <HeatMapGrid
        data={heatmapData.data}
        xLabels={heatmapData.xLabels}
        yLabels={heatmapData.yLabels}
        cellStyle={(background, value, min, max) => cellStyle}
      />
    </div>
  );
};

export default HeatmapOverlay;
