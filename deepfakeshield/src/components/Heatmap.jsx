// import React from "react";
// import { HeatMapGrid } from "react-heatmap-grid";

// const HeatmapOverlay = ({ heatmapData }) => {
//   if (!heatmapData) return null;
  
//   const cellStyle = {
//     background: 'linear-gradient(135deg, rgba(84, 172, 191, 0.3), rgba(167, 235, 242, 0.5))',
//     border: '1px solid rgba(255,255,255,0.1)',
//     borderRadius: '4px'
//   };
  
//   return (
//     <div className="heatmap-overlay">
//       <h4>Detection Heatmap</h4>
//       <HeatMapGrid
//         data={heatmapData.data}
//         xLabels={heatmapData.xLabels}
//         yLabels={heatmapData.yLabels}
//         cellStyle={(background, value, min, max) => cellStyle}
//       />
//     </div>
//   );
// };

// export default HeatmapOverlay;


import React, { useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';

export default function Heatmap({ title, data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data_array = imageData.data;

    // Normalize data to 0-255
    const min = Math.min(...data.flat());
    const max = Math.max(...data.flat());
    const range = max - min || 1;

    let idx = 0;
    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < data[0].length; j++) {
        const normalized = ((data[i][j] - min) / range) * 255;
        
        // Hot colormap
        let r, g, b;
        if (normalized < 85) {
          r = normalized * 3;
          g = 0;
          b = 255 - normalized * 3;
        } else if (normalized < 170) {
          r = 255;
          g = (normalized - 85) * 3;
          b = 0;
        } else {
          r = 255;
          g = 255;
          b = (normalized - 170) * 3;
        }

        data_array[idx] = Math.min(255, r);
        data_array[idx + 1] = Math.min(255, g);
        data_array[idx + 2] = Math.min(255, b);
        data_array[idx + 3] = 255;

        idx += 4;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <Card className="p-4">
      <h3 className="font-semibold mb-2 text-sm">{title}</h3>
      <canvas
        ref={canvasRef}
        width={224}
        height={224}
        className="w-full border rounded"
      />
    </Card>
  );
}
