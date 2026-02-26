import "./Heatmap.css";

const Heatmap = ({ result }) => {

  if (!result?.xai) return null;
  const isFake = result?.isFake;

  return (
    <div className="heatmap-container">
      <h2>🧠 Explainable AI Analysis</h2>

      {result.xai.heatmaps && (
        <div className="heatmap-comparison">
          {result.xai.heatmaps.map((frame, index) => (
            <div key={index} className="heatmap-card">
              <h4>Frame {index + 1}</h4>
              <img
                src={`data:image/jpeg;base64,${frame}`}
                alt={`Frame ${index}`}
              />
            </div>
          ))}
        </div>
      )}
      {/* <div className="heatmap-comparison">
        {result.xai.original && (
          <div className="heatmap-card">
            <h4>Original Image</h4>
            <img
              src={`data:image/png;base64,${result.xai.original}`}
              alt="Original"
            />
          </div>
        )}

        {result.xai.heatmap && (
          <div className="heatmap-card">
            <h4>GradCAM Heatmap</h4>
            <img
              src={`data:image/png;base64,${result.xai.heatmap}`}
              alt="Heatmap"
            />
          </div>
        )}
      </div> */}

      <div className="heatmap-reasoning">
        <h3>AI Reasoning</h3>
        <p>{result.xai.explanation}</p>
        <p><strong>Confidence Level:</strong> {result.xai.confidence_level}</p>

        <h4>Key Indicators</h4>
        <ul>
          {Object.entries(result.xai.key_indicators || {}).map(([key, value]) => (
            <li key={key}>
              <strong>{key}:</strong> {value}
            </li>
          ))}
        </ul>

        <h4>Recommendations</h4>
        <ul>
          {result.xai.recommendations?.map((rec, index) => (
            <li key={index}>{rec}</li>
          ))}
        </ul>
      </div>

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
  );
};

export default Heatmap;