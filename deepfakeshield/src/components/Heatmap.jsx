import "./Heatmap.css";

const Heatmap = ({ result }) => {

  if (!result || !result.xai) return null;

  const isFake = result.prediction === "FAKE";

  return (
    <div className="heatmap-container">
      <h2>🧠 Explainable AI Analysis</h2>

      {/* IMAGE HEATMAP */}
      {result.xai.original && result.xai.heatmap && (
        <div className="heatmap-comparison">
          <div className="heatmap-card">
            <h4>Original Image</h4>
            <img
              src={`data:image/jpeg;base64,${result.xai.original}`}
              alt="Original"
            />
          </div>

          <div className="heatmap-card">
            <h4>GradCAM Heatmap</h4>
            <img
              src={`data:image/jpeg;base64,${result.xai.heatmap}`}
              alt="Heatmap"
            />
          </div>
        </div>
      )}

      {/* VIDEO HEATMAP FRAMES */}
      {Array.isArray(result.xai.heatmap_frames) &&
        result.xai.heatmap_frames.length > 0 && (
          <div className="heatmap-grid">
            {result.xai.heatmap_frames.map((frame, index) => (
              <div key={index} className="heatmap-card">
                <h4>Frame {index + 1}</h4>
                <img
                  src={`data:image/jpeg;base64,${frame}`}
                  alt={`Heatmap ${index}`}
                />
              </div>
            ))}
          </div>
        )}

      {/* AI Reasoning */}
      <div className="heatmap-reasoning">
        <h3>AI Reasoning</h3>
        <p>{result.xai.explanation}</p>
        <p>
          <strong>Confidence Level:</strong>{" "}
          {result.xai.confidence_level}
        </p>

        <h4>Key Indicators</h4>
        <ul>
          {Object.entries(result.xai.key_indicators || {}).map(
            ([key, value]) => (
              <li key={key}>
                <strong>{key}:</strong> {value}
              </li>
            )
          )}
        </ul>

        <h4>Recommendations</h4>
        <ul>
          {result.xai.recommendations?.map((rec, index) => (
            <li key={index}>{rec}</li>
          ))}
        </ul>
      </div>

      {/* STATUS */}
      {isFake ? (
        <div className="recommendation-warning">
          <h4>⚠️ Warning</h4>
          <p>
            This content has been detected as a potential deepfake.
            Please verify the source before sharing.
          </p>
        </div>
      ) : (
        <div className="recommendation-success">
          <h4>✓ Status</h4>
          <p>
            This content appears to be authentic based on our analysis.
          </p>
        </div>
      )}
    </div>
  );
};

export default Heatmap;