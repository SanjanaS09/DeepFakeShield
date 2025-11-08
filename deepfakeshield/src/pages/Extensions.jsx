import React from "react";
import "../styles/Extensions.css";

const Extensions = () => (
  <main className="extensions-page">
    <section className="extensions-header">
      <h1>Browser Extensions</h1>
      <p>
        Experience real-time deepfake detection directly in your browser.
        Our upcoming extensions bring DeepFakeShield’s AI detection engine
        to Chrome, Edge, and Firefox — identifying manipulated media as you browse.
      </p>

      <button className="work-btn" disabled>🚧 Work in Progress 🚧</button>
    </section>

    <section className="extensions-features">
      <h2>Planned Features</h2>
      <div className="feature-grid">
        <div className="feature-card">
          <h3>🔍 Real-Time Detection</h3>
          <p>
            Instantly flags manipulated videos and images while scrolling
            through platforms like YouTube, Twitter, or Instagram.
          </p>
        </div>

        <div className="feature-card">
          <h3>⚡ One-Click Verification</h3>
          <p>
            Right-click or tap an icon to analyze suspicious content
            without leaving your browser tab.
          </p>
        </div>

        <div className="feature-card">
          <h3>🧠 Explainable Insights</h3>
          <p>
            Visual heatmaps and confidence scores explain *why* something
            was detected as fake — transparent, interpretable results.
          </p>
        </div>

        {/* <div className="feature-card">
          <h3>🔒 Privacy-First Design</h3>
          <p>
            All analysis happens locally on your device — no uploads, no data storage, no tracking.
          </p>
        </div> */}
      </div>
    </section>
  </main>
);

export default Extensions;
