import React from "react";


const Tutorial = () => (
  <main className="tutorial-page">
    <section className="glass-card">
      <h1>How to Use DeepFakeShield</h1>
      <div className="tutorial-steps">
        <div className="step">
          <h3>Step 1: Upload Your Media</h3>
          <p>Navigate to the Upload page and drag & drop or select your file (video, audio, or image).</p>
        </div>
        <div className="step">
          <h3>Step 2: Analysis Begins</h3>
          <p>Our AI models analyze the file for signs of manipulation using state-of-the-art algorithms.</p>
        </div>
        <div className="step">
          <h3>Step 3: View Results</h3>
          <p>Get a detailed report with confidence scores, heatmaps, and XAI explanations.</p>
        </div>
        <div className="step">
          <h3>Step 4: Understand the Analysis</h3>
          <p>Review the visual heatmaps and detailed breakdown to understand why content was flagged.</p>
        </div>
      </div>
    </section>
  </main>
);

export default Tutorial;
