import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Home = () => (
  <main className="home-section">
    <section className="hero glass-card">
      <h1>DeepFakeShield: Secure AI Deepfake Detection</h1>
      <p>
        AI-powered, multi-modal detection of deepfakes in video, audio, and images. Explainable results, XAI heatmaps, and transparency on every analysis.
      </p>
      <Link to="/upload" className="main-btn">Upload & Analyze Your File</Link>
    </section>
    <section className="features-section section-gradient glass-card">
      <h2>How It Works</h2>
      <ul>
        <li><strong>Multi-Modal Analysis:</strong> Detects video, audio, and image manipulations</li>
        <li><strong>XAI & Heatmap Visualization:</strong> Visual explanation of detected artifacts</li>
        <li><strong>Powerful AI Models:</strong> State-of-the-art, continuously updated</li>
        <li><strong>Security & Privacy:</strong> Files processed securely and deleted after analysis</li>
      </ul>
    </section>
    <section className="glass-card">
      <h2>Why DeepFakeShield?</h2>
      <p>Our system combines the power of CNNs, audio forensics, temporal analysis, and cross-modal fusion to provide the most reliable and interpretable deepfake detection for journalists, law enforcement, and the public.</p>
      <Link to="/about" className="main-link">Learn more about our technology</Link>
    </section>
    <section className="glass-card">
      <h2>Want to See How?</h2>
      <p>Try our step-by-step tutorial, or jump straight into exploring our upload and analysis interface.</p>
      <Link to="/tutorial" className="main-link">See Tutorial</Link>
    </section>
  </main>
);

export default Home;
