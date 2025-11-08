import React from "react";
import "../styles/About.css";

// ✅ Import all images from assets
import aiImg from "../assets/ai.png";
import audioImg from "../assets/audio.png";
import fusionImg from "../assets/fusion.png";
import xaiImg from "../assets/xai.png";
import browserImg from "../assets/brower.png";

const About = () => {
  // ✅ Define technologies JSON
  const technologies = [
    {
      title: "AI & Machine Learning",
      desc: "Core deepfake detection powered by CNNs, Vision Transformers, and XceptionNet for pixel-level anomaly detection.",
      img: aiImg,
    },
    {
      title: "Audio Intelligence",
      desc: "Voice spoofing detection using ECAPA-TDNN and Wav2Vec 2.0 — uncovering subtle manipulations in cloned speech.",
      img: audioImg,
    },
    {
      title: "Cross-Modal Fusion",
      desc: "Transformer-based fusion unites visual, audio, and textual embeddings for accurate, real-time multimodal analysis.",
      img: fusionImg,
    },
    {
      title: "Explainable AI (XAI)",
      desc: "Integrated XAI heatmaps and visual explanations enhance trust by revealing how decisions are made.",
      img: xaiImg,
    },
    {
      title: "Browser & Platform Integration",
      desc: "Lightweight browser extensions detect manipulations directly within YouTube, WhatsApp, and Zoom environments.",
      img: browserImg,
    },
  ];

  return (
    <main className="about-page">
      <section className="about-header">
        <h1>About DeepFakeShield</h1>
        <p>
          DeepFakeShield is a real-time, cross-platform deepfake detection framework that fuses
          multi-modal AI with explainability. Our mission is to restore digital trust by verifying
          authenticity across video, audio, and image content — instantly, transparently, and ethically.
        </p>
      </section>

      <section className="articles-section">
        <h2>Technology Stack</h2>
        <p className="section-subtext">
          Our system integrates advanced neural architectures, fusion pipelines, and real-time APIs.
        </p>

        <div className="articles-grid">
          {technologies.map((tech, i) => (
            <div key={i} className="article-card">
              <div className="card-image">
                <img src={tech.img} alt={tech.title} />
              </div>
              <div className="card-content">
                <h3>{tech.title}</h3>
                <p>{tech.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="team-section">
        <h2>Our Team</h2>
        <p className="section-subtext">
          DeepFakeShield is built by passionate engineers from Usha Mittal Institute of Technology, Mumbai.
        </p>
        <div className="team-cards">
          <div className="team-member">
            <h3>Tejashree Deore</h3>
            <p>Roll No. 13</p>
          </div>
          <div className="team-member">
            <h3>Ketaki Sakhadeo</h3>
            <p>Roll No. 54</p>
          </div>
          <div className="team-member">
            <h3>Sanjana Shetty</h3>
            <p>Roll No. 60</p>
          </div>
        </div>
      </section>
    </main>
  );
};

export default About;
