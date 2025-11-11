import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";
import dd from "../assets/dd.png"

const Home = () => (
  <div className="home-section">
    <section className="hero">
      <h1>DeepFakeShield</h1>
      <p>
        AI-powered multi-modal detection of deepfakes in video, audio, and images.
        Transparent XAI visualizations that empower digital truth.
      </p>
      <Link to="/upload" className="main-btn">
        Upload & Analyze Your File
      </Link>
    </section>

    <section className="info-section">
      <div className="container">
        <div className="info-header">
          <h2>What’s in it for you?</h2>
          <p>
            DeepFakeShield helps users, journalists, investigators, and organizations
            verify the authenticity of digital media — instantly, transparently, and across platforms.
            From real-time alerts to explainable AI visuals, it’s your defense against digital deception.
          </p>
        </div>

        <div className="info-cards">
          <div className="info-card">
            <h3>Real-Time Multi-Modal Detection</h3>
            <p>
              Instantly detects deepfakes in images, videos, and audio by analyzing
              pixel, voice, and contextual cues using CNNs, Transformers, and fusion networks.
            </p>
            <span className="card-index">01</span>
          </div>

          <div className="info-card">
            <h3>Browser & Platform Extensions</h3>
            <p>
              Seamlessly integrates into browsers and conferencing platforms like
              YouTube, WhatsApp, and Zoom — providing on-screen authenticity alerts while you browse or chat.
            </p>
            <span className="card-index">02</span>
          </div>

          <div className="info-card">
            <h3>Explainable AI (XAI) Insights</h3>
            <p>
              Understand why a clip is flagged. DeepFakeShield visualizes the evidence —
              highlighting suspicious facial regions, voice inconsistencies, and manipulated textures.
            </p>
            <span className="card-index">03</span>
          </div>

          <div className="info-card">
            <h3>Cross-Platform Trust Validation</h3>
            <p>
              Built to handle data from diverse sources — compressed videos, social media clips,
              or live conference feeds — ensuring accurate detection regardless of platform noise or format.
            </p>
            <span className="card-index">04</span>
          </div>

          <div className="info-card">
            <h3>Privacy & Ethical AI</h3>
            <p>
              No media leaves your device. DeepFakeShield uses secure local inference
              and transparent model reporting — ensuring fairness, privacy, and accountability.
            </p>
            <span className="card-index">05</span>
          </div>

          <div className="info-card">
            <h3>Cross-Media Fusion Intelligence</h3>
            <p>
              DeepFakeShield unites visual, audio, and textual features through advanced transformer-based
              fusion networks — detecting subtle manipulations that single-modality models miss.
            </p>
            <span className="card-index">06</span>
          </div>
        </div>
      </div>
    </section>


    <section className="flow-section">
      <div className="flow-content">
        {/* LEFT SIDE - IMAGE / VISUAL */}
        <div className="flow-left">
           <h2>
            <span className="gradient-text">3 Smart Steps</span>
            <br /> to Detect Deepfakes Instantly
          </h2>

          <p className="flow-subtext">
            DeepFakeShield simplifies AI authenticity checks into three intelligent steps —
            leveraging computer vision, audio forensics, and cross-modal fusion to ensure truth
            in every frame and waveform.
          </p>
          <div className="flow-image-wrapper">
            <img
              src={dd}
              style ={{ width: '40%', height: 'auto' }}
              alt="DeepFake Detection Workflow"
              className="flow-illustration"
            />
          </div>
        </div>

        {/* RIGHT SIDE - TEXT + STEPS */}
        <div className="flow-right">
         

          <div className="flow-steps">
            <div className="flow-step">
              <div className="step-number">01</div>
              <div>
                <h3>Upload or Capture Media</h3>
                <p>
                  Select an image, video, or audio file — or use our live extension to capture
                  media directly from YouTube, WhatsApp, or Zoom.
                </p>
              </div>
            </div>

            <div className="flow-step">
              <div className="step-number">02</div>
              <div>
                <h3>AI-Powered Multi-Modal Analysis</h3>
                <p>
                  DeepFakeShield’s CNN + Transformer pipeline runs real-time detection across
                  visual, audio, and contextual layers to expose subtle digital tampering.
                </p>
              </div>
            </div>

            <div className="flow-step">
              <div className="step-number">03</div>
              <div>
                <h3>Verified & Explainable Insights</h3>
                <p>
                  Instantly view authenticity confidence, visual heatmaps, and transparent
                  explainable AI reasoning — all inside your browser.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section className="why-deepfakeshield-section">
      <div className="why-content">
        <div className="why-left">
          <h2>AI-driven, real-time deepfake defense for everyone</h2>
        </div>

        <div className="why-right">
          <p>
            DeepFakeShield uses cutting-edge AI and multi-modal fusion to detect visual,
            audio, and contextual manipulations with unmatched precision. Designed for
            seamless use across browsers and platforms, it empowers users to identify
            misinformation, secure content authenticity, and maintain digital trust — wherever
            they are.
          </p>
        </div>
      </div>
    </section>


  </div >
);

export default Home;
