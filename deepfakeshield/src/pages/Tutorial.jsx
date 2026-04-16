import React, { useState } from "react";
import "../styles/Tutorial.css";

import uploadIcon from "../assets/upload.png";
import analysisIcon from "../assets/analysis.png";
import explainIcon from "../assets/explain.png";
import extensionIcon from "../assets/extension.png";

const Tutorial = () => {
  const [activeVideo, setActiveVideo] = useState("image");

  const steps = [
    {
      id: 1,
      title: "Upload Your Media",
      desc: "Go to the Upload section and select an image, video, or audio file. DeepFakeShield supports most common formats like .jpg, .mp4, and .wav. You can also capture live webcam input for instant checks.",
      icon: uploadIcon,
    },
    {
      id: 2,
      title: "AI-Powered Deepfake Analysis",
      desc: "Once uploaded, our hybrid AI model begins analyzing your media using visual, audio, and temporal signals. It applies CNNs, Vision Transformers, and Wav2Vec-based models to detect manipulation traces.",
      icon: analysisIcon,
    },
    {
      id: 3,
      title: "View Results & Insights",
      desc: "After detection, you'll receive an authenticity confidence score along with a breakdown of detected anomalies. The Explainable AI engine generates visual heatmaps for transparency.",
      icon: explainIcon,
    },
    {
      id: 4,
      title: "Browser Extension Integration",
      desc: "Install the DeepFakeShield Chrome or Edge extension to analyze social media videos or conference streams in real-time. The extension provides on-screen authenticity alerts and highlights suspicious areas.",
      icon: extensionIcon,
    },
  ];

  const tutorialVideos = [
    {
      id: "image",
      label: "Image Upload",
      emoji: "🖼️",
      src: "/tutorials/upload-image.mp4",
      description: "How to upload an image file (.jpg, .png, .webp) for deepfake analysis.",
    },
    {
      id: "video",
      label: "Video Upload",
      emoji: "🎬",
      src: "/tutorials/upload-video.mp4",
      description: "How to upload a video file (.mp4, .avi, .mov) and run temporal deepfake detection.",
    },
    {
      id: "audio",
      label: "Audio Upload",
      emoji: "🎙️",
      src: "/tutorials/upload-audio.mp4",
      description: "How to upload an audio file (.wav, .mp3) to detect voice cloning or manipulation.",
    },
  ];

  const active = tutorialVideos.find((v) => v.id === activeVideo);

  return (
    <main className="tutorial-page">
      <section className="tutorial-header">
        <h1>How DeepFakeShield Works</h1>
        <p>
          Follow these simple steps to understand how DeepFakeShield detects and explains
          digital manipulations in real time using AI and cross-modal analysis.
        </p>
      </section>

      {/* Steps */}
      <section className="steps-section">
        <div className="vertical-line"></div>
        {steps.map((step) => (
          <div className="step-card" key={step.id}>
            <div className="step-icon">
              <img src={step.icon} alt={step.title} />
            </div>
            <div className="step-content">
              <h3>Step {step.id}: {step.title}</h3>
              <p>{step.desc}</p>
            </div>
          </div>
        ))}
      </section>

      {/* Video Tutorials — below all steps */}
      <section className="video-tutorial-section">
        <div className="video-tutorial-block">
          <div className="video-tutorial-header">
            <span className="video-tutorial-icon">📹</span>
            <div>
              <h2 className="video-tutorial-title">Upload Tutorials</h2>
              <p className="video-tutorial-subtitle">
                Select a media type to watch the step-by-step upload walkthrough.
              </p>
            </div>
          </div>

          <div className="video-tabs">
            {tutorialVideos.map((v) => (
              <button
                key={v.id}
                className={`video-tab-btn ${activeVideo === v.id ? "active" : ""}`}
                onClick={() => setActiveVideo(v.id)}
              >
                <span className="tab-emoji">{v.emoji}</span>
                <span>{v.label}</span>
              </button>
            ))}
          </div>

          <div className="video-player-wrapper">
            <video
              key={active.src}
              controls
              className="tutorial-video"
              preload="metadata"
            >
              <source src={active.src} type="video/mp4" />
              Your browser does not support HTML5 video.
            </video>
            <p className="video-description">{active.description}</p>
          </div>
        </div>
      </section>
    </main>
  );
};

export default Tutorial;