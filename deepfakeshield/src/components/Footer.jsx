import React from "react";
import "../styles/Footer.css";
export default function Footer() {
  return (
    <footer className="glass-footer">
      <span>© {new Date().getFullYear()} DeepFakeShield. Built by Tejashree Deore(13), Ketaki Sakhadeo(54), Sanjana Shetty(60).</span>
      <span>
        <a href="https://github.com/SanjanaS09/DeepFakeShield" rel="noopener noreferrer" target="_blank">GitHub</a>
      </span>
    </footer>
  );
}
