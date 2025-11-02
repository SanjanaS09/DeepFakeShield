import React from "react";
import "../styles/Footer.css";
export default function Footer() {
  return (
    <footer className="glass-footer">
      <span>© {new Date().getFullYear()} DeepFakeShield. Built with ❤️ for AI Security</span>
      <span>
        <a href="https://github.com/your-repo" rel="noopener noreferrer" target="_blank">GitHub</a>
      </span>
    </footer>
  );
}
