import React from "react";
import { Link } from "react-router-dom";


const NotFound = () => (
  <main className="notfound-page">
    <section className="glass-card">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
      <Link to="/" className="main-btn">Go Home</Link>
    </section>
  </main>
);

export default NotFound;
