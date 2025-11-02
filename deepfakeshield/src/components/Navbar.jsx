import React from "react";
import { Link, NavLink } from "react-router-dom";
import "../styles/Navbar.css";

const Navbar = () => (
  <nav className="navbar glassmorphism-nav">
    <div className="navbar-logo">
      <Link to="/">DeepFakeShield</Link>
    </div>
    <div className="navbar-links">
      <NavLink to="/" end>Home</NavLink>
      <NavLink to="/upload">Upload</NavLink>
      <NavLink to="/tutorial">Tutorial</NavLink>
      <NavLink to="/about">About</NavLink>
      <NavLink to="/extensions">Extensions</NavLink>
    </div>
  </nav>
);

export default Navbar;
