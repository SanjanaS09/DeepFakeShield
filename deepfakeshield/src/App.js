import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import About from "./pages/About";
import Tutorial from "./pages/Tutorial";
import Extensions from "./pages/Extensions";
import NotFound from "./pages/NotFound";
import "./styles/global.css";
import "./styles/gradients.css";
import "./styles/glassmorphism.css";

function App() {
  return (
    <Router>
      <div className="main-gradient-bg">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/about" element={<About />} />
          <Route path="/tutorial" element={<Tutorial />} />
          <Route path="/extensions" element={<Extensions />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
