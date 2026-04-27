import React, { useState } from "react";
import FileUpload from "../components/FileUpload"; // drag/drop or select
import AnalysisResult from "../components/AnalysisResults";
import ResultWindow from "../components/ResultsWindow"
import Loader from "../components/Loader";
import "../styles/Upload.css";
import { API_BASE_URL } from "../config";

const Upload = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendFileToBackend = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      alert("Upload failed: " + e.message);
    }
    setLoading(false);
  };

  return (
    <div className="upload-wrapper">
      <div className="glass-card upload-pane">
        <h2>Analyze for Deepfakes</h2>
        <FileUpload onFileUpload={sendFileToBackend} />
        {loading && <Loader />}
        {result && <AnalysisResult data={result} />}
      </div>
    </div>
  );
};

export default Upload;
