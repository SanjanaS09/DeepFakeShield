import React, { useRef } from "react";
import "../styles/FileUpload.css";

const FileUpload = ({ onFileUpload }) => {
  const inputRef = useRef();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    onFileUpload(e.target.files[0]);
  };

  return (
    <div
      className="file-drop glass-card"
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        type="file"
        style={{ display: "none" }}
        ref={inputRef}
        onChange={handleChange}
      />
      <span>Drag & drop or click to upload file (video/audio/image)</span>
    </div>
  );
};

export default FileUpload;
