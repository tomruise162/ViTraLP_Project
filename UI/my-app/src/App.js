import React, { useState } from "react";
import axios from "axios";
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [croppedImages, setCroppedImages] = useState([]);
  const [enhancedImages, setEnhancedImages] = useState([]);
  const [ocrResults, setOcrResults] = useState([]);

  // Xử lý upload file
  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadedImage(URL.createObjectURL(e.target.files[0]));
    setCroppedImages([]);
    setEnhancedImages([]);
    setOcrResults([]);
  };

  // Gửi file lên API
  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await axios.post("http://localhost:8001/process", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    setCroppedImages(res.data.cropped_files);
    setEnhancedImages(res.data.enhanced_files);
    setOcrResults(res.data.ocr_results);
  };

  // Download helper
  const handleDownload = (url, filename) => {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
  };

  return (
    <div style={{ display: "flex", gap: 30, padding: 30 }}>
      {/* PHẦN 1: Upload ảnh */}
      <div style={{ flex: 1, border: "1px solid #ccc", padding: 10 }}>
        <h3>1. Upload ảnh gốc</h3>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!selectedFile}>Upload</button>
        {uploadedImage && (
          <div>
            <img src={uploadedImage} alt="Uploaded" style={{ width: "100%", marginTop: 10 }} />
            <button
              onClick={() => handleDownload(uploadedImage, selectedFile.name)}
              style={{ marginTop: 10 }}
            >
              Download ảnh gốc
            </button>
          </div>
        )}
      </div>

      {/* PHẦN 2: Ảnh crop biển số */}
      <div style={{ flex: 1, border: "1px solid #ccc", padding: 10 }}>
        <h3>2. Ảnh biển số đã crop (YOLO)</h3>
        {croppedImages.length === 0 && <p>Chưa có dữ liệu</p>}
        {croppedImages.map((url, idx) => (
          <div key={idx} style={{ marginBottom: 10 }}>
            <img
              src={`http://localhost:8001/crops/${url.split(/[\\/]/).pop()}`}
              alt={`Crop ${idx}`}
              style={{ width: "100%" }}
            />
            <button
              onClick={() =>
                handleDownload(
                  `http://localhost:8001/crops/${url.split(/[\\/]/).pop()}`,
                  url.split(/[\\/]/).pop()
                )
              }
              style={{ marginTop: 5 }}
            >
              Download crop
            </button>
          </div>
        ))}
      </div>

      {/* PHẦN 3: Ảnh enhanced + text */}
      <div style={{ flex: 1, border: "1px solid #ccc", padding: 10 }}>
        <h3>3. Ảnh enhanced & Text</h3>
        {enhancedImages.length === 0 && <p>Chưa có dữ liệu</p>}
        {ocrResults.map((item, idx) => (
          <div key={idx} style={{ marginBottom: 10 }}>
            <img
              src={`http://localhost:8001/${item.enhanced_file}`}
              alt={`Enhanced ${idx}`}
              style={{ width: "100%" }}
            />
            <div style={{ margin: "5px 0", fontWeight: "bold" }}>
              Text: {item.text}
            </div>
            <button
              onClick={() =>
                handleDownload(
                  `http://localhost:8001/${item.enhanced_file}`,
                  item.enhanced_file.split(/[\\/]/).pop()
                )
              }
            >
              Download enhanced
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
