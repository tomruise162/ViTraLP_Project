import React, { useState } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8001'; //Đổi port ở đây để test các file main khác nhau

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processResult, setProcessResult] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'video'

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      
      // Determine file type
      if (file.type.startsWith('image/')) {
        setFileType('image');
      } else if (file.type.startsWith('video/')) {
        setFileType('video');
      }
      
      setProcessResult(null);
    }
  };

  // Handle upload and process
  const handleUpload = async () => {
    if (!uploadedFile) {
      alert('Vui lòng chọn file để upload');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      setProcessResult(result);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Lỗi khi upload file');
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle search
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      alert('Vui lòng nhập từ khóa tìm kiếm');
      return;
    }

    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(searchQuery)}`);
      
      if (!response.ok) {
        throw new Error('Search failed');
      }

      const results = await response.json();
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching:', error);
      alert('Lỗi khi tìm kiếm');
    } finally {
      setIsSearching(false);
    }
  };

  // Clean up preview URL when component unmounts or file changes
  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>OCR Text Recognition System</h1>
        <p>Hỗ trợ xử lý ảnh và video</p>
      </header>

      <div className="container">
        {/* Upload Section */}
        <div className="upload-section">
          <h2>Upload và xử lý ảnh/video</h2>
          <div className="upload-area">
            <input
              type="file"
              accept="image/*,video/*"
              onChange={handleFileSelect}
              id="file-input"
              className="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              <i className="upload-icon">📁</i>
              {uploadedFile ? uploadedFile.name : 'Chọn file ảnh hoặc video'}
            </label>
            
            {previewUrl && (
              <div className="preview-container">
                {fileType === 'image' ? (
                  <img src={previewUrl} alt="Preview" className="preview-image" />
                ) : fileType === 'video' ? (
                  <video controls className="preview-video">
                    <source src={previewUrl} type={uploadedFile.type} />
                    Your browser does not support the video tag.
                  </video>
                ) : null}
                <div className="file-info">
                  <span className={`file-type ${fileType}`}>
                    {fileType === 'image' ? '🖼️ Ảnh' : '🎥 Video'}
                  </span>
                  <span className="file-size">
                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>
            )}

            <button 
              onClick={handleUpload} 
              disabled={!uploadedFile || isProcessing}
              className="upload-button"
            >
              {isProcessing ? (
                <>
                  <span className="spinner"></span>
                  Đang xử lý...
                </>
              ) : (
                'Upload và xử lý'
              )}
            </button>
          </div>

          {/* Process Results */}
          {processResult && (
            <div className="process-results">
              <h3>Kết quả xử lý</h3>
              
              {/* OCR Results */}
              {processResult.ocr_results && processResult.ocr_results.length > 0 && (
                <div className="result-section">
                  <h4>🆕 OCR Results (Mới):</h4>
                  <div className="ocr-results-grid">
                    {processResult.ocr_results.map((result, index) => (
                      <div key={index} className="ocr-result">
                        <div className="result-header">
                          <span className="result-text">{result.text}</span>
                          {result.frame !== undefined && (
                            <span className="frame-info">Frame: {result.frame}</span>
                          )}
                        </div>
                        <div className="result-image-container">
                          <img 
                            src={`${API_BASE_URL}/${result.enhanced_file}`} 
                            alt={result.text}
                            className="result-image"
                            onError={(e) => {
                              e.target.src = 'placeholder.png'; // Add a placeholder image
                              e.target.onerror = null;
                            }}
                          />
                        </div>
                        <p className="file-path">{result.enhanced_file}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Existed Results */}
              {processResult.existed_results && processResult.existed_results.length > 0 && (
                <div className="result-section">
                  <h4>✅ Kết quả đã tồn tại:</h4>
                  <div className="existed-results">
                    {processResult.existed_results.map((result, index) => (
                      <div key={index} className="existed-result">
                        <span className="existed-text">{result.text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Invalid Results */}
              {processResult.invalid_results && processResult.invalid_results.length > 0 && (
                <div className="result-section">
                  <h4>❌ Kết quả không hợp lệ:</h4>
                  <div className="invalid-results">
                    {processResult.invalid_results.map((result, index) => (
                      <div key={index} className="invalid-result">
                        <span className="invalid-text">{result.text}</span>
                        <span className="invalid-message">{result.message}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              <div className="result-summary">
                <p>📊 Tổng kết:</p>
                <ul>
                  <li>Kết quả mới: {processResult.ocr_results?.length || 0}</li>
                  <li>Đã tồn tại: {processResult.existed_results?.length || 0}</li>
                  <li>Không hợp lệ: {processResult.invalid_results?.length || 0}</li>
                  <li>File được crop: {processResult.cropped_files?.length || 0}</li>
                  <li>File được enhance: {processResult.enhanced_files?.length || 0}</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Search Section */}
        <div className="search-section">
          <h2>Tìm kiếm</h2>
          <div className="search-bar">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Nhập text cần tìm..."
              className="search-input"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button 
              onClick={handleSearch}
              disabled={isSearching}
              className="search-button"
            >
              {isSearching ? (
                <>
                  <span className="spinner small"></span>
                  Đang tìm...
                </>
              ) : (
                <>
                  🔍 Tìm kiếm
                </>
              )}
            </button>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="search-results">
              <h3>Kết quả tìm kiếm ({searchResults.length})</h3>
              <div className="results-grid">
                {searchResults.map((result, index) => (
                  <div key={index} className="search-result-item">
                    <p className="recognized-text">{result.recognized_text}</p>
                    <div className="search-image-container">
                      <img 
                        src={`${API_BASE_URL}/${result.enhanced_file_path.replace(/\\/g, '/')}`} 
                        alt={result.recognized_text}
                        className="search-result-image"
                        onError={(e) => {
                          e.target.src = 'placeholder.png';
                          e.target.onerror = null;
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {searchResults.length === 0 && searchQuery && !isSearching && (
            <p className="no-results">Không tìm thấy kết quả nào cho "{searchQuery}"</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;