import React, { useState } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8001';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processResult, setProcessResult] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
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

  return (
    <div className="App">
      <header className="App-header">
        <h1>OCR Text Recognition System</h1>
      </header>

      <div className="container">
        {/* Upload Section */}
        <div className="upload-section">
          <h2>Upload và xử lý ảnh</h2>
          <div className="upload-area">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              id="file-input"
              className="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {uploadedFile ? uploadedFile.name : 'Chọn file ảnh'}
            </label>
            
            {previewUrl && (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
              </div>
            )}

            <button 
              onClick={handleUpload} 
              disabled={!uploadedFile || isProcessing}
              className="upload-button"
            >
              {isProcessing ? 'Đang xử lý...' : 'Upload và xử lý'}
            </button>
          </div>

          {/* Process Results */}
          {processResult && (
            <div className="process-results">
              <h3>Kết quả xử lý</h3>
              
              {processResult.ocr_results.length > 0 && (
                <div className="result-section">
                  <h4>OCR Results (Mới):</h4>
                  {processResult.ocr_results.map((result, index) => (
                    <div key={index} className="ocr-result">
                      <p><strong>Text:</strong> {result.text}</p>
                      <p><strong>File:</strong> {result.enhanced_file}</p>
                      <img 
                        src={`${API_BASE_URL}/${result.enhanced_file}`} 
                        alt={result.text}
                        className="result-image"
                      />
                    </div>
                  ))}
                </div>
              )}

              {processResult.existed_results.length > 0 && (
                <div className="result-section">
                  <h4>Kết quả đã tồn tại:</h4>
                  {processResult.existed_results.map((result, index) => (
                    <div key={index} className="existed-result">
                      <p><strong>Text:</strong> {result.text}</p>
                    </div>
                  ))}
                </div>
              )}
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
              {isSearching ? 'Đang tìm...' : 'Tìm kiếm'}
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
                    <img 
                      src={`${API_BASE_URL}/${result.enhanced_file_path.replace(/\\/g, '/')}`} 
                      alt={result.recognized_text}
                      className="search-result-image"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {searchResults.length === 0 && searchQuery && !isSearching && (
            <p className="no-results">Không tìm thấy kết quả nào</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;