import { useState, useEffect } from 'react'
import './App.css'

const apiBase = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5002'

function App() {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [imagePreviews, setImagePreviews] = useState([])
  const [selectedImageIndex, setSelectedImageIndex] = useState(null)
  const [occasion, setOccasion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [darkMode, setDarkMode] = useState(false)
  const [showOccasionSelector, setShowOccasionSelector] = useState(false)
  const [occasions, setOccasions] = useState([
    'Birthday',
    'Casual Outing',
    'Wedding',
    'Official Events'
  ])

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme === 'dark') {
      setDarkMode(true)
      document.body.classList.add('dark-mode')
    }

    fetch(`${apiBase}/occasions`)
      .then((response) => response.json())
      .then((data) => {
        if (Array.isArray(data.occasions) && data.occasions.length > 0) {
          setOccasions(data.occasions)
          if (!occasion) {
            setOccasion(data.occasions[0])
          }
        }
      })
      .catch((error) => {
        console.error('Failed to fetch occasions:', error)
        // Keep fallback occasion list if backend is unavailable
      })
  }, [])

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    if (!darkMode) {
      document.body.classList.add('dark-mode')
      localStorage.setItem('theme', 'dark')
    } else {
      document.body.classList.remove('dark-mode')
      localStorage.setItem('theme', 'light')
    }
  }

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files)
    setSelectedFiles(files)
    
    // Create previews
    const previews = files.map(file => URL.createObjectURL(file))
    setImagePreviews(previews)
    setSelectedImageIndex(null)
    setResult(null)
    setError(null)
    setOccasion('')
    setShowOccasionSelector(false)
  }

  const handleImageSelect = (index) => {
    setSelectedImageIndex(index)
    setShowOccasionSelector(true)
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    
    if (selectedFiles.length === 0) {
      setError('Please select at least one image file.')
      return
    }

    if (selectedFiles.length > 1 && selectedImageIndex === null) {
      setError('Please select an image from the gallery.')
      return
    }

    if (!occasion) {
      setError('Please select an occasion.')
      return
    }

    setLoading(true)
    setError(null)

    const imageToProcess = selectedFiles[selectedImageIndex !== null ? selectedImageIndex : 0]
    const formData = new FormData()
    formData.append('image', imageToProcess)
    formData.append('occasion', occasion)

    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setResult({
        ...data,
        occasion: occasion,
        selectedImage: imagePreviews[selectedImageIndex !== null ? selectedImageIndex : 0]
      })
    } catch (err) {
      setError('Error predicting image. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="brand">StyleMate</h1>
        <button onClick={toggleDarkMode} className="theme-toggle">
          {darkMode ? '☀️' : '🌙'}
        </button>
      </header>
      
      <main className="main-content">
        <div className="hero">
          <h2>AI Fashion Assistant</h2>
          <p>Upload one or more images to get color predictions and personalized style recommendations.</p>
        </div>
        
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-upload">
            <input
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleFileChange}
              id="file-input"
              className="file-input"
              multiple
            />
            <label htmlFor="file-input" className="file-label">
              {selectedFiles.length === 0 
                ? 'Choose one or more images...' 
                : `${selectedFiles.length} image(s) selected`}
            </label>
          </div>

          {/* Image Gallery for Multiple Images */}
          {selectedFiles.length > 1 && (
            <div className="image-gallery">
              <p className="gallery-title">Select an image:</p>
              <div className="gallery-container">
                {imagePreviews.map((preview, index) => (
                  <div 
                    key={index}
                    className={`gallery-item ${selectedImageIndex === index ? 'active' : ''}`}
                    onClick={() => handleImageSelect(index)}
                  >
                    <img src={preview} alt={`Preview ${index + 1}`} />
                    <span className="image-number">{index + 1}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Occasion Selector */}
          {selectedFiles.length > 0 && (selectedImageIndex !== null || selectedFiles.length === 1) && (
            <div className="occasion-selector-wrapper">
              <label htmlFor="occasion-select" className="occasion-label">Select Occasion:</label>
              <select 
                id="occasion-select"
                value={occasion} 
                onChange={(e) => setOccasion(e.target.value)}
                className="occasion-select"
              >
                <option value="">-- Please select an occasion --</option>
                {occasions.map((occ) => (
                  <option key={occ} value={occ}>{occ}</option>
                ))}
              </select>
            </div>
          )}

          <button 
            type="submit" 
            disabled={loading || selectedFiles.length === 0} 
            className="submit-btn"
          >
            {loading ? 'Analyzing...' : 'Get Style Recommendations'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-card">
            <h3>Analysis Results</h3>
            
            {/* Selected Image Display */}
            <div className="result-image-section">
              <img src={result.selectedImage} alt="Selected" className="preview-image" />
            </div>

            {/* Color Information */}
            <div className="color-info-section">
              <div className="color-info-header">
                <h4>Dominant Color Analysis</h4>
                <div className="color-badge">
                  <span className="color-name">{result.predicted_species}</span>
                  <span className="color-confidence">{result.confidence}%</span>
                </div>
              </div>

              <div className="description-section">
                <h5>Color Description:</h5>
                <p>{result.description}</p>
              </div>

              <div className="dress-pattern-section">
                <h5>Dress Pattern & Style for {result.predicted_species}:</h5>
                <p>{result.dress_pattern_description}</p>
              </div>

              {/* Matching colors guidance */}
              {result.matching_colors && result.matching_colors.length > 0 && (
                <div className="matching-section">
                  <h5>Best Outfit Colors</h5>
                  <p>{result.matching_colors.join(', ')}</p>
                </div>
              )}
              {result.suitable_colors && result.suitable_colors.length > 0 && (
                <div className="matching-section">
                  <h5>Suitable Palette for {result.occasion}</h5>
                  <p>{result.suitable_colors.join(', ')}</p>
                </div>
              )}
            </div>

            {/* Occasion and Gender-Specific Recommendations */}
            <div className="recommendations-section">
              <h4>Style Recommendations for {result.occasion} (based on complementary colors)</h4>
              
              <div className="gender-recommendations">
                {/* Men's Recommendations */}
                <div className="gender-box men">
                  <h5>👔 For Men:</h5>
                  <ul>
                    {result.men_recommendations && result.men_recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>

                {/* Women's Recommendations */}
                <div className="gender-box women">
                  <h5>👗 For Women:</h5>
                  <ul>
                    {result.women_recommendations && result.women_recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
