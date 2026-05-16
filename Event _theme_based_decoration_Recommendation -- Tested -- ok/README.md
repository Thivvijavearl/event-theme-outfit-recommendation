# StyleMate - AI-Powered Fashion Assistant

A full-stack application for color-based fashion recommendations using AI. The project includes a Python Flask backend with TensorFlow model for color prediction and a React frontend for user interaction.

## Project Structure

- `backend/`: Python Flask API with AI color prediction models and API server
- `frontend/`: React application with image upload interface
- `data/`: Raw image dataset, configuration files, and sample inputs
- `models/`: Stored model weights and encoder artifacts
- `training/`: Training scripts, model development code, and prediction helpers
- `docs/legacy/`: Legacy project notes and older documentation
- `run_project.bat`: Script to start both backend and frontend

## Features

- **Backend (Flask)**: AI-powered color classification using VGG16 model with guidance to recommend complementary/neutral outfit colors for event themes rather than matching the background exactly
- **Frontend (React)**: User-friendly interface for image upload and result display showing matching color palettes and sample style suggestions
- **API Integration**: Seamless communication between frontend and backend

## Prerequisites

- Python 3.8+
- Node.js 16+
- pip and npm

## Installation

1. **Backend Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## Running the Project

### Option 1: Run Everything at Once (Recommended)
Double-click `run_project.bat` or run:
```bash
run_project.bat
```

This will start both the Flask backend and React frontend in separate command windows.

### Option 2: Manual Start

1. **Start Backend**:
   ```bash
   python backend\MobileApi_Color.py
   ```
   Backend runs on `http://localhost:5002`

2. **Start Frontend** (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Frontend runs on `http://localhost:5174`

## Usage

1. Open `http://localhost:5174` in your browser
2. Upload a JPG/JPEG image
3. Click "Predict Color" to get AI-powered analysis
4. View results including predicted color, confidence, and recommendations

## API Endpoints

- `POST /predict`: Upload image for color prediction
  - Input: Form data with 'image' field (JPG/JPEG)
  - Output: JSON with prediction results

## Files

- `Predict.py`: Color classification class
- `MobileApi_Color.py`: Flask API server
- `frontend/src/App.jsx`: React upload interface
- `color_dominance_vgg16.h5`: Trained AI model
- `color_label_encoder.pkl`: Label encoder for predictions
