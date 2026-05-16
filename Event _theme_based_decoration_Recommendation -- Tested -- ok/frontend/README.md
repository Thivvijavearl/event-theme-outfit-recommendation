# StyleMate - AI Fashion Assistant Frontend

A React frontend for StyleMate, an AI-powered personal fashion assistant. This application allows users to upload images to get color predictions and style recommendations.

## Features

- Image upload interface
- Real-time color prediction using AI model
- Display of prediction results including confidence and control methods
- Clean, responsive UI built with React and Vite

## Backend

This frontend connects to a Flask API running on `http://localhost:5002/predict` by default. Make sure the backend is running before using the application.

For deployment, you can set the backend host via Vite environment variable:

- `VITE_API_BASE=https://your-backend-host`

Then rebuild the frontend so it points to the deployed backend.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:5174](http://localhost:5174) in your browser.

## Usage

1. Click "Choose File" to select a JPG/JPEG image.
2. Click "Predict Color" to upload and analyze the image.
3. View the prediction results including the predicted color, confidence level, description, and control methods.
