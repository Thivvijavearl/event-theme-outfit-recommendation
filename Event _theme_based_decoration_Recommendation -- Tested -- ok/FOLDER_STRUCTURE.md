# Project Folder Structure

This document describes the current project layout and the purpose of each major folder and file.

## Root Directory

- `.gitignore`
  - Lists files and folders Git should ignore, such as generated models, temporary files, and environment files.
- `README.md`
  - Main project documentation that explains the overall project, how to run it, and key components.
- `requirements.txt`
  - Python dependencies required to run the backend and training scripts.
- `run_project.bat`
  - Windows batch script to start the application or development environment.
- `.git/`
  - Git repository metadata folder. Not part of the application itself.
- `.github/`
  - GitHub workflow or issue automation configuration (if present).

## `backend/`

Contains the Python API and inference code used for serving predictions.

- `__init__.py`
  - Marks this folder as a Python package.
- `MobileApi_Color.py`
  - Backend API logic for color-based model predictions.
- `Predict.py`
  - Prediction helper functions, typically for image or model inference.
- `outfit_recommendation_api.py`
  - Main API entrypoints for outfit recommendation.
- `temp_uploads/`
  - Temporary storage for uploaded files during inference or testing.

## `training/`

Contains scripts for training models, evaluating them, and running batch or single predictions.

- `__init__.py`
  - Marks this folder as a Python package.
- `augmentation.py`
  - Data augmentation utilities used before model training.
- `ColorHistogram_Classifier.py`
  - Script for training and evaluating a color histogram classifier.
- `ColorModel_Train.py`
  - Script for training the base color detection model.
- `ColorModel_Train_FineTuned.py`
  - Script for fine-tuning the color model.
- `ColorModel_Train_Improved.py`
  - Script for training an improved version of the color model.
- `evaluate_color_model.py`
  - Evaluation script for color prediction models.
- `plots.py`
  - Plotting utilities to visualize training results or predictions.
- `run_batch_predictions.py`
  - Batch prediction script for processing many images or data points.
- `run_single_prediction.py`
  - Demo script for a single sample prediction.
- `test_recs.py`
  - Test script for validating the recommendation functionality.
- `test_viva_demo.py`
  - Demo or health-check test script for the project.
- `WeedModel_Train.py`
  - Script for training weed detection or related models.
- `outputs/`
  - Folder for model output data, charts, or generated reports.

## `frontend/`

Contains the web UI application code, likely a React/Vite frontend.

- `.gitignore`
  - Files and folders to ignore inside the frontend package.
- `README.md`
  - Frontend-specific documentation.
- `eslint.config.js`
  - ESLint configuration for the frontend codebase.
- `index.html`
  - Main HTML entrypoint for the frontend app.
- `package.json`
  - Node.js frontend package manifest listing dependencies and scripts.
- `package-lock.json`
  - Locked dependency versions for the frontend.
- `vite.config.js`
  - Vite configuration for building and serving the frontend.
- `dist/`
  - Built frontend output files.
- `node_modules/`
  - Installed frontend dependencies.
- `public/`
  - Static files served by the frontend.
- `src/`
  - Frontend application source code.

## `data/`

Stores datasets and sample data used by training and inference.

- `colors_dataset.csv`
  - Dataset file containing color-related data for training or analysis.
- `color_controls.json`
  - Configuration or control data used during prediction or preprocessing.
- `fashion_event_outfit_dataset.csv`
  - Dataset for outfit recommendation or event-based fashion predictions.
- `raw/`
  - Raw data files used for training or preprocessing.
- `samples/`
  - Sample images or data examples.

## `models/`

Contains saved machine learning model artifacts.

- `color_dominance_vgg16.h5`
- `color_dominance_vgg16_final.h5`
- `color_dominance_vgg16_finetuned.h5`
- `color_dominance_vgg16_finetuned_final.h5`
- `color_dominance_vgg16_improved.h5`
- `color_histogram_classifier.pkl`
- `color_histogram_label_encoder.pkl`
- `color_histogram_scaler.pkl`
- `color_history.pkl`
- `color_label_encoder.pkl`
- `color_training_history.pkl`
- `color_training_history_finetuned.pkl`

These model files store trained weights, encoders, scalers, and training history data.

## `docs/`

Legacy project documentation.

- `legacy/`
  - `QUICK_REFERENCE.md`
  - `readme.txt`
  - `README_VIVA.md`
  - `VIVA_SUMMARY.md`

## `mern/`

Currently empty; reserved for a potential MERN-stack app or future development.

---

### Notes

- The root `README.md` is the main project documentation file that should explain how to run the project, install dependencies, and use the backend and frontend.
- `backend/` and `training/` are Python-centric directories; `frontend/` is Node/React/Vite.
- `data/` holds datasets, while `models/` stores trained artifacts.
- `docs/legacy/` keeps older documentation files that are no longer the active README.

Use this structure document to understand where code, data, models, and documentation are organized in the project.