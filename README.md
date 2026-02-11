# AI Voice Detection API

This project provides a connection between a Python FastAPI backend and a React frontend to detect AI-generated voices.

## Features
- **API Endpoint:** `/api/voice-detect`
- **Detection:** Classifies audio as "HUMAN" or "AI_GENERATED" with confidence scores and explanations.
- **Frontend:** Interactive UI to test the endpoint with sample audio or URLs.

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8+
- Node.js & npm

### 1. Backend Setup
Open a terminal in the root directory:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```
The backend will run on `http://127.0.0.1:8000`.

### 2. Frontend Setup
Open a new terminal in the `frontend` folder:
```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```
The frontend will run on `http://localhost:5173`.

## 🧪 Testing
1. Open the frontend URL (`http://localhost:5173`).
2. Use the "Test Endpoint" form.
3. You can use the included sample audio URLs:
   - `http://localhost:5173/human.wav`
   - `http://localhost:5173/ai.wav`
