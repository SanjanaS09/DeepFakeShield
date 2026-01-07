# DeepFakeShield – Multimodal Deepfake Detection System

DeepFakeShield is an AI-powered detection system that analyzes images and videos to determine whether a face is real or manipulated, providing a prediction label and confidence score through a simple web interface.

## Features

- **Image deepfake detection** using CNN models such as ResNet18 and Xception trained on large-scale face datasets. 
- Web-based UI to upload media and instantly view prediction (REAL/FAKE) with confidence percentage. 
- REST API backend (Flask) for inference, integrated with a React frontend, supporting real-time progress updates and health checks.
- Explainable AI (XAI) components like heatmaps/attention visualization to highlight tampered facial regions. 
- Modular design prepared for multimodal extension (image, video, audio, and fused decision layer) as described in the system architecture.

## Tech Stack

- **Backend**: Python, Flask, PyTorch for model loading and inference, running inside a virtual environment.
- **Models**: ResNet18 and Xception CNNs for image-level detection, with training scripts supporting custom datasets, epochs, batch size, and learning rate.
- **Frontend**: React-based interface for file upload, result display, confidence gauge, and optional explainability controls.
- **Data**: Deepfake image/video datasets such as FaceForensics++ or DFDC, typically converted into image frames organized into train/validation/test folders for REAL and FAKE classes. 

## Installation and Setup

1. Create and activate virtual environment (backend)  
   ```bash
   cd backend
   python -m venv deepfake_env
   deepfake_env\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```   

2. Prepare dataset (optional if already trained)  
   - Download a deepfake dataset such as FaceForensics++ from Kaggle or official sources.  
   - Organize frames into `dataset/image/{train,validation,test}/{REAL,FAKE}` or use the provided extraction script. 

3. Train image model (if needed)  
   ```bash
   cd backend/training
   python train_image_model.py --dataset ../dataset/image --epochs 20 --batch-size 32 --learning-rate 0.0001
   ```  
   This saves the best model weights into the `models/` directory.

4. Start backend server  
   ```bash
   cd backend
   deepfake_env\Scripts\activate
   python app.py
   ```  
   The API will expose endpoints such as `/api/detect/image` and `/api/health`.

5. Start frontend  
   ```bash
   cd frontend
   npm install
   npm start
   ```  
   Open `http://localhost:3000` in a browser to access the DeepFakeShield UI.

## Usage

- Upload an image (or frame from a video) through the frontend upload form.
- The frontend sends the file to the Flask backend, which runs the selected CNN model and returns prediction, class probabilities, and an overall confidence score.
- The result window displays label (REAL/FAKE), confidence arc, and optionally an explanation/heatmap indicating manipulated regions.

## Future Work

- Extend from image-only detection to full multimodal support with dedicated video and audio models and a fusion layer. 
- Optimize inference for GPU deployment and add containerization (Docker) for easier cloud or on-premise installation in real-world platforms. 
```

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/143869999/315183ec-1230-4b80-9caa-4cb3c9250bae/image.jpg)
